use std::sync::Arc;

use axum::{
    Extension, Json,
    extract::ws::{WebSocket, WebSocketUpgrade},
    response::IntoResponse,
    response::Response,
};

type VadReturnTx =
    tokio::sync::oneshot::Sender<Result<Vec<silero_vad_jit::SpeechTimestamp>, String>>;
type VadTX = tokio::sync::mpsc::Sender<(Vec<f32>, VadReturnTx)>;
#[allow(unused)]
type VadRx = tokio::sync::mpsc::Receiver<(Vec<f32>, VadReturnTx)>;

#[derive(Debug, Clone)]
pub struct VadService {
    tx: VadTX,
}

impl VadService {
    // TODO: multiple-threaded support
    pub fn new(model_path: &str, buffer: usize) -> anyhow::Result<Self> {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<(Vec<f32>, VadReturnTx)>(buffer);
        let vad = silero_vad_jit::VadModelJit::init_jit_model(
            model_path,
            silero_vad_jit::tch::Device::cuda_if_available(),
        )?;
        let mut model = silero_vad_jit::SileroVad::from(vad);

        let _handle = tokio::task::spawn_blocking(move || {
            while let Some((audio, return_tx)) = rx.blocking_recv() {
                let params = silero_vad_jit::VadParams {
                    sampling_rate: 16000,
                    ..Default::default()
                };
                match model.get_speech_timestamps(audio, params, None) {
                    Ok(timestamps) => {
                        let _ = return_tx.send(Ok(timestamps));
                    }
                    Err(e) => {
                        log::error!("VAD processing error: {}", e);
                        let _ = return_tx.send(Err("VAD processing error".to_string()));
                    }
                }
            }
        });
        Ok(VadService { tx })
    }

    pub async fn detect_wav(
        &self,
        audio: Vec<u8>,
    ) -> anyhow::Result<Vec<silero_vad_jit::SpeechTimestamp>> {
        let mut reader = wav_io::reader::Reader::from_vec(audio)?;
        let header = reader.read_header()?;
        let mut samples = reader.get_samples_f32()?;
        if header.channels != 1 {
            samples = wav_io::utils::stereo_to_mono(samples)
        }
        if header.sample_rate != 16000 {
            samples = wav_io::resample::linear(samples, 1, header.sample_rate, 16000)
        }

        self.detect_audio_16k(samples).await
    }

    pub async fn detect_audio_16k(
        &self,
        audio: Vec<f32>,
    ) -> anyhow::Result<Vec<silero_vad_jit::SpeechTimestamp>> {
        let (return_tx, return_rx) = tokio::sync::oneshot::channel();
        self.tx
            .send((audio, return_tx))
            .await
            .map_err(|_| anyhow::anyhow!("Failed to send audio for VAD processing"))?;

        match return_rx.await {
            Ok(Ok(timestamps)) => Ok(timestamps),
            Ok(Err(e)) => Err(anyhow::anyhow!(e)),
            Err(_) => Err(anyhow::anyhow!("Failed to receive VAD result")),
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub struct SpeechSampleIndex {
    pub start: i64,
    pub end: i64,
}

impl From<silero_vad_jit::SpeechTimestamp> for SpeechSampleIndex {
    fn from(timestamp: silero_vad_jit::SpeechTimestamp) -> Self {
        SpeechSampleIndex {
            start: timestamp.start,
            end: timestamp.end,
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub struct VadResponse {
    pub timestamps: Vec<SpeechSampleIndex>,
}

pub async fn vad_detect(
    vad_service: Extension<VadService>,
    mut multipart: axum::extract::Multipart,
) -> impl IntoResponse {
    while let Ok(Some(field)) = multipart.next_field().await {
        log::debug!("Processing field: {:?}", field.name());
        if field.name() == Some("audio") {
            if let Ok(audio) = field.bytes().await {
                log::debug!("Received audio field with {} bytes", audio.len());
                match vad_service.detect_wav(audio.to_vec()).await {
                    Ok(timestamps) => {
                        let response = VadResponse {
                            timestamps: timestamps
                                .into_iter()
                                .map(SpeechSampleIndex::from)
                                .collect(),
                        };
                        return Json(serde_json::to_value(response).unwrap());
                    }
                    Err(e) => {
                        log::error!("VAD processing error: {}", e);
                        return Json(serde_json::json!({
                            "error": "VAD processing error"
                        }));
                    }
                }
            }
        }
    }
    Json(serde_json::json!({
        "error": "No audio provided"
    }))
}

#[derive(Debug, Clone)]
pub struct VadFactory {
    pub model_path: String,
}

impl VadFactory {
    pub fn new(model_path: String) -> Self {
        VadFactory { model_path }
    }

    pub fn create(
        &self,
    ) -> anyhow::Result<silero_vad_jit::StreamingVad<silero_vad_jit::VadModelJit>> {
        let vad = silero_vad_jit::VadModelJit::init_jit_model(
            &self.model_path,
            silero_vad_jit::tch::Device::cuda_if_available(),
        )?;

        let params = silero_vad_jit::VadParams {
            sampling_rate: 16000,
            ..Default::default()
        };
        let model = silero_vad_jit::SileroVad::from(vad);

        Ok(silero_vad_jit::StreamingVad::new(model, params))
    }
}

pub async fn websocket_handler(
    ws: WebSocketUpgrade,
    vad_factory: Extension<Arc<VadFactory>>,
) -> Response {
    let session = vad_factory.create();
    if session.is_err() {
        return Response::builder()
            .status(500)
            .body("Failed to create VAD session".into())
            .unwrap();
    }
    let session = session.unwrap();
    ws.on_upgrade(move |socket| handle_websocket(socket, session))
}

// todo: this function is a computationally intensive task, needs to be rewritten
async fn handle_websocket(
    mut socket: WebSocket,
    mut session: silero_vad_jit::StreamingVad<silero_vad_jit::VadModelJit>,
) {
    log::info!("WebSocket connection established");

    let mut ret = bytes::BytesMut::new();
    let windows_size_samples = session.get_window_size_samples();

    async fn process_audio(
        session: &mut silero_vad_jit::StreamingVad<silero_vad_jit::VadModelJit>,
        socket: &mut WebSocket,
        data: &[u8],
    ) -> anyhow::Result<()> {
        let audio_16k = data
            .chunks_exact(2)
            .map(|chunk| {
                let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                sample as f32 / i16::MAX as f32
            })
            .collect::<Vec<f32>>();
        match session.process_chunk(&audio_16k) {
            Ok(Some(silero_vad_jit::VadEvent::SpeechStart)) => {
                let _ = socket
                    .send(axum::extract::ws::Message::Text(
                        serde_json::json!({
                            "event": "speech_start",
                        })
                        .to_string()
                        .into(),
                    ))
                    .await?;
                Ok(())
            }
            Ok(Some(silero_vad_jit::VadEvent::SpeechEnd)) => {
                let _ = socket
                    .send(axum::extract::ws::Message::Text(
                        serde_json::json!({
                            "event": "speech_end",
                        })
                        .to_string()
                        .into(),
                    ))
                    .await?;
                Ok(())
            }
            Ok(None) => Ok(()),
            Err(e) => {
                log::error!("VAD processing error: {}", e);
                let _ = socket
                    .send(axum::extract::ws::Message::Text(
                        serde_json::json!({
                            "error": "VAD processing error",
                            "message": e.to_string()
                        })
                        .to_string()
                        .into(),
                    ))
                    .await?;
                Err(e.into())
            }
        }
    }

    while let Some(Ok(msg)) = socket.recv().await {
        match msg {
            axum::extract::ws::Message::Text(text) => {
                log::info!("Received text message: {}", text);
                if text == "reset" {
                    session.reset();
                    log::info!("VAD session reset");
                }
            }
            axum::extract::ws::Message::Binary(mut data) => {
                let reminder = windows_size_samples * 2 - ret.len();
                if reminder < windows_size_samples * 2 {
                    ret.extend_from_slice(&data[..reminder]);
                    data = data.slice(reminder..);

                    assert_eq!(ret.len(), windows_size_samples * 2);
                    let r = process_audio(&mut session, &mut socket, &ret).await;
                    if let Err(e) = r {
                        log::error!("Failed to process audio: {}", e);
                        break;
                    }
                    ret.clear();
                }

                for chunk in data.chunks(windows_size_samples * 2) {
                    if chunk.len() < windows_size_samples * 2 {
                        ret.extend_from_slice(chunk);
                        continue;
                    }

                    let r = process_audio(&mut session, &mut socket, chunk).await;
                    if let Err(e) = r {
                        log::error!("Failed to process audio: {}", e);
                        break;
                    }
                }
            }
            axum::extract::ws::Message::Close(_) => {
                log::info!("WebSocket connection closed");
                break;
            }
            _ => {}
        }
    }
}
