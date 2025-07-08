use axum::{Extension, Json, response::IntoResponse};

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
