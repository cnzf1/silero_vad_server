use axum::{Extension, Router, extract::DefaultBodyLimit, routing::post};
use clap::Parser;

mod vad;

#[derive(Debug, Parser, Clone)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value = "127.0.0.1:8000", env("VAD_LISTEN"))]
    listen: String,
    #[arg(
        short,
        long,
        default_value = "silero_vad.jit",
        env("SILERO_MODEL_PATH")
    )]
    model_path: String,
}

#[tokio::main]
async fn main() {
    dotenv::dotenv().ok();

    env_logger::init();
    // build our application

    let args = Args::parse();

    let vad_service =
        vad::VadService::new(&args.model_path, 128).expect("Failed to create VAD service");

    let app = app(vad_service);

    log::info!("Listening on {}", args.listen);
    let listener = tokio::net::TcpListener::bind(args.listen).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

fn app(vad_service: vad::VadService) -> Router {
    // build our application with a route
    Router::new()
        .route("/v1/audio/vad", post(vad::vad_detect))
        .layer(DefaultBodyLimit::max(10 * 1024 * 1024)) // 10 MB limit
        .layer(Extension(vad_service))
}
