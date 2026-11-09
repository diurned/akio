use crate::inference::llm::LlamaCpp;

use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

pub async fn chat(host: String, port: u16, model: String) {
    let mut llama_cpp = LlamaCpp::new();
    llama_cpp.load(model);

    let app = Router::new()
        .route("/v1/syn", get(|| async { "syn ack" }))
        .route("/v1/chat/completions", post(generate))
        .with_state(Arc::new(Mutex::new(llama_cpp)));

    let server = format!("{}:{}", host, port);
    let listener = tokio::net::TcpListener::bind(&server).await.unwrap();
    println!("Listening on {}", server);
    axum::serve(listener, app).await.unwrap();
}

async fn generate(
    State(llama_cpp): State<Arc<Mutex<LlamaCpp>>>,
    Json(payload): Json<ChatRequest>,
) -> (StatusCode, Json<ChatResponse>) {
    let prompt = payload.prompt;
    let response = tokio::task::spawn_blocking(move || {
        let mut llm = llama_cpp.lock().expect("llm mutex poisoned");
        llm.complete(prompt)
    })
    .await
    .expect("inference task panicked");

    (StatusCode::OK, Json(ChatResponse { response }))
}

#[derive(Deserialize)]
struct ChatRequest {
    prompt: String,
}

#[derive(Serialize)]
struct ChatResponse {
    response: String,
}
