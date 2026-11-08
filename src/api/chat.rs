use axum::{Router, routing::get};

pub async fn chat(host: String, port: u16) {
    let server = format!("{}:{}", host, port);
    let app = Router::new().route("/v1/syn", get(|| async { "syn ack" }));
    let listener = tokio::net::TcpListener::bind(server.clone()).await.unwrap();
    println!("Listening on {}", server);
    axum::serve(listener, app).await.unwrap();
}
