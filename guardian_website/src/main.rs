use std::net::SocketAddr;
use std::process::exit;

use axum::{
    routing::get,
    response::{Response, IntoResponse},
    extract::Path,
    body::{Empty, Full},
    http::StatusCode,
    Router,
};
use include_dir::{include_dir, Dir};


// Load all files into binary, to speed up file serving
static ASSETS_DIR: Dir<'_> = include_dir!("$CARGO_MANIFEST_DIR/../assets");
static RESOURCES_DIR: Dir<'_>  = include_dir!("$CARGO_MANIFEST_DIR/resources");
static PKG_DIR: Dir<'_> = include_dir!("$CARGO_MANIFEST_DIR/pkg");
static WORKER_POLYFILL: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/workers-polyfill.js"));


#[tokio::main]
async fn main() {

    // initialize tracing
    tracing_subscriber::fmt::init();

    // build our application with routes
    let app = Router::new()
        .route("/", get(index))

        // Requires special headings
        .route("/resources/*path", get(resources_static_file_serve))
        .route("/pkg/*path", get(pkg_static_file_serve))
        .route("/assets/*path", get(assets_static_file_serve))
        .route("/workers-polyfill.js", get(workers_polyfill_file_serve))

    ;

    // Bind and start
    tracing::info!("Starting server");
    let addr = SocketAddr::from(([0, 0, 0, 0], 8000));
    tracing::debug!("listening on {}", addr);
    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .with_graceful_shutdown(shutdown_signal())
        .await
        .unwrap();
}

/// A signal for a graceful shutdown of the server
async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("Expect shutdown signal handler");
    tracing::info!("Signal shutdown");
    exit(0)
}


// region: Static File Serving

/// Serve the index file
async fn index() -> impl IntoResponse  {
    static_file_serve("index.html".to_string(), &RESOURCES_DIR).await
}

/// Must be at top level for it to work :(
/// https://bugzilla.mozilla.org/show_bug.cgi?id=1247687
/// https://bugzilla.mozilla.org/show_bug.cgi?id=1247687
/// https://caniuse.com/?search=module%20worker
async fn workers_polyfill_file_serve() -> impl IntoResponse  {
    Response::builder()
        .status(StatusCode::OK)
        .header("Cross-Origin-Embedder-Policy", "require-corp")
        .header("Cross-Origin-Opener-Policy", "same-origin")
        .header("Content-Type", "application/javascript")
        .body(Full::from(WORKER_POLYFILL))
        .unwrap()
        .into_response()
}

/// Wrapper for resources directory
async fn resources_static_file_serve(Path(path): Path<String>) -> impl IntoResponse {
    static_file_serve(path, &RESOURCES_DIR).await
}

/// Wrapper for pkg directory
async fn pkg_static_file_serve(Path(path): Path<String>) -> impl IntoResponse {
    static_file_serve(path, &PKG_DIR).await
}

/// Wrapper for assets directory
async fn assets_static_file_serve(Path(path): Path<String>) -> impl IntoResponse {
    static_file_serve(path, &ASSETS_DIR).await
}

/// Wrapper for resources directory
async fn static_file_serve(path: String, static_dir: &Dir<'static>) -> impl IntoResponse {
    let path = path.trim_start_matches('/');
    let mime_type = mime_guess::from_path(path)
                               .first_or_text_plain()
                               .to_string();
    let response = match static_dir.get_file(path) {
        None => {
            Response::builder()
                .status(StatusCode::NOT_FOUND)
                .body(Empty::new())
                .unwrap()
                .into_response()
        },
        Some(f) => {
            tracing::debug!("{f:?} found!");
            Response::builder()
                .status(StatusCode::OK)
                .header("Cross-Origin-Embedder-Policy", "require-corp")
                .header("Cross-Origin-Opener-Policy", "same-origin")
                .header("Content-Type", mime_type)
                .body(Full::from(f.contents()))
                .unwrap()
                .into_response()
        }
    };
    response
}

// endregion
