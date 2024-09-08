use tracing::debug;
use guardian_gui;

pub mod guardian;

#[tokio::main]
async fn main() {
    // Log to stdout (if you run with `RUST_LOG=debug`).
    tracing_subscriber::fmt::init();

    let n_threads = num_cpus::get();
    debug!("Creating threadpool with {n_threads}");
    let thread_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .unwrap();

    // Start a new thread, where we will run the GUARDIAN
    debug!("Starting GUARDIAN thread");
    tokio::spawn(async {
        guardian::run(Some(thread_pool)).await;  // Will wait forever
    });

    // Blocking, needs to run on the main thread!
    // Alternative would be to have a CLI and skip this step
    debug!("Starting GUARDIAN GUI");
    guardian_gui::main("GUARDIAN-AI");
}