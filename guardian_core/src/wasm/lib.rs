
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub mod guardian;

// region: WASM

#[cfg(target_arch = "wasm32")]
use {
    wasm_bindgen::prelude::*,
    wasm_bindgen_futures::spawn_local,
    guardian_gui,
    guardian_io,
};

#[cfg(target_arch = "wasm32")]
pub mod wasm_pool;

/// This is the entry-point for all the web-assembly.
/// This is called once from the HTML.
/// It loads the app, installs some callbacks, then returns.
/// You can add more callbacks like this if you want to call in to your code.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub async fn wasm_start(
    can_use_webgpu: bool,
    n_threads:usize,
    pool: Option<wasm_pool::WorkerPool>)
{
    console_error_panic_hook::set_once();
    tracing_wasm::set_as_global_default();
    //guardian_io::webrtc_connection::reconnect_test_receiver_disconnected().await;
    //guardian_io::webrtc_connection::reconnect_test_caller_disconnected().await;
    match pool {
        Some(pool) => {
            let thread_pool = rayon::ThreadPoolBuilder::new()
                .num_threads(n_threads)
                .spawn_handler(|thread| Ok(pool.run(|| thread.run()).unwrap()))
                .build()
                .unwrap();

            pool.run(move || { 
                // Run as async
                spawn_local(async {
                    guardian::run(Some(thread_pool)).await;
                });
            }).unwrap();
        }
        None => {
            spawn_local(async {
                guardian::run(None).await;
            });
        }
    }

    // This needs to be main thread. Do not do it in a worker!
    guardian_gui::main("guardian_ai");
}

// endregion