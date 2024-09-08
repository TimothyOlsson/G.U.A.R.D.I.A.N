
use tracing::info;
use futures::channel::oneshot;
use rayon::prelude::*;

pub async fn run(thread_pool: Option<rayon::ThreadPool>) {
    info!("Hello!");
    let thread_pool = thread_pool.unwrap();
    let value = thread_pool.install(|| {
        let value = vec![1;2];
        let _new: Vec<u8> = value
            .par_iter()
            .map(|x| x + 1)
            .collect();
        1.0
        }
    );
    info!("Recieved value {value}");

    let mut rxs = vec![];
    let now = instant::Instant::now();
    for _ in 0..9 {
        let (tx, rx) = oneshot::channel();
        thread_pool.spawn(move || {
            std::thread::sleep(std::time::Duration::from_secs(1));
            tx.send(0).unwrap();
        });
        rxs.push(rx);
    }
    for rx in rxs {
        let _ = async move {
            info!("WAITING");
            match rx.await {
                Ok(_data) => info!("WORKED!"),
                Err(_) => info!("FAILED!"),
            }
            info!("PASSED");
        }.await;
    }
    info!("Elapsed {:?}", now.elapsed());
}
