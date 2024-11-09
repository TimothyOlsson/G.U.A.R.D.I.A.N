use tokio::main;
use tracing::{debug, info, Level};
use tracing_subscriber::FmtSubscriber;
use rayon::ThreadPoolBuilder;
use rand::SeedableRng;

use glib::cpu::interface::{Genome, State, Network};
use glib::cpu::process::update;
use glib::{get_network_size, GuardianSettings, NetworkSettings};
use glib::visualization;

#[main]
async fn main() {
    let subscriber = FmtSubscriber::builder()
    .with_max_level(Level::INFO)
    .finish();
    tracing::subscriber::set_global_default(subscriber).expect("setting default subscriber failed");
    info!("RUNNING");
    //let _ = test_webgpu().await;

    let thread_count = std::thread::available_parallelism().unwrap().get();
    let pool = ThreadPoolBuilder::new().num_threads(thread_count).build().unwrap();
    let g_settings = GuardianSettings::downlevel_default();
    let n_settings = NetworkSettings::downlevel_default();
    get_network_size(&g_settings, &n_settings);
    let rng = rand::rngs::StdRng::seed_from_u64(1);
    let genome = Genome::new(&g_settings, &n_settings, Some(rng.clone()));
    let mut state = State::new(&g_settings, &n_settings);
    debug!("Randomizing");
    state.randomize(&g_settings, &n_settings, Some(rng));
    debug!("Randomizing done");
    //debug!("\n{:#?}", state.nodes);
    //debug!("\n{:#?}", state.neuron_states);
    //debug!("\n{:#?}", state.inter_connections);
    //debug!("\n{:#?}", state.inter_connections_flags);
    //debug!("\n{:#?}", state.intra_connections);
    let mut network = Network {
        state,
        genome,
        g_settings: g_settings.clone(),
        n_settings: n_settings.clone(),
    };

    let mut state_history = vec![network.state.clone()];
    for i in 0..64 {
        info!("Iteration {i}");
        update(&mut network, &pool);
        debug!("Nodes \n{:#?}", network.state.nodes);
        debug!("NeuronStates \n{:#?}", network.state.neuron_states);
        state_history.push(network.state.clone());
    }
    visualization::graph::visualize_network(state_history, &g_settings, &n_settings);
}