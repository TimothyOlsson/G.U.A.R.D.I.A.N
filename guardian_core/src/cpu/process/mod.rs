use rayon::ThreadPool;
use tracing::trace;

use crate::cpu::interface::Network;

pub mod interconnection_state;
pub mod intraconnection_state;
pub mod neuron_state;
pub mod interconnection_plasticity;
pub mod intraconnection_plasticity;
pub mod io_ports;

/// WIP: Starting with a naive approach
/// TODO: Move to network as impl?
pub fn update(network: &mut Network, pool: &ThreadPool) {
    trace!("Stage 1: Update io ports (network + input if core)");
    io_ports::update(network, pool);
    trace!("Stage 2: Update interconnected state");
    interconnection_state::update(network, pool);
    trace!("Stage 3: Update intraconnected state");
    intraconnection_state::update(network, pool);
    trace!("Stage 4: Update neuron state");
    neuron_state::update(network, pool);
    trace!("Stage 5: Update interconnections (plasticity)");
    interconnection_plasticity::update(network, pool);
    trace!("Stage 6: Update intraconnections (plasticity)");
    intraconnection_plasticity::update(network, pool);
    trace!("Stage 7: Update IO ports");
    io_ports::update(network, pool);
}
