use std::time::Instant;

use tracing::trace;
use ndarray::{s, Axis};
use itertools::multizip;
use ndarray::parallel::prelude::*;
use rayon::iter::ParallelBridge;
use rayon::ThreadPool;

use crate::cpu::interface::Network;
use crate::cpu::*;

// Input
const NEURON_STATE: usize = 0;
const NODE_SELF: usize = 1;
const NODE_OTHER: usize = 2;

// Output
const DELTA_NODE_SELF: usize = 0;
const DELTA_NODE_OTHER: usize = 1;

pub fn update(network: &mut Network, pool: &ThreadPool) {
    let nodes = &mut network.state.nodes;
    let neuron_states = &network.state.neuron_states;
    let intra_connections = &network.state.intra_connections;
    let g_settings = &network.g_settings;
    let genome = &network.genome;
    let model = &genome.intraconnected_node_state_update;

    let zipped = multizip(
        (
            neuron_states.rows(),
            nodes.axis_iter_mut(Axis(0)),
            intra_connections.axis_iter(Axis(0)),
        )
    );

    let now = Instant::now();
    let operation = zipped
    .into_iter()
    .par_bridge()
    .for_each(|(neuron_state, mut node_states_source, intra_connections)| {
        let neuron_state = unpack_array(neuron_state);
        let precalculated_neuron_state = model.precalculate(NEURON_STATE, neuron_state.view());
        let node_states = unpack_array(node_states_source.view());
        let mut delta_node_states_min = Array2::from_elem((g_settings.n_nodes_per_neuron, g_settings.node_size), 0.0);
        let mut delta_node_states_max = Array2::from_elem((g_settings.n_nodes_per_neuron, g_settings.node_size), 0.0);
        for (node_local_index_self, node_state_self) in node_states.rows().into_iter().enumerate() {
            let precalculated_node = model.precalculate(NODE_SELF, node_state_self);
            let connections = intra_connections.row(node_local_index_self);
            let precalculated = [
                &precalculated_neuron_state,
                &precalculated_node
            ];
            for connection in connections {
                let node_local_index_other = connection.get_index();
                let node_state_other = node_states.slice(s![node_local_index_other, ..]);

                let inputs = [
                    (NODE_OTHER, expand(node_state_other))
                ];

                let output = model.forward_from_precalc(&inputs, &precalculated);

                let delta_node_self = squeeze(output[DELTA_NODE_SELF].view());
                min_array_inplace(&mut delta_node_states_min.row_mut(node_local_index_self), delta_node_self);
                max_array_inplace(&mut delta_node_states_max.row_mut(node_local_index_self), delta_node_self);

                let delta_node_other = squeeze(output[DELTA_NODE_OTHER].view());
                min_array_inplace(&mut delta_node_states_min.row_mut(node_local_index_other), delta_node_other);
                max_array_inplace(&mut delta_node_states_max.row_mut(node_local_index_other), delta_node_other);
            }
        }
        let delta_node_states = delta_node_states_max + delta_node_states_min;
        let updated_node_states = node_states + delta_node_states;
        let updated_node_states = pack_array(updated_node_states);
        node_states_source.assign(&updated_node_states);
    });
    pool.install(|| operation);
    trace!("It took {:?} to update intraconnection states", now.elapsed());
}
