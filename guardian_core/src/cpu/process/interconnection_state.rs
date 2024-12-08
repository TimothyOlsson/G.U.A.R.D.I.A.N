use std::time::Instant;

use tracing::trace;
use ndarray::{Array1, Axis};
use itertools::multizip;
use ndarray::parallel::prelude::*;
use rayon::iter::ParallelBridge;
use rayon::ThreadPool;

use crate::cpu::interface::Network;
use crate::cpu::*;

// Input
const NEURON_STATE_SELF: usize = 0;
const NEURON_STATE_OTHER: usize = 1;
const NODE_STATE_SELF: usize = 2;
const NODE_STATE_OTHER: usize = 3;

// Output
const DELTA_NODE_STATE_SELF: usize = 0;

pub fn update(network: &mut Network, pool: &ThreadPool) {
    let nodes = &network.state.nodes;
    let neuron_states = &network.state.neuron_states;
    let inter_connections_source = &network.state.inter_connections;
    let genome = &network.genome;

    let g_settings = &network.g_settings;
    let model = &genome.interconnected_node_state_update;

    let zipped = multizip(
        (
            neuron_states.outer_iter(),
            nodes.axis_iter(Axis(0)),
            inter_connections_source.axis_iter(Axis(0))
        )
    );


    let now = Instant::now();
    let operation = zipped
    .into_iter()
    .enumerate()
    .par_bridge()
    .for_each(|(neuron_index_self, (neuron_state, node_states, inter_connections))| {
        let node_index_offset = neuron_index_self * g_settings.n_nodes_per_neuron;
        let neuron_state = unpack_array(neuron_state);
        let precalculated_forward = model.precalculate(NEURON_STATE_SELF, neuron_state.view());
        let precalculated_forward = [&precalculated_forward];
        let precalculated_backward = model.precalculate(NEURON_STATE_OTHER, neuron_state.view());
        let precalculated_backward = [&precalculated_backward];
        for (node_local_index_self, node_state_self) in node_states.outer_iter().enumerate() {
            let mut node_state_self = unpack_array(node_state_self);
            let connection_self = inter_connections.get(node_local_index_self).unwrap();
            let node_global_index_self = node_local_index_self + node_index_offset;
            let node_global_index_other = connection_self.get_index();
            let (neuron_b_index, node_b_local_index) = node_global_to_local_index(node_global_index_other, g_settings);
            let connection_other = get_inter_connection(neuron_b_index, node_b_local_index, inter_connections_source);
            let is_connected = check_is_connected(node_global_index_self, connection_other);
            let (neuron_state_other,  mut node_state_other) = if is_connected {
                if node_global_index_other > node_global_index_self { continue; }  // Only highest index calculates if connected
                (
                    get_neuron_state(neuron_b_index, neuron_states),
                    get_node(neuron_b_index, node_b_local_index, nodes)
                )
            } else {
                // NOTE: Could skip also, but then the node would behave as a intra-node
                // Could skip 0 input for faster processing!
                (
                    Array1::zeros(g_settings.neuron_state_size),
                    Array1::zeros(g_settings.node_size)
                )
            };

            // Calculate forward
            let inputs = [
                (NEURON_STATE_OTHER, expand(neuron_state_other.view())),
                (NODE_STATE_SELF, expand(node_state_self.view())),
                (NODE_STATE_OTHER, expand(node_state_other.view()))
            ];
            let delta_node_self = &model.forward_from_precalc(&inputs, &precalculated_forward)[DELTA_NODE_STATE_SELF];
            node_state_self = node_state_self + squeeze(delta_node_self.view());

            // If not connected, this could be skipped
            if is_connected {
                let inputs = [
                    (NEURON_STATE_SELF, expand(neuron_state_other.view())),
                    (NODE_STATE_SELF, expand(node_state_other.view())),
                    (NODE_STATE_OTHER, expand(node_state_self.view()))
                ];
                let delta_node_other = &model.forward_from_precalc(&inputs, &precalculated_backward)[DELTA_NODE_STATE_SELF];
                node_state_other = node_state_other + squeeze(delta_node_other.view());
            }

            let node_state_self = pack_array(node_state_self);
            let node_state_other = pack_array(node_state_other);

            // Allows for multiple process to work with the same vector without locking
            let ptr = nodes.as_ptr() as *mut u8;
            let to_write = if is_connected {
                vec![
                    (node_global_index_self, node_state_self),
                    (node_global_index_other, node_state_other)
                ]
            } else {
                vec![
                    (node_global_index_self, node_state_self),
                ]
            };
            let row_size = g_settings.node_size;
            for (node_global_index, node_state) in to_write {
                let offset = node_global_index * row_size;
                let data = &node_state.as_slice().unwrap();
                unsafe { non_locking_write(ptr, offset, row_size, data); }
            }
        }
    });

    pool.install(|| operation);

    trace!("It took {:?} to update interconnection states", now.elapsed());
}