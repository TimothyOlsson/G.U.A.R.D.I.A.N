use std::time::Instant;

use tracing::trace;
use ndarray::{Array1, Axis};
use itertools::multizip;
use ndarray::parallel::prelude::*;
use rayon::iter::ParallelBridge;

use crate::cpu::interface::Network;
use super::*;
use crate::node_global_to_local_index;

const NEURON_STATE_A: usize = 0;
const NEURON_STATE_B: usize = 1;
const NODE_A: usize = 2;
const NODE_B: usize = 3;
const DELTA_NODE: usize = 0;

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
    .for_each(|(neuron_a_index, (neuron_state, node_states, inter_connections))| {
        let node_index_offset = neuron_a_index * g_settings.n_nodes_per_neuron;
        let neuron_state = unpack_array(neuron_state);
        let precalculated_forward = model.precalculate(NEURON_STATE_A, neuron_state.view());
        let precalculated_forward = [&precalculated_forward];
        let precalculated_backward = model.precalculate(NEURON_STATE_B, neuron_state.view());
        let precalculated_backward = [&precalculated_backward];
        for (node_a_local_index, node_a) in node_states.outer_iter().enumerate() {
            let mut node_a = unpack_array(node_a);
            let connection_a = inter_connections.get(node_a_local_index).unwrap();
            let node_a_global_index = node_a_local_index + node_index_offset;
            let node_b_global_index = connection_a.get_index();
            let (neuron_b_index, node_b_local_index) = node_global_to_local_index(node_b_global_index, g_settings);
            let connection_b = get_inter_connection(neuron_b_index, node_b_local_index, inter_connections_source);
            let is_connected = check_is_connected(node_a_global_index, connection_b);
            let (neuron_state_b,  mut node_b) = if is_connected {
                if node_b_global_index > node_a_global_index { continue; }  // Only highest index calculates if connected
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
                (NEURON_STATE_B, expand(neuron_state_b.view())),
                (NODE_A, expand(node_a.view())),
                (NODE_B, expand(node_b.view()))
            ];
            let delta_node_a = &model.forward_from_precalc(&inputs, &precalculated_forward)[DELTA_NODE];
            node_a = node_a + squeeze(delta_node_a.view());

            // If not connected, this could be skipped
            if is_connected {
                let inputs = [
                    (NEURON_STATE_A, expand(neuron_state_b.view())),
                    (NODE_A, expand(node_b.view())),
                    (NODE_B, expand(node_a.view()))
                ];
                let delta_node_b = &model.forward_from_precalc(&inputs, &precalculated_backward)[DELTA_NODE];
                node_b = node_b + squeeze(delta_node_b.view());

            }

            let node_a = pack_array(node_a);
            let node_b = pack_array(node_b);

            // Allows for multiple process to work with the same vector without locking
            let ptr = nodes.as_ptr() as *mut u8;
            let to_write = if is_connected {
                vec![
                    (node_a_global_index, node_a),
                    (node_b_global_index, node_b)
                ]
            } else {
                vec![
                    (node_a_global_index, node_a),
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