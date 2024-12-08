use std::time::Instant;

use tracing::trace;
use ndarray::{Array1, Array2, Array3, Axis};
use itertools::multizip;
use ndarray::parallel::prelude::*;
use rayon::iter::ParallelBridge;
use rayon::ThreadPool;

use crate::cpu::interface::Network;
use crate::{GuardianSettings, InterConnection};
use crate::cpu::model::Model;
use crate::cpu::*;

// Input
const NEURON_STATE_SELF: usize = 0;
const NEURON_STATE_OTHER: usize = 1;
const NODE_STATE_SELF: usize = 2;
const NODE_STATE_OTHER: usize = 3;
const FORCE_SELF: usize = 4;
const FORCE_OTHER: usize = 5;

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

        // Done here, so it can be used for all nodes in the neuron

        // self -> other
        let precalculated_forward = model.precalculate(NEURON_STATE_SELF, neuron_state.view());
        let precalculated_forward = &precalculated_forward;

        // other -> self
        let precalculated_backward = model.precalculate(NEURON_STATE_OTHER, neuron_state.view());
        let precalculated_backward = &precalculated_backward;

        for (node_local_index_self, node_state_self) in node_states.outer_iter().enumerate() {

            // Get self
            let node_state_self = unpack_array(node_state_self);
            let connection_self = inter_connections.get(node_local_index_self).unwrap();
            let node_global_index_self = node_local_index_self + node_index_offset;

            update_node_state(
                node_global_index_self,
                node_state_self,
                connection_self,
                precalculated_forward,
                precalculated_backward,
                model,
                nodes,
                neuron_states,
                inter_connections_source,
                g_settings
            );
        }
    });
    trace!("It took {:?} to run interconnected node state update", now.elapsed());
    pool.install(|| operation);
}


/// This updates the state and main connections, thus pending cannot be done here
fn update_node_state(
    node_global_index_self: usize,
    mut node_state_self: Array1<f32>,
    connection_self: &InterConnection,
    precalculated_forward: &Array1<f32>,
    precalculated_backward: &Array1<f32>,
    model: &Model,
    nodes: &Array3<u8>,
    neuron_states: &Array2<u8>,
    inter_connections: &Array2<InterConnection>,
    g_settings: &GuardianSettings,
) {

    // Get other
    let node_global_index_other = connection_self.get_index();
    let (neuron_b_index, node_b_local_index) = node_global_to_local_index(node_global_index_other, g_settings);
    let connection_other = get_inter_connection(neuron_b_index, node_b_local_index, inter_connections);

    let is_connected = check_is_connected(node_global_index_self, connection_other);
    let (neuron_state_other,  mut node_state_other) = if is_connected {
        if node_global_index_other > node_global_index_self { return; }  // Only highest index calculates if connected
        (
            get_neuron_state(neuron_b_index, neuron_states),
            get_node(neuron_b_index, node_b_local_index, nodes)
        )
    } else {
        // NOTE: Could skip also, but then the node would behave as a intra-node
        // Could skip 0 input for faster processing!
        connection_self.reset_main();
        (
            Array1::zeros(g_settings.neuron_state_size),
            Array1::zeros(g_settings.node_size)
        )
    };

    let (force_self, force_other) = connection_self.get_forces();  // Copies them here
    let (force_self_arr, force_other_arr) = (value_to_array(force_self), value_to_array(force_other));

    let node_state_self_clone = node_state_self.clone();

    // Calculate forward
    let inputs = [
        (NEURON_STATE_OTHER, expand(neuron_state_other.view())),
        (NODE_STATE_SELF, expand(node_state_self_clone.view())),
        (NODE_STATE_OTHER, expand(node_state_other.view())),
        (FORCE_SELF, force_self_arr.view()),
        (FORCE_OTHER, force_other_arr.view()),
    ];
    let output = &model.forward_from_precalc(&inputs, precalculated_forward);
    let delta_node_self = &output[DELTA_NODE_STATE_SELF];
    node_state_self = node_state_self + squeeze(delta_node_self.view());
    drop(inputs);

    unsafe {
        write_node_non_locking_write(nodes, node_state_self, node_global_index_other, g_settings);
    }

    // If not connected, this could be skipped
    if !is_connected {
        return;
    }

    // Calculate backward (flipped)
    let inputs = [
        (NEURON_STATE_SELF, expand(neuron_state_other.view())),
        (NODE_STATE_SELF, expand(node_state_other.view())),
        (NODE_STATE_OTHER, expand(node_state_self_clone.view())),
        (FORCE_SELF, force_other_arr.view()),
        (FORCE_OTHER, force_self_arr.view()),
    ];

    let delta_node_other = &model.forward_from_precalc(&inputs, precalculated_backward)[DELTA_NODE_STATE_SELF];
    node_state_other = node_state_other + squeeze(delta_node_other.view());

    // Write backwards
    unsafe {
        write_node_non_locking_write(nodes, node_state_other, node_global_index_other, g_settings);
    }
}


unsafe fn write_node_non_locking_write(
    nodes: &Array3<u8>,
    node_state: Array1<f32>,
    node_global_index: usize,
    g_settings: &GuardianSettings
) {
    // Write forward
    let ptr = nodes.as_ptr() as *mut u8;
    let node_state_packed = pack_array(node_state);
    let offset = node_global_index * g_settings.node_size;
    let data = node_state_packed.as_slice().unwrap();
    unsafe { non_locking_write(ptr, offset, g_settings.node_size, data); }
}