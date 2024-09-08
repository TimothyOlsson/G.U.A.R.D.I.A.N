use std::time::Instant;

use ndarray::{Array1, Array2, Axis};
use itertools::multizip;
use ndarray::parallel::prelude::*;
use rayon::iter::ParallelBridge;

use crate::cpu::model::Model;
use crate::cpu::interface::{IntraConnection, Network};
use super::*;
use crate::GuardianSettings;

const NEURON_STATE: usize = 0;
const NODE_A: usize = 1;
const NODE_B: usize = 2;
const STRENGTH: usize = 3;
const PUSHBACK: usize = 4;

pub fn update_intra_connections(network: &mut Network) {
    let nodes = &network.state.nodes;
    let neuron_states = &network.state.neuron_states;
    let intra_connections = &network.state.intra_connections;
    let genome = &network.genome;
    let model = &genome.intraconnections_update;
    let g_settings = &network.g_settings;

    let zipped = multizip(
        (
            neuron_states.rows(),
            nodes.axis_iter(Axis(0)),
            intra_connections.axis_iter(Axis(0)),
        )
    );

    let now = Instant::now();
    zipped
    .into_iter()
    .enumerate()
    .par_bridge()
    .into_par_iter()
    .for_each(|(neuron_index, (neuron_state, node_states, intra_connections))| {
        let neuron_state = unpack_array(neuron_state);
        let precalculated_neuron_state = model.precalculate(NEURON_STATE, neuron_state.view());
        let node_states = unpack_array(node_states.view());
        let mut will_move_connection = Array2::from_elem(
            (g_settings.n_nodes, g_settings.n_intraconnections_per_node),
            false
        );
        for (node_a_local_index, node_a) in node_states.rows().into_iter().enumerate() {
            let precalculated_node = model.precalculate(NODE_A, node_a);
            let precalculated = [
                &precalculated_neuron_state,
                &precalculated_node
            ];
            let already_connected: Vec<usize> = intra_connections.row(node_a_local_index).iter().map(|c| c.get_index()).collect();
            for (connection_index, connection) in intra_connections.row(node_a_local_index).iter().enumerate() {
                let main_gradient = update_main_connection(connection, model, &precalculated, &node_states);
                let pending_gradient = update_pending_connection(connection, model, &precalculated, &node_states, intra_connections.view(), g_settings);
                if main_gradient > pending_gradient {
                    connection.reset_pending();  // TODO: Is this a good idea?
                    continue;
                }

                // Do not connect to nodes already connected with this node
                // TODO: Can this be done in a better way?
                let pending_index = connection.get_pending_index();
                // TODO: Doublecheck that this works, something is off
                if already_connected.contains(&pending_index) {
                    continue;
                }

                let pending_bond_force = connection.get_pending_connection_bond_force();
                let bond_force = connection.get_connection_bond_force();
                if pending_bond_force > bond_force {
                    //debug!("Bond force for Neuron {neuron_index} Node {node_a_local_index} pending bond force {pending_bond_force} > bond force {bond_force}");
                    *will_move_connection.get_mut((node_a_local_index, connection_index)).unwrap() = true;
                }
            }
        }

        // Need to do this last, since otherwise the index will move during the search
        for node_a_local_index in 0..g_settings.n_nodes {
            for (connection_index, connection) in intra_connections.row(node_a_local_index).iter().enumerate() {
                let will_move = *will_move_connection.get((node_a_local_index, connection_index)).unwrap();
                if will_move {
                    //debug!("Moving Neuron {neuron_index} Node {node_a_local_index} connection {connection_index} to Node {}", connection.get_pending_index());
                    connection.move_pending_to_main();
                }
            }
        }
    });
    trace!("It took {:?} to update intraconnections", now.elapsed());
}



fn update_main_connection(
    connection: &IntraConnection,
    model: &Model,
    precalculated: &[&Array1<f32>],
    nodes: &Array2<f32>,
) -> f32 {
    let node_b_index = connection.get_index();
    let node_b = nodes.slice(s![node_b_index, ..]);
    let (strength, pushback) = connection.get_strength_and_pushback();
    let strength_arr = value_to_array(strength);
    let pushback_arr = value_to_array(pushback);
    let inputs = [
        (NODE_B, expand(node_b.view())),
        (STRENGTH, strength_arr.view()),
        (PUSHBACK, pushback_arr.view())
    ];
    let output = model.forward_from_precalc(&inputs, precalculated);
    let connection_params = unpack_connection_model_output(output);
    connection.store_strength_and_pushback(strength + connection_params.strength, pushback + connection_params.pushback);
    connection_params.gradient.max(0.0)
}

/// Search neurons side-by-side and the connecting neuron. Same there
/// TODO: Split function? Very big input
fn update_pending_connection(
    connection: &IntraConnection,
    model: &Model,
    precalculated: &[&Array1<f32>],
    nodes: &Array2<f32>,
    intra_connections: ArrayView2<IntraConnection>,
    g_settings: &GuardianSettings,
) -> f32 {
    let pending_node_index = connection.get_pending_index();
    let mut index_highest_gradient: usize = 0;
    let mut params_highest_gradient = ConnectionParams { strength: 0.0, pushback: 0.0, gradient: f32::MIN };

    // Start with searching neuron and vicinity where pending is index
    search_area(
        &mut index_highest_gradient,
        &mut params_highest_gradient,
        pending_node_index,
        pending_node_index,
        connection,
        model,
        precalculated,
        nodes,
        g_settings
    );

    // Search connected nodes
    for connection in intra_connections.row(pending_node_index) {
        let node_b_index = connection.get_index();
        let node_b = nodes.slice(s![node_b_index, ..]);
        let inputs = [
            (NODE_B, expand(node_b.view()))
            // Strength and pushback is always 0!
        ];
        let output = model.forward_from_precalc(&inputs, precalculated);
        let connection_params = unpack_connection_model_output(output);
        if connection_params.gradient > params_highest_gradient.gradient {
            params_highest_gradient = connection_params;
            index_highest_gradient = node_b_index;
        }
    }
    connection.store_pending_strength_and_pushback(params_highest_gradient.strength, params_highest_gradient.pushback);
    connection.store_pending_index(index_highest_gradient);
    params_highest_gradient.gradient.max(0.0)
}

fn search_area(
    index_highest_gradient: &mut usize,
    params_highest_gradient: &mut ConnectionParams,
    start_node_index: usize,
    pending_node_index: usize,
    connection: &IntraConnection,
    model: &Model,
    precalculated: &[&Array1<f32>],
    nodes: &Array2<f32>,
    g_settings: &GuardianSettings,
) {
    // Search nodes in close proximity
    let node_range = -(g_settings.n_intraconnected_nodes_search as isize)..=(g_settings.n_intraconnected_nodes_search as isize);
    for node_offset in node_range {
        let node_b_index = wrap_index(start_node_index, node_offset, g_settings.n_nodes);
        let node_b = nodes.slice(s![node_b_index, ..]);
        let is_pending = pending_node_index == node_b_index;
        let (strength, pushback) = if is_pending {
            connection.get_pending_strength_and_pushback()
        } else {
            (0.0, 0.0)
        };
        let strength_arr = value_to_array(strength);
        let pushback_arr = value_to_array(pushback);
        let inputs = [
            (NODE_B, expand(node_b.view())),
            (STRENGTH, strength_arr.view()),
            (PUSHBACK, pushback_arr.view())
        ];
        let output = model.forward_from_precalc(&inputs, precalculated);
        let mut connection_params = unpack_connection_model_output(output);
        // TODO: Limit to -1.0 and 1.0?
        connection_params.strength += strength;
        connection_params.pushback += pushback;
        if connection_params.gradient > params_highest_gradient.gradient {
            *params_highest_gradient = connection_params;
            *index_highest_gradient = node_b_index;
        }
    }
}
