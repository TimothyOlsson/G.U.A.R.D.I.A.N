use std::time::Instant;

use tracing::debug;
use ndarray::{Array1, Array2, Array3, Axis};
use itertools::multizip;
use ndarray::parallel::prelude::*;
use rayon::iter::ParallelBridge;

use crate::cpu::model::Model;
use crate::cpu::interface::{Flags, Flag, InterConnection, Network};
use super::*;
use crate::{GuardianSettings, NetworkSettings, node_global_to_local_index, node_local_to_global_index};

const NEURON_STATE_A: usize = 0;
const NEURON_STATE_B: usize = 1;
const NODE_A: usize = 2;
const NODE_B: usize = 3;
const STRENGTH: usize = 4;
const PUSHBACK: usize = 5;

pub fn update(network: &mut Network, pool: &ThreadPool) {
    update_connections(network, pool);
    flag_if_should_attempt_connection(network, pool);
    attempt_connection(network, pool);
    compete_over_node(network, pool);
    check_attempt(network, pool);
}

fn update_connections(network: &mut Network, pool: &ThreadPool) {
    let nodes = &network.state.nodes;
    let neuron_states = &network.state.neuron_states;
    let inter_connections_source = &network.state.inter_connections;
    let inter_connections_flags_source = &network.state.inter_connections_flags;
    let genome = &network.genome;

    let g_settings = &network.g_settings;
    let n_settings = &network.n_settings;
    let model = &genome.interconnections_update;

    let zipped = multizip(
        (
            neuron_states.rows(),
            nodes.axis_iter(Axis(0)),
            inter_connections_source.axis_iter(Axis(0)),
            inter_connections_flags_source.axis_iter(Axis(0)),
        )
    );

    let now = Instant::now();
    let operation = zipped
    .into_iter()
    .enumerate()
    .par_bridge()
    .for_each(|(neuron_a_index, (neuron_state, node_states, inter_connections, inter_connections_flags))| {
        let neuron_state = unpack_array(neuron_state);
        let precalculated_neuron_state_a = model.precalculate(NEURON_STATE_A, neuron_state.view());
        let node_index_offset = neuron_a_index * g_settings.n_nodes_per_neuron;
        for (node_a_local_index, node_a) in node_states.rows().into_iter().enumerate() {
            let node_a_global_index = node_a_local_index + node_index_offset;
            let connection_a = inter_connections.get(node_a_local_index).unwrap();
            let flags = inter_connections_flags.get(node_a_local_index).unwrap();
            let node_a = unpack_array(node_a);
            let precalculated_node_a = model.precalculate(NODE_A, node_a.view());
            let precalculated = [&precalculated_neuron_state_a, &precalculated_node_a];
            let main_gradient = update_main_connection(node_a_global_index, connection_a, flags, model, &precalculated, nodes, neuron_states, inter_connections_source, g_settings);
            let pending_gradient = update_pending_connection(connection_a, model, &precalculated, nodes, neuron_states, inter_connections_source, g_settings, n_settings);
            // TODO: Is it a good idea to also reset if graident is higher?
            if main_gradient > pending_gradient {
                //debug!("Resetting! Main gradient {main_gradient}, pending {pending_gradient}!");
                connection_a.reset_pending();
            }
        }
    });
    pool.install(|| operation);
    trace!("It took {:?} to update interconnections", now.elapsed());
}


fn flag_if_should_attempt_connection(network: &mut Network, pool: &ThreadPool) {
    let inter_connections_source = &network.state.inter_connections;
    let inter_connections_flags_source = &network.state.inter_connections_flags;
    let zipped = multizip(
        (
            inter_connections_source.axis_iter(Axis(0)),
            inter_connections_flags_source.axis_iter(Axis(0))
        )
    );

    let g_settings = &network.g_settings;
    let now = Instant::now();
    let operation = zipped
    .into_iter()
    .enumerate()
    .par_bridge()
    .for_each(|(neuron_a_index, (inter_connections, inter_connections_flags))| {
        let node_index_offset = neuron_a_index * g_settings.n_nodes_per_neuron;
        for (node_a_local_index, connection_a) in inter_connections.into_iter().enumerate() {
            let node_a_global_index = node_a_local_index + node_index_offset;

            let pending_bond_force = connection_a.get_pending_bond_force();
            let mut bond_force_to_beat = 0.0;  // if negative, do not connect!

            // Check self (if connected)
            let bond_force = get_bond_force_between_two_connections(node_a_global_index, connection_a, inter_connections_source, g_settings);
            bond_force_to_beat = f32::max(bond_force_to_beat, bond_force);

            // Check pending connection
            let node_b_global_index = connection_a.get_pending_index();
            let (neuron_b_index, node_b_local_index) = node_global_to_local_index(node_b_global_index, g_settings);
            let connection_b = get_inter_connection(neuron_b_index, node_b_local_index, inter_connections_source);
            let bond_force = get_bond_force_between_two_connections(node_b_global_index, connection_b, inter_connections_source, g_settings);
            bond_force_to_beat = f32::max(bond_force_to_beat, bond_force);

            /*
            debug!(
                "Node {node_a_global_index} with index {} pending on {} with flag {}. force to beat = {bond_force_to_beat}, pending force = {pending_bond_force}",
                connection_a.get_index(),
                connection_a.get_pending_index(),
                source
            );
            */
            if pending_bond_force > bond_force_to_beat {
                //debug!("Node {node_a_global_index} attempting connection on {node_b_global_index}");
                let flags = inter_connections_flags.get(node_a_local_index).unwrap();
                flags.add_flag(Flag::CONNECTING);
            }
        }
    });
    pool.install(|| operation);
    trace!("It took {:?} to flag attempting interconnections", now.elapsed());

}

pub fn attempt_connection(network: &mut Network, pool: &ThreadPool) {
    let inter_connections_source = &network.state.inter_connections;
    let inter_connections_flags_source = &network.state.inter_connections_flags;
    let g_settings = &network.g_settings;

    let zipped = multizip(
        (
            inter_connections_source.axis_iter(Axis(0)),
            inter_connections_flags_source.axis_iter(Axis(0))
        )
    );

    let now = Instant::now();
    let operation = zipped
    .into_iter()
    .enumerate()
    .par_bridge()
    .for_each(|(neuron_a_index, (inter_connections, inter_connections_flags))| {
        let node_index_offset = neuron_a_index * g_settings.n_nodes_per_neuron;
        for (node_a_local_index, connection_a) in inter_connections.into_iter().enumerate() {
            let _node_a_global_index = node_a_local_index + node_index_offset;

            // Check if should connect
            let flags_a = inter_connections_flags.get(node_a_local_index).unwrap();
            if !flags_a.check_flag(Flag::CONNECTING) {
                continue;
            }

            let node_b_global_index = connection_a.get_pending_index();
            let (neuron_b_index, node_b_local_index) = node_global_to_local_index(node_b_global_index, g_settings);
            let connection_b = get_inter_connection(neuron_b_index, node_b_local_index, inter_connections_source);

            // Check if the node is already trying to connect
            let flags_b = get_inter_connection_flags(neuron_b_index, node_b_local_index, inter_connections_flags_source);
            if flags_b.check_flag(Flag::CONNECTING) {  // B is already connecting!
                //debug!("Node {node_a_global_index} CANNOT connect to Node {node_b_global_index}, which is currently attempting a new connection");
                connection_a.reset_pending();
                continue;
            }

            //debug!("Node {node_a_global_index} will try to connect to Node {node_b_global_index}, which is not attempting new connection.");

            // If all ok, compete over the same connection
            let pending_bond_force = connection_a.get_pending_bond_force();
            connection_b.add_maximum_bond_force(pending_bond_force);
            connection_b.store_index(0);
        }
    });
    pool.install(|| operation);
    trace!("It took {:?} to attempt interconnections", now.elapsed());
}


pub fn compete_over_node(network: &mut Network, pool: &ThreadPool) {
    let inter_connections_source = &network.state.inter_connections;
    let inter_connections_flags_source = &network.state.inter_connections_flags;
    let g_settings = &network.g_settings;

    let zipped = multizip(
        (
            inter_connections_source.axis_iter(Axis(0)),
            inter_connections_flags_source.axis_iter(Axis(0))
        )
    );

    let now = Instant::now();
    let operation = zipped
    .into_iter()
    .enumerate()
    .par_bridge()
    .for_each(|(neuron_a_index, (inter_connections, inter_connections_flags))| {
        let node_index_offset = neuron_a_index * g_settings.n_nodes_per_neuron;
        for (node_a_local_index, connection_a) in inter_connections.into_iter().enumerate() {
            let node_a_global_index = node_a_local_index + node_index_offset;

            // Check if should connect
            let flags = inter_connections_flags.get(node_a_local_index).unwrap();
            if flags.check_flag(Flag::FAILED) {
                flags.remove_flag(Flag::CONNECTING);
                flags.remove_flag(Flag::FAILED);
                continue;
            } else if !flags.check_flag(Flag::CONNECTING) {
                continue;
            }

            let node_b_global_index = connection_a.get_pending_index();
            let (neuron_b_index, node_b_local_index) = node_global_to_local_index(node_b_global_index, g_settings);
            let connection_b = get_inter_connection(neuron_b_index, node_b_local_index, inter_connections_source);

            // Only the highest bond-force will win
            let pending_bond_force = connection_a.get_pending_bond_force();
            let bond_force = connection_b.get_bond_force();
            if pending_bond_force >= bond_force {
                //debug!("Node {node_a_global_index} will compete over Node {node_b_global_index}");
                connection_b.add_maximum_index(node_a_global_index);
                //debug!("Connection b max index {}", connection_b.get_index());
            } else {
                //debug!("Node {node_a_global_index} will NOT compete over Node {node_b_global_index}");
                flags.remove_flag(Flag::CONNECTING);
                connection_a.reset_pending();  // TODO: Should this be done?
            }
        }
    });
    pool.install(|| operation);
    trace!("It took {:?} to compete over interconnections", now.elapsed());
}


pub fn check_attempt(network: &mut Network, pool: &ThreadPool) {
    let inter_connections_source = &network.state.inter_connections;
    let inter_connections_flags_source = &network.state.inter_connections_flags;
    let g_settings = &network.g_settings;

    let zipped = multizip(
        (
            inter_connections_source.axis_iter(Axis(0)),
            inter_connections_flags_source.axis_iter(Axis(0))
        )
    );

    let now = Instant::now();
    let operation = zipped
    .into_iter()
    .enumerate()
    .par_bridge()
    .for_each(|(neuron_a_index, (inter_connections, inter_connections_flags))| {
        let node_index_offset = neuron_a_index * g_settings.n_nodes_per_neuron;
        for (node_a_local_index, connection_a) in inter_connections.into_iter().enumerate() {
            let node_a_global_index = node_a_local_index + node_index_offset;

            // Check if should connect
            let flags = inter_connections_flags.get(node_a_local_index).unwrap();
            if !flags.check_flag(Flag::CONNECTING) {
                continue;
            }

            let node_b_global_index = connection_a.get_pending_index();
            let (neuron_b_index, node_b_local_index) = node_global_to_local_index(node_b_global_index, g_settings);
            let connection_b = get_inter_connection(neuron_b_index, node_b_local_index, inter_connections_source);

            flags.remove_flag(Flag::CONNECTING);
            if connection_b.get_index() == node_a_global_index {
                // Success!
                debug!("Node {node_a_global_index} SUCESSFULLY connected to Node {node_b_global_index}");
                connection_b.store_strength_and_pushback(0.0, 0.0);
                connection_a.move_pending_to_main();
            } else {
                // Failed, some other connection won
                let node_c_global_index = connection_b.get_index();
                let (neuron_c_index, node_c_local_index) = node_global_to_local_index(node_c_global_index, g_settings);
                let connection_c = get_inter_connection(neuron_c_index, node_c_local_index, inter_connections_source);
                debug!("Node {node_a_global_index} FAILED connected to Node {node_b_global_index}. con_b {} con_c {}", connection_b.get_index(), connection_c.get_index());
                connection_a.reset_pending();  // TODO: Should this be done?
            }
        }
    });
    pool.install(|| operation);
    trace!("It took {:?} to check compete interconnections", now.elapsed());
}



fn get_bond_force_between_two_connections(
    node_a_global_index: usize,
    connection_a: &InterConnection,
    inter_connections: &Array2<InterConnection>,
    g_settings: &GuardianSettings,
) -> f32 {
    let mut combined_bond_force = f32::MIN;
    let node_b_global_index = connection_a.get_index();
    let (neuron_b_index, node_b_local_index) = node_global_to_local_index(node_b_global_index, g_settings);
    let connection_b = get_inter_connection(neuron_b_index, node_b_local_index, inter_connections);
    if check_is_connected(node_a_global_index, connection_b) {
        let bond_force = connection_a.get_bond_force();
        combined_bond_force = f32::max(combined_bond_force, bond_force);
        let bond_force = connection_b.get_bond_force();
        combined_bond_force = f32::max(combined_bond_force, bond_force)
    }
    combined_bond_force
}

fn update_main_connection(
    node_a_global_index: usize,
    connection_a: &InterConnection,
    _flags: &Flags,
    model: &Model,
    precalculated: &[&Array1<f32>],
    nodes: &Array3<u8>,
    neuron_states: &Array2<u8>,
    inter_connections: &Array2<InterConnection>,
    g_settings: &GuardianSettings,
) -> f32 {
    let node_b_global_index = connection_a.get_index();
    let (neuron_b_index, node_b_local_index) = node_global_to_local_index(node_b_global_index, g_settings);
    let connection_b = get_inter_connection(neuron_b_index, node_b_local_index, inter_connections);

    // Is this a good idea?
    if !check_is_connected(node_a_global_index, connection_b) {
        connection_a.store_strength_and_pushback(0.0, 0.0);  // Detected disconnection, not connected here!
        return 0.0;
    }

    let neuron_state_b = get_neuron_state(neuron_b_index, neuron_states);
    let node_b = get_node(neuron_b_index, node_b_local_index, nodes);
    let (strength, pushback) = connection_a.get_strength_and_pushback();
    let strength_arr = value_to_array(strength);
    let pushback_arr = value_to_array(pushback);
    let inputs = [
        (NEURON_STATE_B, expand(neuron_state_b.view())),
        (NODE_B, expand(node_b.view())),
        (STRENGTH, strength_arr.view()),
        (PUSHBACK, pushback_arr.view())
    ];
    let output = model.forward_from_precalc(&inputs, precalculated);
    let connection_params = unpack_connection_model_output(output);
    connection_a.store_strength_and_pushback(strength + connection_params.strength, pushback + connection_params.pushback);
    connection_params.gradient.max(0.0)
}

/// Search neurons side-by-side and the connecting neuron. Same there
/// TODO: Split function? Very big input
fn update_pending_connection(
    connection_a: &InterConnection,
    model: &Model,
    precalculated: &[&Array1<f32>; 2],
    nodes: &Array3<u8>,
    neuron_states: &Array2<u8>,
    inter_connections: &Array2<InterConnection>,
    g_settings: &GuardianSettings,
    n_settings: &NetworkSettings
) -> f32 {
    let (pending_neuron_index, pending_node_index) = node_global_to_local_index(connection_a.get_pending_index(), g_settings);
    let connection_b = get_inter_connection(pending_neuron_index, pending_node_index, inter_connections);
    let (other_neuron_index, other_node_index) = node_global_to_local_index(connection_b.get_index(), g_settings);

    let mut index_highest_gradient: usize = 0;
    let mut params_highest_gradient = ConnectionParams { strength: 0.0, pushback: 0.0, gradient: f32::MIN };

    // Start with searching neuron and vicinity where pending is index
    let areas = [
        (pending_neuron_index, pending_node_index),
        (other_neuron_index, other_node_index)
    ];
    for (start_neuron, start_node) in areas {
        search_area(
            &mut index_highest_gradient,
            &mut params_highest_gradient,
            start_neuron,
            start_node,
            pending_neuron_index,
            pending_node_index,
            connection_a,
            model,
            precalculated,
            nodes,
            neuron_states,
            g_settings,
            n_settings
        );
    }
    connection_a.store_pending_strength_and_pushback(params_highest_gradient.strength, params_highest_gradient.pushback);
    connection_a.store_pending_index(index_highest_gradient);
    params_highest_gradient.gradient.max(0.0)
}

fn search_area(
    // TODO: Way too many inputs. Make it better or cleaner?
    index_highest_gradient: &mut usize,
    params_highest_gradient: &mut ConnectionParams,
    start_neuron_index: usize,
    start_node_index: usize,
    pending_neuron_index: usize,
    pending_node_index: usize,
    connection_a: &InterConnection,
    model: &Model,
    precalculated_source: &[&Array1<f32>; 2],
    nodes: &Array3<u8>,
    neuron_states: &Array2<u8>,
    g_settings: &GuardianSettings,
    n_settings: &NetworkSettings
) {
    // Search neurons and nodes in close proximity
    // NOTE: Needs to be inclusive! Otherwise, if 1, it will be -1..1 -> -1 and 0
    let neuron_range = -(g_settings.n_interconnected_neuron_search as isize)..=(g_settings.n_interconnected_neuron_search as isize);
    for neuron_offset in neuron_range {
        let neuron_b_index = wrap_index(start_neuron_index, neuron_offset, n_settings.n_neurons);
        let neuron_state_b = get_neuron_state(neuron_b_index, neuron_states);
        let precalculated = [
            precalculated_source[0],
            precalculated_source[1],
            &model.precalculate(NEURON_STATE_B, neuron_state_b.view())
        ];
        let node_range = -(g_settings.n_interconnected_nodes_search as isize)..=(g_settings.n_interconnected_nodes_search as isize);
        for node_offset in node_range {
            let node_b_local_index = wrap_index(start_node_index, node_offset, g_settings.n_nodes_per_neuron);
            let node_b = get_node(neuron_b_index, node_b_local_index, nodes);
            let is_pending = neuron_b_index == pending_neuron_index && pending_node_index == node_b_local_index;
            let (strength, pushback) = if is_pending {
                connection_a.get_pending_strength_and_pushback()
            } else {
                (0.0, 0.0)  // TODO: This could be skipped! Uneccessary calculations
            };
            let strength_arr = value_to_array(strength);
            let pushback_arr = value_to_array(pushback);
            let inputs = [
                (NODE_B, expand(node_b.view())),
                (STRENGTH, strength_arr.view()),
                (PUSHBACK, pushback_arr.view())
            ];
            let output = model.forward_from_precalc(&inputs, &precalculated);
            let mut connection_params = unpack_connection_model_output(output);
            // TODO: Limit to -1.0 and 1.0?
            connection_params.strength += strength;
            connection_params.pushback += pushback;

            if connection_params.gradient > params_highest_gradient.gradient {
                *params_highest_gradient = connection_params;
                *index_highest_gradient = node_local_to_global_index(neuron_b_index, node_b_local_index, g_settings);
            }
        }
    }
}
