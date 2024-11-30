use std::fmt::Debug;

use regex::Regex;
use ndarray::{Array, Dimension};

use crate::{GuardianSettings, NetworkSettings};
use crate::cpu::interface::State;
use crate::cpu::node_global_to_local_index;

pub fn add_param<T: Debug>(js_string: &mut String, variable: &str, data: T) {
    js_string.push_str(&format!("const {variable} = {:?};\n", data).replace('"', "").replace('\\', ""));
}

/// https://github.com/rust-ndarray/ndarray/pull/713
/// https://github.com/rust-ndarray/ndarray/issues/705
pub fn arr_to_string<T, D>(arr: &Array<T, D>) -> String
where
    T: Debug,
    D: Dimension
{
    let arr = format!("{arr:#?}").replace("\n", "");
    let re = Regex::new("(, shape=.*)").unwrap();
    let arr = re.replace_all(&arr, "").to_string();
    arr
}

pub fn visualize_network(state_history: Vec<State>, g_settings: &GuardianSettings, n_settings: &NetworkSettings) {
    let mut data = String::new();
    // Params
    add_param(&mut data, "n_neurons", n_settings.n_neurons);
    add_param(&mut data, "n_nodes_per_neuron", g_settings.n_nodes_per_neuron);
    add_param(&mut data, "node_size", g_settings.node_size);
    add_param(&mut data, "neuron_state_size", g_settings.neuron_state_size);

    // Timeline
    let timeline: Vec<usize> = (0..state_history.len()).into_iter().collect();
    add_param(&mut data, "timesteps", timeline);

    // Add links
    let mut interconnections = vec![];
    let mut pending_interconnections = vec![];
    let mut intraconnections = vec![];
    let mut pending_intraconnections = vec![];
    let mut interconnection_counters = vec![];
    let mut intraconnection_counters = vec![];
    for state in state_history.iter() {

        // Add interconnections
        let connections = state.inter_connections.indexed_iter().map(|((neuron, node), connection)| {
            let node_b_global_index = connection.get_index();
            let (target_neuron, target_node) = node_global_to_local_index(node_b_global_index, g_settings);
            let (force_self, force_other) = connection.get_raw_force_values();
            format!("{{ source: [{neuron}, {node}], target: [{target_neuron}, {target_node}], force_self: {force_self}, force_other: {force_other} }}")
        })
        .collect();
        let connections = Array::from_shape_vec(state.inter_connections.shape(), connections).unwrap();
        let connections = arr_to_string(&connections);
        interconnections.push(connections);

        // Add pending interconnections
        let connections = state.inter_connections.indexed_iter().map(|((neuron, node), connection)| {
            let node_b_global_index = connection.get_pending_index();
            let (target_neuron, target_node) = node_global_to_local_index(node_b_global_index, g_settings);
            let (force_self, force_other) = connection.get_raw_pending_force_values();
            format!("{{ source: [{neuron}, {node}], target: [{target_neuron}, {target_node}], force_self: {force_self}, force_other: {force_other} }}")
        })
        .collect();
        let connections = Array::from_shape_vec(state.inter_connections.shape(), connections).unwrap();
        let connections = arr_to_string(&connections);
        pending_interconnections.push(connections);

        // Add intraconnections
        let connections = state.intra_connections.indexed_iter().map(|((neuron, node, _), connection)| {
            let target_node = connection.get_index();
            let (force_self, force_other) = connection.get_raw_force_values();
            format!("{{ source: [{neuron}, {node}], target: [{neuron}, {target_node}], force_self: {force_self}, force_other: {force_other} }}")
        })
        .collect();
        let connections = Array::from_shape_vec(state.intra_connections.shape(), connections).unwrap();
        let connections = arr_to_string(&connections);
        intraconnections.push(connections);

        // Add pending interconnections
        let connections = state.intra_connections.indexed_iter().map(|((neuron, node, _), connection)| {
            let target_node = connection.get_pending_index();
            let (force_self, force_other) = connection.get_raw_pending_force_values();
            format!("{{ source: [{neuron}, {node}], target: [{neuron}, {target_node}], force_self: {force_self}, force_other: {force_other} }}")
        })
        .collect();
        let connections = Array::from_shape_vec(state.intra_connections.shape(), connections).unwrap();
        let connections = arr_to_string(&connections);
        pending_intraconnections.push(connections);

        // Add interconnections counters
        let counters = state.inter_connection_counters.indexed_iter().map(|((_, _), counter)| {
            let value = counter.get_value();
            let state = counter.get_state(g_settings);
            format!("{{ value: {value}, state: {state:?} }}")
        })
        .collect();
        let counters = Array::from_shape_vec(state.inter_connection_counters.shape(), counters).unwrap();
        let counters = arr_to_string(&counters);
        interconnection_counters.push(counters);

        // Add intraconnections counters
        let counters = state.intra_connection_counters.indexed_iter().map(|((_, _, _), counter)| {
            let value = counter.get_value();
            let state = counter.get_state(g_settings);
            format!("{{ value: {value}, state: {state:?} }}")
        })
        .collect();
        let counters = Array::from_shape_vec(state.intra_connection_counters.shape(), counters).unwrap();
        let counters = arr_to_string(&counters);
        intraconnection_counters.push(counters);
    }

    add_param(&mut data, "interconnections", interconnections);
    add_param(&mut data, "pending_interconnections", pending_interconnections);
    add_param(&mut data, "intraconnections", intraconnections);
    add_param(&mut data, "pending_intraconnections", pending_intraconnections);
    add_param(&mut data, "interconnection_counters", interconnection_counters);
    add_param(&mut data, "intraconnection_counters", intraconnection_counters);

    // Add states
    let mut node_states = vec![];
    let mut neuron_states = vec![];
    for state in state_history.iter() {
        let js_arr = arr_to_string(&state.nodes);
        node_states.push(js_arr);

        let js_arr = arr_to_string(&state.neuron_states);
        neuron_states.push(js_arr);
    }

    add_param(&mut data, "node_states", node_states);
    add_param(&mut data, "neuron_states", neuron_states);

    // Finalize
    std::fs::write("data.js", data).expect("Unable to write file");
}

