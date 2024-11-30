use core::f32;
use std::time::Instant;

use interface::NodeState;
use tracing::trace;
use ndarray::{Array1, Array2, Array3, Axis};
use itertools::multizip;
use ndarray::parallel::prelude::*;
use rayon::iter::ParallelBridge;
use rayon::ThreadPool;

use crate::cpu::model::Model;
use crate::cpu::interface::{InterConnection, Network};
use crate::{GuardianSettings, NetworkSettings};
use crate::cpu::*;

// Input
const NEURON_STATE_SELF: usize = 0;
const NEURON_STATE_OTHER: usize = 1;
const NODE_SELF: usize = 2;
const NODE_OTHER: usize = 3;
const FORCE_SELF: usize = 4;
const FORCE_OTHER: usize = 5;

// Output
const DELTA_FORCE_SELF: usize = 0;

pub fn update(network: &mut Network, pool: &ThreadPool) {
    update_connections(network, pool);
    attempt_connection(network, pool);
}

fn update_connections(network: &mut Network, pool: &ThreadPool) {
    let nodes = &network.state.nodes;
    let neuron_states = &network.state.neuron_states;
    let counters = &network.state.inter_connection_counters;
    let inter_connections_source = &network.state.inter_connections;
    let genome = &network.genome;

    let g_settings = &network.g_settings;
    let n_settings = &network.n_settings;
    let model = &genome.interconnections_update;

    let zipped = multizip(
        (
            neuron_states.rows(),
            nodes.axis_iter(Axis(0)),
            inter_connections_source.axis_iter(Axis(0)),
            counters.axis_iter(Axis(0))
        )
    );

    let now = Instant::now();
    let operation = zipped
    .into_iter()
    .enumerate()
    .par_bridge()
    .for_each(|(neuron_index_self, (neuron_state, node_states, inter_connections, counters))| {
        let neuron_state = unpack_array(neuron_state);
        let precalculated_neuron_state_self = model.precalculate(NEURON_STATE_SELF, neuron_state.view());
        let precalculated_neuron_state_other = model.precalculate(NEURON_STATE_OTHER, neuron_state.view());
        let node_index_offset = neuron_index_self * g_settings.n_nodes_per_neuron;
        for (node_local_index_self, node_self) in node_states.rows().into_iter().enumerate() {
            let node_self_global_index = node_local_index_self + node_index_offset;
            let connection_self = inter_connections.get(node_local_index_self).unwrap();
            let node_self = unpack_array(node_self);
            let counter_self = counters.get(node_local_index_self).unwrap();
            let precalculated_node_self = model.precalculate(NODE_SELF, node_self.view());
            let precalculated_node_other = model.precalculate(NODE_OTHER, node_self.view());
            let connection_other = get_other_inter_connection(connection_self, inter_connections_source, g_settings);

            // This is not that nice
            let precalculated = [
                &precalculated_neuron_state_self,
                &precalculated_node_self,
                &precalculated_neuron_state_other,
                &precalculated_node_other
            ];
            update_main_connection(
                node_self_global_index,
                connection_self,
                connection_other,
                model,
                &precalculated,
                nodes,
                neuron_states,
                g_settings
            );
            update_pending_connection(
                connection_self,
                connection_other,
                counter_self,
                model,
                &precalculated,
                nodes,
                neuron_states,
                inter_connections_source,
                g_settings,
                n_settings
            );
        }
    });
    pool.install(|| operation);
    trace!("It took {:?} to update interconnections", now.elapsed());
}


fn update_main_connection(
    node_self_global_index: usize,
    connection_self: &InterConnection,
    connection_other: &InterConnection,
    model: &Model,
    precalculated: &[&Array1<f32>],
    nodes: &Array3<u8>,
    neuron_states: &Array2<u8>,
    g_settings: &GuardianSettings,
) {
    let node_other_global_index = connection_self.get_index();
    let (neuron_other_index, node_other_local_index) = node_global_to_local_index(node_other_global_index, g_settings);

    // Not connected anymore!
    if !check_is_connected(node_self_global_index, connection_other) {
        connection_self.reset_main();
        return;
    } else if node_other_global_index > node_self_global_index {
        // Highest index calculates both!
        return;
    }

    let (force_self, force_other) = connection_self.get_forces();
    let (delta_force_self, delta_force_other) = get_delta_forces(
        neuron_other_index,
        node_other_local_index,
        force_self,
        force_other,
        model,
        precalculated,
        nodes,
        neuron_states
    );
    let updated_force_self = force_self + delta_force_self;
    let updated_force_other = force_other + delta_force_other;
    connection_self.store_forces(updated_force_self, updated_force_other);
    connection_other.store_forces(updated_force_other, updated_force_self);
}

/// Search neurons side-by-side and the connecting neuron. Same there
/// TODO: Split function? Very big input
fn update_pending_connection(
    connection_self: &InterConnection,
    connection_other: &InterConnection,
    counter: &CounterInterConnection,
    model: &Model,
    precalculated: &[&Array1<f32>],
    nodes: &Array3<u8>,
    neuron_states: &Array2<u8>,
    inter_connections: &Array2<InterConnection>,
    g_settings: &GuardianSettings,
    n_settings: &NetworkSettings
) {
    // Failed -> Searching. Will try one time, otherwise reset
    let failed_previous = match counter.get_state(g_settings) {
        NodeState::Failed => {
            counter.reset();
            true
        },
        _ => false
    };
    match counter.get_state(g_settings) {
        NodeState::Searching => {
            let mut highest_net_force = f32::MIN;
            let mut forces = (f32::MIN, f32::MIN);
            let mut neuron_node_index = (0, 0);
            let search = get_area_to_search(connection_self, inter_connections, g_settings, n_settings);
            for (neuron_index, node_local_index) in search {
                let (force_self, force_other) = get_delta_forces(
                    neuron_index,
                    node_local_index,
                    0.0,
                    0.0,
                    model,
                    precalculated,
                    nodes,
                    neuron_states
                );
                let net_force = force_self + force_other;
                if net_force > highest_net_force {
                    forces = (force_self, force_other);
                    highest_net_force = net_force;
                    neuron_node_index = (neuron_index, node_local_index);
                }
            }
            let pending_index = connection_self.get_pending_index();
            let highest_index = node_local_to_global_index(neuron_node_index.0, neuron_node_index.1, g_settings);
            connection_self.store_pending_index(highest_index);
            if failed_previous && highest_index == pending_index {
                // Stuck in a local maxima. Force reset
                connection_self.reset_pending();
            } else if pending_index == highest_index {  // found local maximum, nothing higher around. Attempt connection
                counter.inc();
                connection_self.store_pending_forces(forces.0, forces.1);
            } else {
                connection_self.store_pending_forces(0.0, 0.0);
            }
        }
        NodeState::Connecting => {
            let node_global_index = connection_self.get_pending_index();
            let (force_self, force_other) = connection_self.get_pending_forces();
            let (neuron_index, node_local_index) = node_global_to_local_index(node_global_index, g_settings);
            let (delta_force_self, delta_force_other) = get_delta_forces(
                neuron_index,
                node_local_index,
                force_self,
                force_other,
                model,
                precalculated,
                nodes,
                neuron_states
            );
            let updated_force_self = force_self + delta_force_self;
            let updated_force_other = force_other + delta_force_other;
            connection_self.store_pending_forces(
                updated_force_self,
                updated_force_other
            );
            let net_force = connection_self.get_net_pending_force();
            let net_force_other_to_beat = connection_other.get_net_force();
            // TODO: Think about this one! Maybe just enough to beat force_self?
            let net_force_self_to_beat = connection_self.get_net_force();
            if net_force > net_force_other_to_beat && net_force > net_force_self_to_beat {
                counter.saturate();
            } else {
                counter.inc();
            }
        },
        _ => {}  // Should never be here
    }
}

fn get_area_to_search(
    connection_self: &InterConnection,
    inter_connections: &Array2<InterConnection>,
    g_settings: &GuardianSettings,
    n_settings: &NetworkSettings
) -> Vec<(usize, usize)> {
    // Skip main
    let (main_neuron_index, main_node_index) = node_global_to_local_index(connection_self.get_index(), g_settings);

    // Start with searching neuron and vicinity where pending is index
    let (pending_neuron_index, pending_node_index) = node_global_to_local_index(connection_self.get_pending_index(), g_settings);
    let connection_other = get_inter_connection(pending_neuron_index, pending_node_index, inter_connections);
    let (neuron_index_other, node_index_other) = node_global_to_local_index(connection_other.get_index(), g_settings);

    // NOTE: Needs to be inclusive! Otherwise, if 1, it will be -1..1 -> -1 and 0
    let neuron_range: Vec<isize> = (-(g_settings.n_interconnected_neuron_search as isize)..=(g_settings.n_interconnected_neuron_search as isize)).collect();
    let node_range: Vec<isize> = (-(g_settings.n_interconnected_nodes_search as isize)..=(g_settings.n_interconnected_nodes_search as isize)).collect();

    // This could be optimized to skip the vector
    let mut search = vec![];
    let start_points = [
        (pending_neuron_index, pending_node_index),
        (neuron_index_other, node_index_other)
    ];
    for (start_neuron_index, start_node_index) in start_points {
        for neuron_offset in neuron_range.iter() {
            let neuron_index = wrap_index(start_neuron_index, *neuron_offset, n_settings.n_neurons);
            for node_offset in node_range.iter() {
                let node_local_index = wrap_index(start_node_index, *node_offset, g_settings.n_nodes_per_neuron);
                if neuron_index == main_neuron_index && main_node_index == node_local_index {
                    continue;
                }
                search.push((neuron_index, node_local_index));
            }
        }
    }
    search
}


fn get_delta_forces(
    neuron_index: usize,
    node_local_index: usize,
    force_self: f32,
    force_other: f32,
    model: &Model,
    precalculated: &[&Array1<f32>],
    nodes: &Array3<u8>,
    neuron_states: &Array2<u8>,
) -> (f32, f32) {
    // Could optimize this, so it reuses the neuron state if it already exist
    let neuron_state_other = get_neuron_state(neuron_index, neuron_states);
    let node_other = get_node(neuron_index, node_local_index, nodes);

    let force_self = value_to_array(force_self);
    let force_other = value_to_array(force_other);

    // self -> other
    let inputs = [
        (NEURON_STATE_OTHER, expand(neuron_state_other.view())),
        (NODE_OTHER, expand(node_other.view())),
        (FORCE_SELF, force_self.view()),
        (FORCE_OTHER, force_other.view())
    ];
    let output = model.forward_from_precalc(&inputs, &precalculated[0..2]);  // neuron_self, node_self
    let delta_force_self = *output[DELTA_FORCE_SELF].first().unwrap();

    // other -> self
    let inputs = [
        (NEURON_STATE_SELF, expand(neuron_state_other.view())),
        (NODE_SELF, expand(node_other.view())),
        (FORCE_SELF, force_other.view()),
        (FORCE_OTHER, force_self.view())
    ];
    let output = model.forward_from_precalc(&inputs, &precalculated[2..]);  // neuron_other, node_other
    let delta_force_other = *output[DELTA_FORCE_SELF].first().unwrap();

    (delta_force_self, delta_force_other)
}


pub fn attempt_connection(network: &mut Network, pool: &ThreadPool) {
    let inter_connection_counters = &network.state.inter_connection_counters;
    let inter_connections_source = &network.state.inter_connections;
    let g_settings = &network.g_settings;

    let zipped = multizip(
        (
            inter_connections_source.axis_iter(Axis(0)),
            inter_connection_counters.axis_iter(Axis(0))
        )
    );
    let zipped_iter = zipped
        .into_iter()
        .enumerate()
        .par_bridge();

    // Step 1: Check if other is also connecting, otherwise, try to connect
    let now = Instant::now();
    let operation = zipped_iter.clone()
    .for_each(|(_neuron_index, (inter_connections, counters))| {
        let iter = inter_connections.iter().zip(counters);
        for (connection_self, counter_self) in iter {
            let node_other_global_index = connection_self.get_pending_index();
            let (neuron_other_index, node_other_local_index) = node_global_to_local_index(node_other_global_index, g_settings);
            let counter_other = get_inter_connection_counter(neuron_other_index, node_other_local_index, inter_connection_counters);

            let node_state_self = counter_self.get_state(g_settings);
            let node_state_other = counter_other.get_state(g_settings);
            match (node_state_self, node_state_other) {
                (
                    NodeState::AttemptingTakeover,
                    NodeState::AttemptingTakeover | NodeState::Failed
                ) => {
                    // The other connections is trying to go somewhere else! Here, it is difficult to sync
                    // the connections. Just reset to search mode again
                    counter_self.failed();
                    connection_self.reset_pending();
                },
                (NodeState::AttemptingTakeover, _) => {
                    // Will attempt to connect
                    // However, since there are 2 values, the highest force that the other connection want wins
                    let connection_other = get_inter_connection(neuron_other_index, node_other_local_index, inter_connections_source);
                    let (_force_self, force_other) = connection_self.get_pending_forces();
                    connection_other.add_maximum_force_self(force_other);  // yes, it should be other here!
                }
                _ => {}
            }
        }
    });
    pool.install(|| operation);
    trace!("It took {:?} to check the counters", now.elapsed());

    // Step 2: Could be multiple "winners". If multiple that have the exact same value, the highest index wins
    let now = Instant::now();
    let operation = zipped_iter.clone()
    .for_each(|(neuron_index, (inter_connections, counters))| {
        let iter = inter_connections.iter().zip(counters);
        for (node_local_index, (connection_self, counter_self)) in iter.enumerate() {
            let node_state_self = counter_self.get_state(g_settings);
            match node_state_self {
                NodeState::AttemptingTakeover => {
                    let node_other_global_index = connection_self.get_pending_index();
                    let (neuron_other_index, node_other_local_index) = node_global_to_local_index(node_other_global_index, g_settings);
                    let connection_other = get_inter_connection(neuron_other_index, node_other_local_index, inter_connections_source);
                    let (_, force_other) = connection_self.get_pending_forces();
                    let (force_self, _) = connection_other.get_forces();
                    if force_other == force_self {
                        // This node has won, but there might be multiple!
                        // Thus, the highest index wins
                        let global_index = node_local_to_global_index(neuron_index, node_local_index, g_settings);
                        connection_other.add_maximum_index(global_index);
                    } else {
                        // It failed, did not win the competition. Go back to searching
                        connection_self.reset_pending();
                        counter_self.reset();
                    }
                },
                NodeState::Failed => {
                    connection_self.reset_pending();
                    counter_self.reset();
                },
                _ => {}
            }
        }
    });
    pool.install(|| operation);
    trace!("It took {:?} to check competition", now.elapsed());

    // Step 3: Check if it won, in that case, establish the connection
    let now = Instant::now();
    let operation = zipped_iter.clone()
    .for_each(|(neuron_index, (inter_connections, counters))| {
        let iter = inter_connections.iter().zip(counters).enumerate();
        for (node_local_index_self, (connection_self, counter_self)) in iter {
            let node_state_self = counter_self.get_state(g_settings);
            match node_state_self {
                NodeState::AttemptingTakeover => {
                    // If it is still in this state, it has won and will connect
                    let node_other_global_index = connection_self.get_pending_index();
                    let (neuron_other_index, node_other_local_index) = node_global_to_local_index(node_other_global_index, g_settings);
                    let connection_other = get_inter_connection(neuron_other_index, node_other_local_index, inter_connections_source);

                    let node_global_index_self = node_local_to_global_index(neuron_index, node_local_index_self, g_settings);
                    if connection_other.get_index() != node_global_index_self {
                        // Failed, something else with a higher index won
                        connection_self.reset_pending();
                        counter_self.reset();
                    }

                    // Index has already been set, just forces
                    let (force_self, force_other) = connection_self.get_pending_forces();
                    connection_self.move_pending_to_main();  // This order is correct, otherwise it might take the wrong one!
                    connection_other.store_forces(force_other, force_self);  // Yes, it should be this way
                },
                _ => {}
            }
        }
    });
    pool.install(|| operation);
    trace!("It took {:?} to check competition", now.elapsed());
}

