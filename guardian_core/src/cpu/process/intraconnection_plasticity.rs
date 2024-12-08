use std::time::Instant;
use std::ops::Range;

use itertools::Itertools;
use ndarray::{Array1, Array2, Axis};
use itertools::multizip;
use ndarray::parallel::prelude::*;
use rayon::iter::ParallelBridge;
use rayon::ThreadPool;
use tracing::trace;

use interface::{CounterIntraConnection, IntraConnection, Network, NodeState};
use crate::cpu::model::Model;
use crate::GuardianSettings;
use crate::cpu::*;

// Input
const NEURON_STATE: usize = 0;
const NODE_STATE_SELF: usize = 1;
const NODE_STATE_OTHER: usize = 2;
const FORCE_SELF: usize = 3;
const FORCE_OTHER: usize = 4;

// Output
const DELTA_FORCE_SELF: usize = 0;
const DELTA_FORCE_OTHER: usize = 1;

pub fn update(network: &mut Network, pool: &ThreadPool) {
    let nodes = &network.state.nodes;
    let neuron_states = &network.state.neuron_states;
    let intra_connections = &mut network.state.intra_connections;
    let inter_connection_counters = &mut network.state.intra_connection_counters;
    let genome = &network.genome;
    let model = &genome.intraconnections_update;
    let g_settings = &network.g_settings;

    let zipped = multizip(
        (
            neuron_states.rows(),
            nodes.axis_iter(Axis(0)),
            intra_connections.axis_iter_mut(Axis(0)),
            inter_connection_counters.axis_iter_mut(Axis(0)),
        )
    );

    let now = Instant::now();
    let operation = zipped
    .into_iter()
    .enumerate()
    .par_bridge()
    .for_each(|(_neuron_index, (neuron_state, node_states, mut intra_connections, mut counters))| {
        let neuron_state = unpack_array(neuron_state);
        let node_states = unpack_array(node_states.view());
        let precalculated_neuron_state = model.precalculate(NEURON_STATE, neuron_state.view());
        for (node_local_index_self, node_state_self) in node_states.rows().into_iter().enumerate() {
            let precalculated_node = model.precalculate(NODE_STATE_SELF, node_state_self);
            let precalculated = [
                &precalculated_neuron_state,
                &precalculated_node
            ];
            let mut node_intra_connections = intra_connections.row_mut(node_local_index_self);
            for (connection_index, connection) in node_intra_connections.iter_mut().enumerate() {
                let counter = counters.get_mut((node_local_index_self, connection_index)).unwrap();
                update_main_connection(
                    connection,
                    model,
                    &precalculated,
                    &node_states
                );
                update_pending_connection(
                    connection,
                    model,
                    &precalculated,
                    &node_states,
                    counter,
                    g_settings,
                );
            }

            // Compare against each other, the highest main will win and kick out the rest
            let mut occupied = vec![false; g_settings.n_intraconnections_per_node];
            for comb in node_intra_connections.iter().enumerate().combinations(2) {
                let (a_index, a) = comb[0];
                let (b_index, b) = comb[1];

                let counter_a = counters.get((node_local_index_self, a_index)).unwrap();
                let counter_b = counters.get((node_local_index_self, b_index)).unwrap();

                if counter_a.get_state(g_settings) == NodeState::Searching || counter_b.get_state(g_settings) == NodeState::Searching {
                    // One of them is just passing through
                    continue;
                } else if a.get_pending_index() != b.get_pending_index() {
                    // The pending indices the same, it's ok
                    continue;
                }
                let force_a = a.get_net_pending_force();
                let force_b = b.get_net_pending_force();
                if force_a == force_b {
                    // Highest index wins
                    if b_index > a_index {
                        occupied[a_index] = true;
                    } else {
                        occupied[b_index] = true;
                    }
                } else if force_b > force_a {
                    // kick out a
                    occupied[a_index] = true;
                } else if force_a > force_b {
                    // kick out b
                    occupied[b_index] = true;
                }
            }

            let iter = node_intra_connections.iter_mut().zip(occupied).enumerate();
            for (connection_index, (connection, should_move)) in iter {
                if should_move {
                    // If failed, move the pending to the opposite node
                    let counter = counters.get_mut((node_local_index_self, connection_index)).unwrap();
                    let new_index = opposite_index(connection.get_pending_index(), g_settings.n_nodes_per_neuron);
                    connection.reset_pending();
                    counter.reset();
                    connection.store_pending_index(new_index);
                }
            }
    }
    });
    pool.install(|| operation);
    trace!("It took {:?} to update intraconnections", now.elapsed());
}


fn get_pending_delta_forces(
    node_index_other: usize,
    force_self: f32,
    force_other: f32,
    model: &Model,
    precalculated: &[&Array1<f32>],
    nodes: &Array2<f32>,
) -> (f32, f32) {
    let node_state_other = nodes.slice(s![node_index_other, ..]);
    let force_self_arr = value_to_array(force_self);
    let force_other_arr = value_to_array(force_other);
    let inputs = [
        (NODE_STATE_OTHER, expand(node_state_other.view())),
        (FORCE_SELF, force_self_arr.view()),
        (FORCE_OTHER, force_other_arr.view())
    ];
    let output = model.forward_from_precalc(&inputs, precalculated);
    let delta_force_self = *output[DELTA_FORCE_SELF].first().unwrap();
    let delta_force_other = *output[DELTA_FORCE_OTHER].first().unwrap();
    (delta_force_self, delta_force_other)
}


fn update_main_connection(
    connection: &mut IntraConnection,
    model: &Model,
    precalculated: &[&Array1<f32>],
    nodes: &Array2<f32>,
) {
    let node_index_other = connection.get_index();
    let node_state_other = nodes.slice(s![node_index_other, ..]);
    let (force_self, force_other) = connection.get_forces();
    let force_self_arr = value_to_array(force_self);
    let force_other_arr = value_to_array(force_other);
    let inputs = [
        (NODE_STATE_OTHER, expand(node_state_other.view())),
        (FORCE_SELF, force_self_arr.view()),
        (FORCE_OTHER, force_other_arr.view())
    ];
    let output = model.forward_from_precalc(&inputs, precalculated);
    let delta_force_self = *output[DELTA_FORCE_SELF].first().unwrap();
    let delta_force_other = *output[DELTA_FORCE_OTHER].first().unwrap();
    connection.store_forces(force_self + delta_force_self, force_other + delta_force_other);
}


fn get_area_to_search(
    connection_self: &IntraConnection,
    g_settings: &GuardianSettings,
) -> Vec<usize> {
    // Start with searching neuron and vicinity where pending is index
    let main_node = connection_self.get_index();
    let pending_node = connection_self.get_pending_index();
    let mut n_searched = 0;

    // This could be optimized to skip the vector
    let mut search = vec![];
    let mut callback_fn_add_search = |range: Range<isize>, reversed: bool| {
        for mut node_offset in range {
            if reversed {
                node_offset *= -1;
            }
            if n_searched >= g_settings.n_intraconnected_nodes_search {
                n_searched = 0;
                break;
            }
            let node_local_index = wrap_index(pending_node, node_offset, g_settings.n_nodes_per_neuron);
            if node_local_index == main_node {
                continue;
            }
            search.push(node_local_index);
            n_searched += 1;
        }
    };
    let range = 0..(g_settings.n_nodes_per_neuron as isize);  // 0 received here added here
    callback_fn_add_search(range.clone(), false);
    callback_fn_add_search(range, true);
    search
}

fn update_pending_connection(
    connection_self: &mut IntraConnection,
    model: &Model,
    precalculated: &[&Array1<f32>],
    nodes: &Array2<f32>,
    counter: &mut CounterIntraConnection,
    g_settings: &GuardianSettings,
) {
    let failed_previous = match counter.get_state(g_settings) {
        NodeState::Failed => {
            connection_self.reset_pending_forces();
            counter.reset();
            true
        },
        _ => false
    };

    match counter.get_state(g_settings) {
        NodeState::Searching => {
            let mut strongest_net_force = f32::MIN;
            let mut forces = (f32::MIN, f32::MIN);
            let mut strongest_node_index = 0;  // TODO: What if nothing is searched?
            let search: Vec<usize> = get_area_to_search(connection_self, g_settings);
            if search.len() == 0 {  // This search is stuck! It cannot move anywhere. That means the settings are bad
                connection_self.reset_pending();
                return;
            }
            for node_index in search {
                let (force_self, force_other) = get_pending_delta_forces(
                    node_index,
                    0.0,
                    0.0,
                    model,
                    precalculated,
                    nodes,
                );
                let net_force = force_self + force_other;
                if net_force > strongest_net_force {
                    forces = (force_self, force_other);
                    strongest_net_force = net_force;
                    strongest_node_index = node_index;
                }
            }
            let pending_index = connection_self.get_pending_index();  // copy
            connection_self.store_pending_index(strongest_node_index);
            if failed_previous && strongest_node_index == pending_index {
                // Stuck in a local maxima. Force reset
                connection_self.reset_pending();
            } else if pending_index == strongest_node_index {  // found local maximum, nothing higher around. Attempt connection
                counter.inc();
                connection_self.store_pending_forces(forces.0, forces.1);
            }
        }
        NodeState::Connecting => {
            let node_index_other = connection_self.get_pending_index();
            let (force_self, force_other) = connection_self.get_pending_forces();
            let (delta_force_self, delta_force_other) = get_pending_delta_forces(
                node_index_other,
                force_self,
                force_other,
                model,
                precalculated,
                nodes,
            );
            let updated_force_self = force_self + delta_force_self;
            let updated_force_other = force_other + delta_force_other;
            connection_self.store_pending_forces(
                updated_force_self,
                updated_force_other
            );
            let net_force = updated_force_self + updated_force_other;  // Store and load might not always be synced! used other values
            let net_force_to_beat = connection_self.get_net_force();
            if net_force > net_force_to_beat {
                counter.saturate();
            } else {
                counter.inc();
            }
        },
        NodeState::AttemptingTakeover => {
            // Nothing to compete to, just do it
            connection_self.move_pending_to_main();
            counter.reset();
        },
        NodeState::Failed => {
            connection_self.reset_pending();
            counter.reset();
        }
    }
}
