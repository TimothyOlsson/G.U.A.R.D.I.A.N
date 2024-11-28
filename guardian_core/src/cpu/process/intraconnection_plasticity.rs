use std::time::Instant;
use std::ops::Range;

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
            let already_connected = node_intra_connections.iter().map(|c| c.get_index()).collect();
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
                    &already_connected,
                );
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
    already_connected: &Vec<usize>,
    g_settings: &GuardianSettings,
) -> Vec<usize> {
    // Start with searching neuron and vicinity where pending is index
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
            if already_connected.contains(&node_local_index) {
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


/// Search neurons side-by-side and the connecting neuron. Same there
/// TODO: Split function? Very big input
fn update_pending_connection(
    connection_self: &mut IntraConnection,
    model: &Model,
    precalculated: &[&Array1<f32>],
    nodes: &Array2<f32>,
    counter: &mut CounterIntraConnection,
    g_settings: &GuardianSettings,
    //_intra_connections: &ArrayViewMut2<IntraConnection>,
    already_connected: &Vec<usize>,
) {
    let failed_previous = match counter.get_state(g_settings) {
        NodeState::Failed => {
            counter.reset();
            true
        },
        _ => false
    };

    match counter.get_state(g_settings) {
        NodeState::Searching => {
            let mut strongest_net_force = f32::MIN;
            let mut forces = (f32::MIN, f32::MIN);
            let mut strongest_node_index = 0;
            let search: Vec<usize> = get_area_to_search(connection_self, already_connected, g_settings);
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
            let pending_index = connection_self.get_pending_index();
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
            let node_index = connection_self.get_pending_index();
            let (force_self, force_other) = connection_self.get_pending_forces();
            let (delta_force_self, delta_force_other) = get_pending_delta_forces(
                node_index,
                force_self,
                force_other,
                model,
                precalculated,
                nodes,
            );
            connection_self.store_pending_forces(
                force_self + delta_force_self,
                force_other + delta_force_other
            );
            let net_force = connection_self.get_net_pending_force();
            let net_force_to_beat = connection_self.get_net_force();
            if net_force > net_force_to_beat {
                counter.saturate();
            }
        },
        _ => {}  // Should never be here
    }
}
