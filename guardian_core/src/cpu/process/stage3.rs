use std::time::Instant;

use tracing::{trace, debug};
use ndarray::Axis;
use itertools::multizip;
use ndarray::parallel::prelude::*;
use rayon::iter::ParallelBridge;

use crate::cpu::interface::Network;
use super::*;

const NEURON_STATE: usize = 0;
const NODE: usize = 1;
const DELTA_NEURON_STATE: usize = 0;
const DELTA_NODE: usize = 1;

pub fn update_neuron_states(network: &mut Network) {
    let nodes = &mut network.state.nodes;
    let neuron_states = &mut network.state.neuron_states;
    let genome = &network.genome;
    let model = &genome.neuron_state_update;
    let g_settings = &network.g_settings;

    let zipped = multizip(
        (
            neuron_states.rows_mut(),
            nodes.axis_iter_mut(Axis(0)),
        )
    );

    let now = Instant::now();
    zipped
    .into_iter()
    .par_bridge()
    .into_par_iter()
    .for_each(|(mut neuron_state_source, mut node_states_source)| {
        let neuron_state = unpack_array(neuron_state_source.view());
        let precalculated = [&model.precalculate(NEURON_STATE, neuron_state.view())];
        let mut delta_neuron_state_min = Array1::from_elem(g_settings.neuron_state_size, 0.0);
        let mut delta_neuron_state_max = Array1::from_elem(g_settings.neuron_state_size, 0.0);
        for mut node_source in node_states_source.rows_mut().into_iter() {
            let mut node = unpack_array(node_source.view());
            let inputs = [
                (NODE, expand(node.view()))
            ];
            let output = &model.forward_from_precalc(&inputs, &precalculated);
            let delta_neuron_state = squeeze(output[DELTA_NEURON_STATE].view());
            min_array_inplace(&mut delta_neuron_state_min.view_mut(), delta_neuron_state);
            max_array_inplace(&mut delta_neuron_state_max.view_mut(), delta_neuron_state);
            node = node + squeeze(output[DELTA_NODE].view());
            let node = pack_array(node);
            node_source.assign(&node);
        }
        let delta_neuron_state = delta_neuron_state_max + delta_neuron_state_min;
        debug!("{delta_neuron_state:?}");
        let updated_neuron_state = neuron_state + delta_neuron_state;

        let updated_neuron_state = pack_array(updated_neuron_state);
        neuron_state_source.assign(&updated_neuron_state);
    });
    trace!("It took {:?} to update neuron states", now.elapsed());
}


