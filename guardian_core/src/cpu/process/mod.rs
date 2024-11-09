use rayon::ThreadPool;
use tracing::trace;
use ndarray::{Array, ArrayView, ArrayView1, ArrayView2, ArrayViewMut1, Array1, Array2, Array3, Axis, s};

use crate::cpu::interface::{Network, InterConnection};
use super::interface::Flags;

pub mod interconnection_state;
pub mod intraconnection_state;
pub mod neuron_state;
pub mod interconnection_plasticity;
pub mod intraconnection_plasticity;
pub mod io_ports;

/// WIP: Starting with a naive approach
/// TODO: Move to network as impl?
pub fn update(network: &mut Network, pool: &ThreadPool) {
    trace!("Stage 1: Update io ports (network + input if core)");
    io_ports::update(network, pool);
    trace!("Stage 2: Update interconnected state");
    interconnection_state::update(network, pool);
    trace!("Stage 3: Update intraconnected state");
    intraconnection_state::update(network, pool);
    trace!("Stage 4: Update neuron state");
    neuron_state::update(network, pool);
    trace!("Stage 5: Update interconnections (plasticity)");
    interconnection_plasticity::update(network, pool);
    trace!("Stage 6: Update intraconnections (plasticity)");
    intraconnection_plasticity::update(network, pool);
}

#[derive(Debug)]
pub struct ConnectionParams {
    strength: f32,
    pushback: f32,
    gradient: f32
}

pub fn wrap_index(local_index: usize, offset: isize, max_index: usize) -> usize {
    (
        (local_index as isize + offset).rem_euclid(max_index as isize)  // Fast if pow2
    ) as usize
}

pub fn pack(value: f32) -> u8 {
    (value.min(1.0) * 255.0).max(0.0).round() as u8
}

// 1 / 255  <-- precomputed, multiply is faster than division
const DIVIDE_BY_255: f32 = 1.0 / 255.0;
pub fn unpack(packed: u8) -> f32 {
    (packed & 255) as f32 * DIVIDE_BY_255  // & 255 not needed, unless it is u32
}

pub fn unpack_array<D>(arr: ArrayView<u8, D>) -> Array<f32, D>
    where D: ndarray::Dimension
{
    arr.map(|v| unpack(*v))
}

pub fn pack_array<D>(arr: Array<f32, D>) -> Array<u8, D>
    where D: ndarray::Dimension
{
    arr.map(|v| pack(*v))
}

pub fn expand<'a>(arr: ArrayView1<'a, f32>) -> ArrayView2<'a, f32> {
    arr.insert_axis(Axis(0))
}

pub fn squeeze<'a>(arr: ArrayView2<'a, f32>) -> ArrayView1<'a, f32> {
    arr.remove_axis(Axis(0))
}

pub fn clip<D>(arr: Array<f32, D>, min_val: f32, max_val: f32) -> Array<f32, D>
    where D: ndarray::Dimension
{
    arr.mapv(|x| x.min(max_val).max(min_val))
}

pub fn min_array_inplace<'a>(arr1: &mut ArrayViewMut1<'a, f32>, arr2: ArrayView1<'a, f32>) {
    arr1.zip_mut_with(&arr2, |v1, v2| {
        *v1 = v1.min(*v2);
    });
}

pub fn max_array_inplace<'a>(arr1: &mut ArrayViewMut1<'a, f32>, arr2: ArrayView1<'a, f32>) {
    arr1.zip_mut_with(&arr2, |v1, v2| {
        *v1 = v1.max(*v2);
    });
}

fn get_node(neuron_index: usize, node_local_index: usize, nodes: &Array3<u8>) -> Array1<f32> {
    unpack_array(nodes.slice(s![neuron_index, node_local_index, ..]))
}

fn get_neuron_state(neuron_index: usize, neuron_states: &Array2<u8>) -> Array1<f32> {
    unpack_array(neuron_states.row(neuron_index))
}

fn get_inter_connection<'a>(neuron_index: usize, node_local_index: usize, inter_connections: &'a Array2<InterConnection>) -> &'a InterConnection {
    inter_connections.get((neuron_index, node_local_index)).unwrap()
}

fn get_inter_connection_flags<'a>(neuron_index: usize, node_local_index: usize, inter_connections_flags: &'a Array2<Flags>) -> &'a Flags {
    inter_connections_flags.get((neuron_index, node_local_index)).unwrap()
}

pub fn check_is_connected(node_a_index: usize, connection_b: &InterConnection) -> bool {
    connection_b.get_index() == node_a_index
}

pub fn value_to_array(value: f32) -> Array2<f32> {
    // TODO: Is this expensive to do? Any better way?
    Array2::from_elem((1, 1), value)
}

fn unpack_connection_model_output(output: Vec<Array2<f32>>) -> ConnectionParams {
    const DELTA_STRENGTH: usize = 0;
    const DELTA_PUSHBACK: usize = 1;
    const GRADIENT: usize = 2;
    let delta_strength = *output[DELTA_STRENGTH].first().unwrap();
    let delta_pushback = *output[DELTA_PUSHBACK].first().unwrap();
    let gradient = *output[GRADIENT].first().unwrap();
    ConnectionParams {
        strength: delta_strength,
        pushback: delta_pushback,
        gradient
    }
}

/// Allows for multiple process to work with the same vector without locking the whole vector
/// Is unsafe. Only use this if you know what you are doing.
unsafe fn non_locking_write(ptr: *mut u8, offset: usize, size: usize, data: &[u8]) {
    let dest_ptr = ptr.add(offset);  // what if 32 bit computer?
    let dest_slice = std::slice::from_raw_parts_mut(dest_ptr, size);
    dest_slice.copy_from_slice(data);
}

#[cfg(test)]
pub mod test {
    use super::*;

    use crate::{get_network_size, GuardianSettings, NetworkSettings};
    use crate::cpu::interface::{Genome, State};

    use rayon::ThreadPoolBuilder;
    use tracing::debug;
    use tracing::Level;
    use tracing_subscriber::FmtSubscriber;
    use rand::SeedableRng;

    fn add_tracing() {
        let subscriber = FmtSubscriber::builder()
            .with_max_level(Level::INFO)
            .finish();
        tracing::subscriber::set_global_default(subscriber).expect("setting default subscriber failed");
    }

    #[test]
    fn test_cpu_process() {
        add_tracing();
        let thread_count = std::thread::available_parallelism().unwrap().get();
        let pool = ThreadPoolBuilder::new().num_threads(thread_count).build().unwrap();
        let g_settings = GuardianSettings::downlevel_default();
        let n_settings = NetworkSettings::downlevel_default();
        get_network_size(&g_settings, &n_settings);
        let rng = rand::rngs::StdRng::seed_from_u64(1);
        let genome = Genome::new(&g_settings, &n_settings, Some(rng.clone()));
        let mut state = State::new(&g_settings, &n_settings);
        debug!("Randomizing");
        state.randomize(&g_settings, &n_settings, Some(rng));
        debug!("Randomizing done");
        //debug!("\n{:#?}", state.nodes);
        //debug!("\n{:#?}", state.neuron_states);
        //debug!("\n{:#?}", state.inter_connections);
        //debug!("\n{:#?}", state.inter_connections_flags);
        //debug!("\n{:#?}", state.intra_connections);
        let mut network = Network {
            state,
            genome,
            g_settings,
            n_settings
        };
        //loop {
            update(&mut network, &pool);
        //}
    }
}
