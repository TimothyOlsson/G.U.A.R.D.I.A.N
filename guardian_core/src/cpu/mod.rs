use interface::{CounterInterConnection, InterConnection};
use ndarray::{Array, ArrayView, ArrayView1, ArrayView2, ArrayViewMut1, Array1, Array2, Array3, Axis, s};

use crate::GuardianSettings;

pub mod interface;
pub mod process;
pub mod model;


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
    packed as f32 * DIVIDE_BY_255  // If above u8, a & 255 would needed
}

pub fn pack_with_negative(value: f32) -> u8 {
    ((value.clamp(-1.0, 1.0) + 1.0) * 127.5).round() as u8
}

pub const NEGATIVE_PACK_MIDPOINT: u8 = 128;  // Uneven, so it will be around 0.003921628
const DIVIDE_BY_127_5: f32 = 1.0 / 127.5;
pub fn unpack_with_negative(value: u8) -> f32 {
    value as f32 * DIVIDE_BY_127_5 - 1.0
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

fn get_other_inter_connection<'a>(connection_self: &InterConnection, inter_connections: &'a Array2<InterConnection>, g_settings: &GuardianSettings) -> &'a InterConnection {
    let node_other_global_index = connection_self.get_index();
    let (neuron_other_index, node_other_local_index) = node_global_to_local_index(node_other_global_index, g_settings);
    let connection_other = get_inter_connection(neuron_other_index, node_other_local_index, inter_connections);
    &connection_other
}

fn get_inter_connection_counter<'a>(neuron_index: usize, node_local_index: usize, inter_connection_counters: &'a Array2<CounterInterConnection>) -> &'a CounterInterConnection {
    inter_connection_counters.get((neuron_index, node_local_index)).unwrap()
}

pub fn check_is_connected(node_a_index: usize, connection_b: &InterConnection) -> bool {
    connection_b.get_index() == node_a_index
}

pub fn value_to_array(value: f32) -> Array2<f32> {
    // TODO: Is this expensive to do? Any better way?
    Array2::from_elem((1, 1), value)
}

/// Could use a struct, but that becomes a hassle when unpacking
pub fn node_global_to_local_index(node_global_index: usize, g_settings: &GuardianSettings) -> (usize, usize) {
    let neuron_index = node_global_index / g_settings.n_nodes_per_neuron;  // Faster if n_interconnected_nodes is the power of 2, then could do bit-shifting instead
    let node_local_index = node_global_index - (neuron_index * g_settings.n_nodes_per_neuron);
    (neuron_index, node_local_index)
}

pub fn node_local_to_global_index(neuron_index: usize, node_local_index: usize, g_settings: &GuardianSettings) -> usize {
    neuron_index * g_settings.n_nodes_per_neuron + node_local_index
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
    use crate::{get_network_size, GuardianSettings, NetworkSettings};
    use crate::cpu::interface::{Genome, State, Network};
    use crate::cpu::process::update;

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
