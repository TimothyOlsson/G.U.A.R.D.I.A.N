use std::sync::atomic::{AtomicI8, AtomicU32, AtomicU8, Ordering};

use ndarray::prelude::*;
use rand::distributions::{Distribution, Uniform};
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::{NetworkSettings, GuardianSettings};

use super::{pack_with_negative, unpack_with_negative};
use super::model::{Model, ModelSettings};

#[derive(Debug)]
pub struct CounterInterConnection(AtomicU8);

#[derive(Debug, Clone)]
pub struct CounterIntraConnection(u8);

#[derive(Debug, PartialEq)]
pub enum NodeState {
    Searching,
    Connecting,
    AttemptingTakeover,
    Failed,
}

/// Interconnection = nodes between neurons
/// Intraconnections = nodes within neurons
#[derive(Debug, Clone)]
pub struct State {
    pub nodes: Array3<u8>,
    pub neuron_states: Array2<u8>,
    pub inter_connections: Array2<InterConnection>,
    pub intra_connections: Array3<IntraConnection>,
    pub intra_connection_counters: Array3<CounterIntraConnection>,
    pub inter_connection_counters: Array2<CounterInterConnection>

}

#[derive(Clone)]
// TODO: What about IO ports and network ports?
pub struct Genome {
    // Guardian
    pub interconnected_node_state_update: Model,
    pub intraconnected_node_state_update: Model,
    pub neuron_state_update: Model,
    pub interconnections_update: Model,
    pub intraconnections_update: Model,

    // Interactions
    // TODO: Add genome for ports and other such things!
}

#[derive(Clone)]
pub struct Network {
    pub state: State,
    pub genome: Genome,
    pub g_settings: GuardianSettings,
    pub n_settings: NetworkSettings,
}

#[derive(Debug, Default)]
pub struct InterConnection {
    pub index: AtomicU32,
    pub pending_index: AtomicU32,  // The one with the highest index "wins"
    pub force_self: AtomicI8,
    // Why this is added:
    // In practice, the other InterConnection will have that value in force_self
    // However, adding this have some benifits:
    // * Alignment is divisible by 4 bytes
    // * All information needed is in the InterConnection during gradient search
    // * Can use this as a check if the connection is connected or not
    pub force_other: AtomicI8,
    pub pending_force_self: AtomicI8,
    pub pending_force_other: AtomicI8,
}

#[derive(Debug, Default, Clone)]
/// Reduced index -> limits nodes per neuron, but it saves 4 bytes per internal connection which adds up to quite a lot
pub struct IntraConnection {
    pub index: u16,
    pub pending_index: u16,
    pub force_self: i8,
    pub force_other: i8,
    pub pending_force_self: i8,
    pub pending_force_other: i8,
}

impl Clone for CounterInterConnection {
    fn clone(&self) -> Self {
        Self(self.0.load(Ordering::Relaxed).into())
    }
}

impl CounterInterConnection {
    pub fn new() -> Self {
        Self(AtomicU8::new(0))
    }

    pub fn get_value(&self) -> u8 {
        self.0.load(Ordering::Relaxed)
    }

    pub fn get_state(&self, g_settings: &GuardianSettings) -> NodeState {  // TODO: Set failed state from g_settings
        let value = self.0.load(Ordering::Relaxed);
        let threshold = g_settings.interconnection_max_connection_time as u8;
        if value >= threshold && value != 0xFF {
            return NodeState::Failed;
        }
        let state = match value {
            0xFF => NodeState::AttemptingTakeover,
            0xFE => NodeState::Failed,
            0x00 => NodeState::Searching,
            _ => NodeState::Connecting
        };
        state
    }

    pub fn inc(&self) {
        // Should be cleared before overflow
        self.0.fetch_add(1, Ordering::Relaxed);
    }

    pub fn saturate(&self) {
        self.0.store(0xFF, Ordering::Relaxed);
    }

    pub fn failed(&self) {
        // Back to seaching
        self.0.store(0xFE, Ordering::Relaxed);
    }

    pub fn reset(&self) {
        // Back to seaching
        self.0.store(0, Ordering::Relaxed);
    }
}

impl CounterIntraConnection {
    pub fn new() -> Self {
        Self(0)
    }

    pub fn get_value(&self) -> u8 {
        self.0
    }

    pub fn get_state(&self, g_settings: &GuardianSettings) -> NodeState {  // TODO: Set failed state from g_settings
        let value = self.0;
        let threshold = g_settings.intraconnection_max_connection_time as u8;
        if value >= threshold && value != 0xFF {
            return NodeState::Failed;
        }
        let state = match value {
            0xFF => NodeState::AttemptingTakeover,
            0xFE => NodeState::Failed,
            0x00 => NodeState::Searching,
            _ => NodeState::Connecting
        };
        state
    }

    pub fn inc(&mut self) {
        // Should be cleared before overflow
        self.0 += 1;
    }

    pub fn saturate(&mut self) {
        self.0 = 0xFF;
    }

    pub fn reset(&mut self) {
        // Back to seaching
        self.0 = 0;
    }
}



impl Clone for InterConnection {
    fn clone(&self) -> Self {
        let index = AtomicU32::new(self.index.load(Ordering::Relaxed));
        let pending_index = AtomicU32::new(self.pending_index.load(Ordering::Relaxed));
        let force_self =  AtomicI8::new(self.force_self.load(Ordering::Relaxed));
        let force_other =  AtomicI8::new(self.force_other.load(Ordering::Relaxed));
        let pending_force_self =  AtomicI8::new(self.pending_force_self.load(Ordering::Relaxed));
        let pending_force_other =  AtomicI8::new(self.pending_force_other.load(Ordering::Relaxed));
        Self {
            index,
            pending_index,
            force_self,
            force_other,
            pending_force_self,
            pending_force_other,
            ..*self
        }
    }
}

impl InterConnection {

    // Get values
    pub fn get_index(&self) -> usize {
        self.index.load(Ordering::Relaxed) as usize
    }

    pub fn get_pending_index(&self) -> usize {
        self.pending_index.load(Ordering::Relaxed) as usize
    }

    pub fn get_forces(&self) -> (f32, f32) {
        (
            unpack_with_negative(self.force_self.load(Ordering::Relaxed)),
            unpack_with_negative(self.force_other.load(Ordering::Relaxed))
        )
    }

    pub fn get_pending_forces(&self) -> (f32, f32) {
        (
            unpack_with_negative(self.pending_force_self.load(Ordering::Relaxed)),
            unpack_with_negative(self.pending_force_other.load(Ordering::Relaxed))
        )
    }

    pub fn get_raw_force_values(&self) -> (i8, i8) {
        (
            self.force_self.load(Ordering::Relaxed),
            self.force_other.load(Ordering::Relaxed)
        )
    }

    pub fn get_raw_pending_force_values(&self) -> (i8, i8) {
        (
            self.pending_force_self.load(Ordering::Relaxed),
            self.pending_force_other.load(Ordering::Relaxed)
        )
    }

    pub fn get_net_force(&self) -> f32 {
        unpack_with_negative(self.force_self.load(Ordering::Relaxed))
        +
        unpack_with_negative(self.force_other.load(Ordering::Relaxed))
    }

    pub fn get_net_pending_force(&self) -> f32 {
        unpack_with_negative(self.pending_force_self.load(Ordering::Relaxed))
        +
        unpack_with_negative(self.pending_force_other.load(Ordering::Relaxed))
    }

    // Store
    pub fn store_index(&self, index: usize) {
        assert!(index <= u32::MAX as usize);
        self.index.store(index as u32, Ordering::Relaxed);
    }

    pub fn store_pending_index(&self, index: usize) {
        assert!(index <= u32::MAX as usize);
        self.pending_index.store(index as u32, Ordering::Relaxed);
    }

    pub fn store_force_self(&self, force_self: f32) {
        self.force_self.store(pack_with_negative(force_self), Ordering::Relaxed);
    }

    pub fn store_force_other(&self, force_other: f32) {
        self.force_other.store(pack_with_negative(force_other), Ordering::Relaxed);
    }

    pub fn store_forces(&self, force_self: f32, force_other: f32) {
        self.force_self.store(pack_with_negative(force_self), Ordering::Relaxed);
        self.force_other.store(pack_with_negative(force_other), Ordering::Relaxed);
    }

    pub fn store_pending_forces(&self, force_self: f32, force_other: f32) {
        self.pending_force_self.store(pack_with_negative(force_self), Ordering::Relaxed);
        self.pending_force_other.store(pack_with_negative(force_other), Ordering::Relaxed);
    }

    // Competition
    pub fn add_maximum_force_self(&self, force_self: f32) {
        self.force_self.fetch_max(pack_with_negative(force_self), Ordering::Relaxed);
    }

    pub fn add_maximum_index(&self, index: usize) {
        self.index.fetch_max(index as u32, Ordering::Relaxed);
    }

    // Resets and successful connection
    pub fn move_pending_to_main(&self) {
        self.index.store(self.pending_index.load(Ordering::Relaxed), Ordering::Relaxed);
        self.force_self.store(self.pending_force_self.load(Ordering::Relaxed), Ordering::Relaxed);
        self.force_other.store(self.pending_force_other.load(Ordering::Relaxed), Ordering::Relaxed);
    }

    pub fn reset_pending_forces(&self) {
        self.pending_force_self.store(-127, Ordering::Relaxed);
        self.pending_force_other.store(-127, Ordering::Relaxed);
    }

    pub fn reset_pending(&self) {
        self.pending_index.store(self.index.load(Ordering::Relaxed), Ordering::Relaxed);
        self.pending_force_self.store(-127, Ordering::Relaxed);
        self.pending_force_other.store(-127, Ordering::Relaxed);
    }

    pub fn reset_main(&self) {
        // The main index is still there
        self.force_self.store(-127, Ordering::Relaxed);
        self.force_other.store(-127, Ordering::Relaxed);
    }
}

impl IntraConnection {

    pub fn get_index(&self) -> usize {
        self.index as usize
    }

    pub fn get_pending_index(&self) -> usize {
        self.pending_index as usize
    }

    pub fn get_forces(&self) -> (f32, f32) {
        (
            unpack_with_negative(self.force_self),
            unpack_with_negative(self.force_other)
        )
    }

    pub fn get_pending_forces(&self) -> (f32, f32) {
        (
            unpack_with_negative(self.pending_force_self),
            unpack_with_negative(self.pending_force_other)
        )
    }

    pub fn get_raw_pending_force_values(&self) -> (i8, i8) {
        (
            self.pending_force_self,
            self.pending_force_other
        )
    }

    pub fn get_raw_force_values(&self) -> (i8, i8) {
        (
            self.force_self,
            self.force_other
        )
    }

    pub fn get_net_force(&self) -> f32 {
        unpack_with_negative(self.force_self) + unpack_with_negative(self.force_other)
    }

    pub fn get_net_pending_force(&self) -> f32 {
        unpack_with_negative(self.pending_force_self) + unpack_with_negative(self.pending_force_other)
    }

    pub fn store_index(&mut self, index: usize) {
        assert!(index <= u16::MAX as usize);
        self.index = index as u16;
    }

    pub fn store_pending_index(&mut self, index: usize) {
        assert!(index <= u16::MAX as usize);
        self.pending_index = index as u16;
    }

    pub fn store_forces(&mut self, force_self: f32, force_other: f32) {
        self.force_self = pack_with_negative(force_self);
        self.force_other = pack_with_negative(force_other);
    }

    pub fn store_pending_forces(&mut self, force_self: f32, force_other: f32) {
        self.pending_force_self = pack_with_negative(force_self);
        self.pending_force_other = pack_with_negative(force_other);
    }

    pub fn move_pending_to_main(&mut self) {
        self.index = self.pending_index;
        self.force_self = self.pending_force_self;
        self.force_other = self.pending_force_other;
        self.reset_pending();
    }

    pub fn reset_pending(&mut self) {
        self.pending_index = self.index;
        self.pending_force_self = -127;
        self.pending_force_other = -127;
    }

    pub fn reset_pending_forces(&mut self) {
        self.pending_force_self = -127;
        self.pending_force_other = -127;
    }
}

impl State {
    pub fn new(g_settings: &GuardianSettings, n_settings: &NetworkSettings) -> Self {
        // NOTE: Zeros does not allocate anything!
        let nodes = Array3::ones((
            n_settings.n_neurons,
            g_settings.n_nodes_per_neuron,
            g_settings.node_size
        ));
        let neuron_states = Array2::ones((
            n_settings.n_neurons,
            g_settings.neuron_state_size
        ));
        let inter_connections = Array2::from_elem(
            (
                n_settings.n_neurons,
                g_settings.n_nodes_per_neuron,
            ),
            InterConnection::default()
        );
        let intra_connections = Array3::from_elem(
            (
                n_settings.n_neurons,
                g_settings.n_nodes_per_neuron,
                g_settings.n_intraconnections_per_node,
            ),
            IntraConnection::default()
        );
        let inter_connection_counters = Array2::from_shape_fn(
            (
                n_settings.n_neurons,
                g_settings.n_nodes_per_neuron,
            ),
            |_| { CounterInterConnection::new() }
        );
        let intra_connection_counters = Array3::from_shape_fn(
            (
                n_settings.n_neurons,
                g_settings.n_nodes_per_neuron,
                g_settings.n_intraconnections_per_node
            ),
            |_| { CounterIntraConnection::new() }
        );

        Self {
            nodes,
            neuron_states,
            inter_connections,
            intra_connections,
            inter_connection_counters,
            intra_connection_counters
        }
    }

    pub fn randomize(
        &mut self,
        g_settings: &GuardianSettings,
        n_settings: &NetworkSettings,
        mut rng: Option<StdRng>
    ) {
        if rng.is_none() {
            rng = Some(rand::rngs::StdRng::from_entropy());
        }
        let mut rng = rng.unwrap();

        // Mutate states
        let between_state = Uniform::<u8>::from(0..=255); // 256 is exclusive, so this is 0 to 255
        self.nodes.map_mut(|v| *v = between_state.sample(&mut rng));
        self.neuron_states.map_mut(|v| *v = between_state.sample(&mut rng));

        // Mutate connections
        let n_nodes_total = n_settings.n_neurons * g_settings.n_nodes_per_neuron;
        let between_inter_index = Uniform::<usize>::from(0..n_nodes_total);
        let between_intra_index = Uniform::<usize>::from(0..g_settings.n_nodes_per_neuron);
        let between_state = Uniform::<f32>::from(0.0..1.0);
        self.inter_connections.map_mut(|c| {
            c.store_index(between_inter_index.sample(&mut rng));
            c.store_pending_index(between_inter_index.sample(&mut rng));
            c.store_forces(between_state.sample(&mut rng), between_state.sample(&mut rng));
            c.store_pending_forces(between_state.sample(&mut rng), between_state.sample(&mut rng));
        });
        self.intra_connections.map_mut(|c| {
            c.store_index(between_intra_index.sample(&mut rng));
            c.store_pending_index(between_intra_index.sample(&mut rng));
            c.store_forces(between_state.sample(&mut rng), between_state.sample(&mut rng));
            c.store_pending_forces(between_state.sample(&mut rng), between_state.sample(&mut rng));
        });
    }
}

impl Genome {
    pub fn new(
        g_settings: &GuardianSettings,
        _n_settings: &NetworkSettings,  // TODO: Add models for IO
        mut rng: Option<StdRng>
    ) -> Self {
        if rng.is_none() {
            rng = Some(rand::rngs::StdRng::from_entropy());
        }
        let mut rng = rng.unwrap();

        // Interconnected
        let settings = ModelSettings::new(
            vec![
                g_settings.neuron_state_size,
                g_settings.neuron_state_size,
                g_settings.node_size,
                g_settings.node_size
            ],
            g_settings.hidden_sizes.clone(),
            vec![g_settings.node_size],
        ).unwrap();
        let interconnected_node_state_update = Model::new(settings, &mut rng).unwrap();

        // Intraconnected
        let settings = ModelSettings::new(
            vec![
                g_settings.neuron_state_size,
                g_settings.node_size,
                g_settings.node_size,
            ],
            g_settings.hidden_sizes.clone(),
            vec![
                g_settings.node_size,
                g_settings.node_size
            ],
        ).unwrap();
        let intraconnected_node_state_update = Model::new(settings, &mut rng).unwrap();

        // Neuron state
        let settings = ModelSettings::new(
            vec![
                g_settings.neuron_state_size,
                g_settings.node_size,
            ],
            g_settings.hidden_sizes.clone(),
            vec![
                g_settings.neuron_state_size,
                g_settings.node_size
            ],
        ).unwrap();
        let neuron_state_update = Model::new(settings, &mut rng).unwrap();

        // Interconnections
        let settings = ModelSettings::new(
            vec![
                g_settings.neuron_state_size,
                g_settings.neuron_state_size,
                g_settings.node_size,
                g_settings.node_size,
                1, // force_self
                1, // force_other
            ],
            g_settings.hidden_sizes.clone(),
            vec![1],  // delta_force_self,
        ).unwrap();
        let interconnections_update = Model::new(settings, &mut rng).unwrap();

        // Intraconnections
        let settings = ModelSettings::new(
            vec![
                g_settings.neuron_state_size,
                g_settings.node_size,
                g_settings.node_size,
                1, // force_self
                1, // force_other
            ],
            g_settings.hidden_sizes.clone(),
            vec![1, 1],  // delta_force_self
        ).unwrap();
        let intraconnections_update = Model::new(settings, &mut rng).unwrap();

        Self {
            interconnected_node_state_update,
            intraconnected_node_state_update,
            neuron_state_update,
            interconnections_update,
            intraconnections_update,
        }
    }
}