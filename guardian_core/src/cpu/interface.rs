use std::sync::atomic::{AtomicU16, AtomicU32, AtomicU8, Ordering};

use ndarray::prelude::*;
use rand::distributions::{Distribution, Uniform};

use crate::{NetworkSettings, GuardianSettings};

use super::process::{pack, unpack};
use super::model::{Model, ModelSettings};

#[derive(Debug)]
pub struct Flags(AtomicU8);

#[allow(non_snake_case)]
pub mod Flag {
    pub const CONNECTING: u8 = 1 << 7;
    pub const FAILED: u8 = 1 << 6;  // Could be skipped with better logic
    // TODO: Add more if needed
}

/// Interconnection = between neurons (T&T)
/// Intraconnections = within neuron (T&T or D&T or D&D)
#[derive(Debug, Clone)]
pub struct State {
    pub nodes: Array3<u8>,
    pub neuron_states: Array2<u8>,
    pub inter_connections: Array2<InterConnection>,
    pub intra_connections: Array3<IntraConnection>,
    pub inter_connections_flags: Array2<Flags>
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
    pub strength: AtomicU8,
    pub pushback: AtomicU8,
    pub pending_strength: AtomicU8,
    pub pending_pushback: AtomicU8,
}

#[derive(Debug, Default)]
/// Reduced index -> limits nodes per neuron, but it saves 4 bytes per internal connection which adds up to quite a lot
pub struct IntraConnection {
    pub index: AtomicU16,
    pub pending_index: AtomicU16,
    pub strength: AtomicU8,
    pub pushback: AtomicU8,
    pub pending_strength: AtomicU8,
    pub pending_pushback: AtomicU8,
}

impl Clone for Flags {
    fn clone(&self) -> Self {
        Self(self.0.load(Ordering::Relaxed).into())
    }
}

impl Flags {
    pub fn new() -> Self {
        Self(AtomicU8::new(0))
    }

    pub fn add_flag(&self, flag: u8) {
        self.0.fetch_or(flag, Ordering::Relaxed);
    }

    pub fn remove_flag(&self, flag: u8) {
        self.0.fetch_and(!flag, Ordering::Relaxed);
    }

    pub fn check_flag(&self, flag: u8) -> bool {
        (self.0.load(Ordering::Relaxed) & flag) != 0
    }

    pub fn increment_counter(&self) {

    }

    pub fn reset_counter(&self) {

    }
}



impl Clone for InterConnection {
    fn clone(&self) -> Self {
        let index = AtomicU32::new(self.index.load(Ordering::Relaxed));
        let pending_index = AtomicU32::new(self.pending_index.load(Ordering::Relaxed));
        let strength = AtomicU8::new(self.strength.load(Ordering::Relaxed));
        let pushback = AtomicU8::new(self.pushback.load(Ordering::Relaxed));
        let pending_strength = AtomicU8::new(self.pending_strength.load(Ordering::Relaxed));
        let pending_pushback = AtomicU8::new(self.pending_pushback.load(Ordering::Relaxed));
        Self {
            index,
            pending_index,
            strength,
            pushback,
            pending_strength,
            pending_pushback,
            ..*self
        }
    }
}

impl Clone for IntraConnection {
    fn clone(&self) -> Self {
        let index = AtomicU16::new(self.index.load(Ordering::Relaxed));
        let pending_index = AtomicU16::new(self.pending_index.load(Ordering::Relaxed));
        let strength = AtomicU8::new(self.strength.load(Ordering::Relaxed));
        let pushback = AtomicU8::new(self.pushback.load(Ordering::Relaxed));
        let pending_strength = AtomicU8::new(self.pending_strength.load(Ordering::Relaxed));
        let pending_pushback = AtomicU8::new(self.pending_pushback.load(Ordering::Relaxed));
        Self {
            index,
            pending_index,
            strength,
            pushback,
            pending_strength,
            pending_pushback,
            ..*self
        }
    }
}

impl InterConnection {
    pub fn get_index(&self) -> usize {
        self.index.load(Ordering::Relaxed) as usize
    }

    pub fn get_pending_index(&self) -> usize {
        self.pending_index.load(Ordering::Relaxed) as usize
    }

    pub fn get_strength_and_pushback(&self) -> (f32, f32) {
        let strength = unpack(self.strength.load(Ordering::Relaxed));
        let pushback = unpack(self.pushback.load(Ordering::Relaxed));
        (strength, pushback)
    }

    pub fn get_pending_strength_and_pushback(&self) -> (f32, f32) {
        let strength = unpack(self.pending_strength.load(Ordering::Relaxed));
        let pushback = unpack(self.pending_pushback.load(Ordering::Relaxed));
        (strength, pushback)
    }

    pub fn get_connection_bond_force(&self) -> f32 {
        let (strength, pushback) = self.get_strength_and_pushback();
        strength - pushback
    }

    pub fn get_pending_connection_bond_force(&self) -> f32 {
        let (strength, pushback) = self.get_pending_strength_and_pushback();
        strength - pushback
    }

    pub fn get_bond_force(&self) -> f32 {
        let bond_force = unpack(self.strength.load(Ordering::Relaxed));
        bond_force
    }

    pub fn store_index(&self, index: usize) {
        assert!(index <= u32::MAX as usize);
        self.index.store(index as u32, Ordering::Relaxed);
    }

    pub fn store_pending_index(&self, index: usize) {
        assert!(index <= u32::MAX as usize);
        self.pending_index.store(index as u32, Ordering::Relaxed);
    }

    pub fn store_strength_and_pushback(&self, strength: f32, pushback: f32) {
        self.strength.store(pack(strength), Ordering::Relaxed);
        self.pushback.store(pack(pushback), Ordering::Relaxed);
    }

    pub fn store_pending_strength_and_pushback(&self, strength: f32, pushback: f32) {
        self.pending_strength.store(pack(strength), Ordering::Relaxed);
        self.pending_pushback.store(pack(pushback), Ordering::Relaxed);
    }

    pub fn store_bond_force(&self, bond_force: f32) {
        self.strength.store(pack(bond_force), Ordering::Relaxed);  // Will re-use strength
    }

    pub fn add_maximum_bond_force(&self, bond_force: f32) {
        self.strength.fetch_max(pack(bond_force), Ordering::Relaxed);
    }

    pub fn add_maximum_index(&self, index: usize) {
        self.index.fetch_max(index as u32, Ordering::Relaxed);
    }

    pub fn move_pending_to_main(&self) {
        self.index.store(self.pending_index.load(Ordering::Relaxed), Ordering::Relaxed);
        self.strength.store(self.pending_strength.load(Ordering::Relaxed), Ordering::Relaxed);
        self.pushback.store(self.pending_pushback.load(Ordering::Relaxed), Ordering::Relaxed);
    }

    pub fn reset_pending(&self) {
        self.pending_index.store(self.index.load(Ordering::Relaxed), Ordering::Relaxed);
        self.pending_strength.store(0, Ordering::Relaxed);
        self.pending_pushback.store(0, Ordering::Relaxed);
    }
}

impl IntraConnection {
    pub fn get_index(&self) -> usize {
        self.index.load(Ordering::Relaxed) as usize
    }

    pub fn get_pending_index(&self) -> usize {
        self.pending_index.load(Ordering::Relaxed) as usize
    }

    pub fn get_strength_and_pushback(&self) -> (f32, f32) {
        let strength = unpack(self.strength.load(Ordering::Relaxed));
        let pushback = unpack(self.pushback.load(Ordering::Relaxed));
        (strength, pushback)
    }

    pub fn get_pending_strength_and_pushback(&self) -> (f32, f32) {
        let strength = unpack(self.pending_strength.load(Ordering::Relaxed));
        let pushback = unpack(self.pending_pushback.load(Ordering::Relaxed));
        (strength, pushback)
    }

    pub fn get_connection_bond_force(&self) -> f32 {
        let (strength, pushback) = self.get_strength_and_pushback();
        strength - pushback
    }

    pub fn get_pending_connection_bond_force(&self) -> f32 {
        let (strength, pushback) = self.get_pending_strength_and_pushback();
        strength - pushback
    }

    pub fn get_bond_force(&self) -> f32 {
        let bond_force = unpack(self.strength.load(Ordering::Relaxed));
        bond_force
    }

    pub fn store_index(&self, index: usize) {
        assert!(index <= u16::MAX as usize);
        self.index.store(index as u16, Ordering::Relaxed);
    }

    pub fn store_pending_index(&self, index: usize) {
        assert!(index <= u16::MAX as usize);
        self.pending_index.store(index as u16, Ordering::Relaxed);
    }

    pub fn store_strength_and_pushback(&self, strength: f32, pushback: f32) {
        self.strength.store(pack(strength), Ordering::Relaxed);
        self.pushback.store(pack(pushback), Ordering::Relaxed);
    }

    pub fn move_pending_to_main(&self) {
        self.index.store(self.pending_index.load(Ordering::Relaxed), Ordering::Relaxed);
        self.strength.store(self.pending_strength.load(Ordering::Relaxed), Ordering::Relaxed);
        self.pushback.store(self.pending_pushback.load(Ordering::Relaxed), Ordering::Relaxed);
    }

    pub fn store_pending_strength_and_pushback(&self, strength: f32, pushback: f32) {
        self.pending_strength.store(pack(strength), Ordering::Relaxed);
        self.pending_pushback.store(pack(pushback), Ordering::Relaxed);
    }

    pub fn reset_pending(&self) {
        self.pending_index.store(self.index.load(Ordering::Relaxed), Ordering::Relaxed);
        self.pending_strength.store(0, Ordering::Relaxed);
        self.pending_pushback.store(0, Ordering::Relaxed);
    }
}

impl State {
    pub fn new(g_settings: &GuardianSettings, n_settings: &NetworkSettings) -> Self {
        // NOTE: Zeros does not allocate anything!
        let nodes = Array3::ones((
            n_settings.n_neurons,
            g_settings.n_nodes,
            g_settings.node_size
        ));
        let neuron_states = Array2::ones((
            n_settings.n_neurons,
            g_settings.neuron_state_size
        ));
        let inter_connections = Array2::from_elem(
            (
                n_settings.n_neurons,
                g_settings.n_nodes,
            ),
            InterConnection::default()
        );

        let inter_connections_flags = Array2::from_shape_fn(
            (
                n_settings.n_neurons,
                g_settings.n_nodes,
            ),
            |_| { Flags::new() }
        );
        let intra_connections = Array3::from_elem(
            (
                n_settings.n_neurons,
                g_settings.n_nodes,
                g_settings.n_intraconnections_per_node,
            ),
            IntraConnection::default()
        );
        Self {
            nodes,
            neuron_states,
            inter_connections,
            intra_connections,
            inter_connections_flags
        }
    }

    pub fn randomize(&mut self, g_settings: &GuardianSettings, n_settings: &NetworkSettings) {
        let mut rng = rand::thread_rng();

        // Mutate states
        let between_state = Uniform::<u8>::from(0..=255); // 256 is exclusive, so this is 0 to 255
        self.nodes.map_mut(|v| *v = between_state.sample(&mut rng));
        self.neuron_states.map_mut(|v| *v = between_state.sample(&mut rng));

        // Mutate connections
        let n_nodes_total = n_settings.n_neurons * g_settings.n_nodes;
        let between_inter_index = Uniform::<usize>::from(0..n_nodes_total);
        let between_intra_index = Uniform::<usize>::from(0..g_settings.n_nodes);
        let between_state = Uniform::<f32>::from(0.0..1.0);
        self.inter_connections.map_mut(|c| {
            c.store_index(between_inter_index.sample(&mut rng));
            c.store_pending_index(between_inter_index.sample(&mut rng));
            c.store_strength_and_pushback(between_state.sample(&mut rng), between_state.sample(&mut rng));
            c.store_pending_strength_and_pushback(between_state.sample(&mut rng), between_state.sample(&mut rng));
        });
        self.intra_connections.map_mut(|c| {
            c.store_index(between_intra_index.sample(&mut rng));
            c.store_pending_index(between_intra_index.sample(&mut rng));
            c.store_strength_and_pushback(between_state.sample(&mut rng), between_state.sample(&mut rng));
            c.store_pending_strength_and_pushback(between_state.sample(&mut rng), between_state.sample(&mut rng));
        });
    }
}

impl Genome {
    pub fn new(g_settings: &GuardianSettings, n_settings: &NetworkSettings) -> Self {
        let default_hidden_sizes = vec![256, 256];

        // Interconnected
        let settings = ModelSettings::new(
            vec![
                g_settings.neuron_state_size,
                g_settings.neuron_state_size,
                g_settings.node_size,
                g_settings.node_size
            ],
            default_hidden_sizes.clone(),
            vec![g_settings.node_size],
        ).unwrap();  // We know that this is ok
        let interconnected_node_state_update = Model::new(settings).unwrap();

        // Intraconnected
        let settings = ModelSettings::new(
            vec![
                g_settings.neuron_state_size,
                g_settings.node_size,
                g_settings.node_size,
            ],
            default_hidden_sizes.clone(),
            vec![
                g_settings.node_size,
                g_settings.node_size
            ],
        ).unwrap();  // We know that this is ok
        let intraconnected_node_state_update = Model::new(settings).unwrap();

        // Neuron state
        let settings = ModelSettings::new(
            vec![
                g_settings.neuron_state_size,
                g_settings.node_size,
            ],
            default_hidden_sizes.clone(),
            vec![
                g_settings.neuron_state_size,
                g_settings.node_size
            ],
        ).unwrap();  // We know that this is ok
        let neuron_state_update = Model::new(settings).unwrap();

        // Interconnections
        let settings = ModelSettings::new(
            vec![
                g_settings.neuron_state_size,
                g_settings.neuron_state_size,
                g_settings.node_size,
                g_settings.node_size,
                1, // Strength
                1, // Pushback
            ],
            default_hidden_sizes.clone(),
            vec![
                1, 1, 1,
                // Strength, Pushback, Gradient
            ],
        ).unwrap();  // We know that this is ok
        let interconnections_update = Model::new(settings).unwrap();

        // Intraconnections
        let settings = ModelSettings::new(
            vec![
                g_settings.neuron_state_size,
                g_settings.node_size,
                g_settings.node_size,
                1, // Strength
                1, // Pushback
            ],
            default_hidden_sizes.clone(),
            vec![
                1, 1, 1,
                // Strength, Pushback, Gradient
            ],
        ).unwrap();  // We know that this is ok
        let intraconnections_update = Model::new(settings).unwrap();

        Self {
            interconnected_node_state_update,
            intraconnected_node_state_update,
            neuron_state_update,
            interconnections_update,
            intraconnections_update,
        }
    }
}