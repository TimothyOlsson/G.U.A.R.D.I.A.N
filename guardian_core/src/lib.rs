//! # G.U.A.R.D.I.A.N
//!
//! The general idea:
//!
//! The network is split into many small nodes (neurons). All neurons have a state of its components, which updated with the genome.
//! All neurons in the network share the same genome, which dictates how the state of each neuron should update.
//!
//! What the genome is any type of model (Neural Network or similar).
//!
//! The training of the network is not a network "HAS learned", but to get a network "that CAN learn"
//!
//! A neuron has the following components:
//!
//! * Nodes (N):
//!     Nodes are intraconnected inside a neuron with multiple connections (dendrites) as well as to other neurons (terminals)
//!     Another way to do this would be to split the nodes up to dendrite nodes (only interconnected) and terminal nodes (interconnected)
//!     There are some benifits with this, but makes the program and the model much more complex
//! * NeuronState (S):
//!     Models the sum of the activity of the nodes. It also functions as the state of the DNA
//! * InterConnections:
//!     Models connection between neurons (terminals)
//! * IntraConnections:
//!     Models dendrites and connections inside neurons
//!
//! The components should fulfill the following:,
//!
//! * Competitive learning <https://en.wikipedia.org/wiki/Competitive_learning>
//! * Coincidence detection: <https://en.wikipedia.org/wiki/Coincidence_detection_in_neurobiology>
//! * Long Term Potentiation (LTP): <https://en.wikipedia.org/wiki/Long-term_potentiation>
//! * Long Term Depression (LTD): <https://en.wikipedia.org/wiki/Long-term_depression>
//! * Spike-timing-dependent plasticity (STDP): <https://en.wikipedia.org/wiki/Spike-timing-dependent_plasticity>
//! * Hebbian Learning: <https://en.wikipedia.org/wiki/Hebbian_theory>
//! * Different synapse types: <https://en.wikipedia.org/wiki/Synapse#/media/File:Blausen_0843_SynapseTypes.png>
//! * + more
//!
//! Where the calculations are done does not matter (NPU, TPU, FPGA, GPU, CPU etc), as long the calculations are being
//! done in the correct order.
//!
//!
//! Order of calculations:
//!
//! NOTES:
//! For interconnections, calculating with S(A) & S(B) might be expensive. Perhaps only used D for internal usage?
//! N = Node
//! S = Strength
//! P = Pushback
//! G = Gradient
//! Seperate connections and inference! What if we only want to train reconnections? Not possible if they are merged!!
//!
//! 1: Update interconnected terminals.
//! * Model: S(A) & S(B) & N(A) & N(B) -> ΔN(A) (do twice, A and B switches places)
//! * Description: Connection between neurons. To prevent memory errors, the "highest" index of the terminals will do the calculation.
//!  When iterating over A, D(A) can be reused
//!
//! 2: Update intraconnections nodes
//! * Model: S & N(A) & N(B) -> ΔN(A)
//! * Description: Do for every connection and sum up the delta. NOTE: a working memory fitting the whole neuron needs to be used! Otherwise the
//! state will change when calculating. D input can be reused for all. N(A) can be reused for connections.
//!
//! 3: Update neuron states
//! * Model: S & N -> ΔD & ΔN
//! * Description: Do for every T and D. Needs to save a D, so it wont change during calculation. D input can be reused for all
//!
//! 4: Update interconnections:
//! * Model: S(A) & S(B) & N(A) & N(B) & S(A) & P(A) -> ΔS(A), ΔP(A), G(A)  (do twice, A and B switches places)
//! * Description: Do for all connections in its close vicinity. If not connected, S and P are 0.
//! S(A) is T(A) strength to T(B). P(A) is T(B) "refusal" of connection to T(A). Though it is stored in T(A)'s connection
//! The search algorithm should look for other terminals to connect to
//! Dispatch 0: Update connection and pending connection. Check where the pending connection should be
//! Dispatch 1: Do calculations and check if should connect to anything. If not, go to highest gradient. If yes, set self to "attempting connection"
//! Dispatch 2: All "attempting connections" will check if other is also attempting. If yes, ignore or reset self. Otherwise, do MAX of all attempting strenghts
//! Dispatch 3: All nodes check again if MAX attempt strength = self. If yes, do MAX index
//! Dispatch 4: The node with the highest index will win, since if there are multiple with same strength, only one can take the connection
//!
//! 5: Update intraconnections:
//! * Model: S & N(A) & N(B) & S(A) & P(A) -> ΔS(A), ΔP(A), G(A)
//! * Description: Do for all connections. As above, but might be able to do it all in one dispatch, because each internal workings should be done on one core / SM or similar.
//! The internal connections are different, since there can be multiple connections for each node.
//! Pushback is the other node rejecting the connecting node
//!
//! 6: Update network ports
//! * Model: S & P -> ΔP
//! * Description: TODO
//!
//! 7: Read from network ports
//! * Model: S & P -> ΔC
//! * Description: Will be 1 back in time!
//!
//! 8: Read input ports
//! * Model: TODO
//! * Description: TODO
//!
//! 9: Apply output ports
//! * Model: TODO
//! * Description: TODO


// TODO: List
// * Change delta to difference between max and min. If sum, it always stray to one state! (ex always +, big - will not be seen) DONE?
// * Change views to arrays (squeeze and such, makes more sense that way)
// * Add so that intraconnections not on the same
// * Add counter to interconnections

use std::default::Default;

// Modules
pub mod gpu;
pub mod cpu;
// TODO: Add more? TPU? FPGA?

pub mod visualization;


use crate::cpu::interface::{InterConnection, IntraConnection};

// NOTE: Is this needed? -> #[repr(C)]

/// Length of array MUST be divisible by 4
/// Settings for the neurons.
/// Any change of the size makes it incompatible with other genomes
/// Any change in connections is compatible, but "might" be behaving weird
#[derive(Clone)]
pub struct GuardianSettings {
    // Model
    pub node_size: usize,
    pub neuron_state_size: usize,
    pub n_nodes_per_neuron: usize,
    pub n_intraconnections_per_node: usize,

    // Searching
    pub n_interconnected_nodes_search: usize,  // TODO: Better name, -offset..offset
    pub n_interconnected_neuron_search: usize,
    pub n_intraconnected_nodes_search: usize,
    pub interconnection_max_connection_time: usize,
    pub intraconnection_max_connection_time: usize,

    // Network
    // TODO: Add sizes?

    // IO
    // TODO: Add io sizes? Dynamic?

    // Genome
    hidden_sizes: Vec<usize>
}

#[derive(Clone)]
pub struct NetworkSettings {
    pub n_neurons: usize,
    pub n_io_ports: usize,
    pub n_network_ports: usize,
    pub neurons_per_network_connection: usize,
}

impl Default for NetworkSettings {
    fn default() -> Self {
        Self {
            n_neurons: 64,
            n_io_ports: 0,
            n_network_ports: 0,
            neurons_per_network_connection: 16
        }
    }
}

impl NetworkSettings {
    pub fn downlevel_default() -> Self {
        Self {
            n_neurons: 16,
            n_io_ports: 0,
            n_network_ports: 0,
            neurons_per_network_connection: 0
        }
    }
}

impl Default for GuardianSettings {
    fn default() -> Self {
        Self {
            node_size: 128,
            neuron_state_size: 2048,
            n_nodes_per_neuron: 16,
            n_intraconnections_per_node: 4,
            n_interconnected_nodes_search: 4,
            n_interconnected_neuron_search: 1,
            n_intraconnected_nodes_search: 1,
            interconnection_max_connection_time: 8,
            intraconnection_max_connection_time: 8,
            hidden_sizes: vec![64, 64]
        }
    }
}

impl GuardianSettings {
    pub fn downlevel_default() -> Self {
        Self {
            node_size: 16,
            neuron_state_size: 32,
            n_nodes_per_neuron: 8,
            n_intraconnections_per_node: 4,
            n_interconnected_nodes_search: 4,
            n_interconnected_neuron_search: 1,
            n_intraconnected_nodes_search: 1,
            interconnection_max_connection_time: 8,
            intraconnection_max_connection_time: 8,
            hidden_sizes: vec![64, 64]
        }
    }
}

impl GuardianSettings {
    pub fn bytes_per_neuron(&self) -> usize {
        let nodes = self.node_size * self.n_nodes_per_neuron;
        let neuron_state = self.neuron_state_size;
        println!("Size for node states: {:?} per neuron", humansize::format_size(nodes, humansize::DECIMAL));
        println!("Size for neuron state: {:?} per neuron", humansize::format_size(neuron_state, humansize::DECIMAL));

        // Connections
        let inter_connections = std::mem::size_of::<InterConnection>() * self.n_nodes_per_neuron + 1;  // +1 byte for flags
        let intra_connections = std::mem::size_of::<IntraConnection>() * (self.n_nodes_per_neuron * self.n_intraconnections_per_node);
        println!("Size for inter_connections: {:?} per neuron", humansize::format_size(inter_connections, humansize::DECIMAL));
        println!("Size for intra_connections: {:?} per neuron", humansize::format_size(intra_connections, humansize::DECIMAL));

        let size = nodes + neuron_state + inter_connections + intra_connections;
        println!("Total size: {:?} per neuron", humansize::format_size(size, humansize::DECIMAL));
        size
    }
}

pub fn get_network_size(g_settings: &GuardianSettings, n_settings: &NetworkSettings) {
    let mut size = 0;
    size += g_settings.bytes_per_neuron() * n_settings.n_neurons;
    println!("Size of network: {:?}", humansize::format_size(size, humansize::DECIMAL));
    // TODO: Add ports!
}

pub fn get_genome_size() {

}