// This will be the same for all pipelines
// Minimize the amount of groups needed, clump this together
// Could optimize so that some of the bindings in group 0 is used, but that
// just to squeeze out a bit more memory. Perhaps not needed
@group(0) @binding(0) var<storage> network_parameters: NetworkParameters;
@group(0) @binding(1) var<storage> model: ModelParameters;
@group(0) @binding(2) var<storage, read> model_weights: array<f32>;  // Column major
@group(0) @binding(3) var<storage, read> model_bias: array<f32>;  // Column major
@group(0) @binding(4) var<storage, read_write> working_memory: array<f32>;  // bitcast<u32>(float_value) if u32 is needed

// Assuming the default limits:
// max_bind_groups = 4
// max_bindings_per_bind_group = 1000
// max_dynamic_storage_buffers_per_pipeline_layout = 4 (not needed)

struct NetworkParameters {
    n_neurons: u32,
}

const MAX_LAYERS: u32 = $MAX_LAYERS;

// Fully-connected neural network
struct ModelParameters {
    n_layers: u32,
    layer_sizes: array<u32, MAX_LAYERS>,
}


// All are uninitiated. Initialize in main
// Typically 65535 bytes maximum in total
// Use these to calculate once where to work,
// to skip passing these between functions

// Worker values
var<private> thread_index: u32;
var<private> worker_index: u32;

// These could be constant or coupled to the model parameters
// However, for now it is calculated and added to private for
// faster access to the values
var<private> model_weights_offset: array<Range, MAX_LAYERS>;
var<private> model_bias_offset: array<Range, MAX_LAYERS>;
var<private> model_largest_layer: u32;
var<private> model_first_layer_size: u32;
var<private> model_input_size: u32;
var<private> model_output_size: u32;

// Working memory values
var<private> working_memory_worker_size: u32;
var<private> working_memory_worker_offset: u32;
var<private> working_memory_init_offset: u32;
var<private> working_memory_layer_offset: u32;
var<private> working_memory_output_offset: u32;

// Model values
var<private> model_node_ranges: array<Range, MAX_LAYERS>;  // Where to work in each layer
