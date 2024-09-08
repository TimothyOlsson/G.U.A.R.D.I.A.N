// Here are the inputs
$PARAMS
$TERMINALS
$NEURON_STATES
$SYNAPSES

// Rest of the imports
$RANGES
$PACKING_UNPACKING

// Number of threads to use per workgroup
// Depends on the GPU on the possible / optimal setup
const WORKGROUP_SIZE: u32 = 256u;
// How many things to process in a batch. Higher -> more working memory required, but less idle threads per workgroup unit
const BATCH_SIZE: u32 = 1u;

// ** Model functions ** //

fn process_neuron(neuron_index: u32) {
    model_init(neuron_index);  // Initial pre-calculations to reuse
    for ( var terminal_index: u32 = 0u; terminal_index < TERMINALS_PER_Neuron; terminal_index++ ) {
        process_terminal(neuron_index, terminal_index);
        // TODO: Handle BATCH_SIZE
    }
}

fn process_terminal(neuron_index: u32, terminal_index: u32) {

    // Calculate self and store it in output
    model_run(neuron_index, terminal_index);

    // Calculate other and store it to output
    model_run(synapse_self.neuron_index, synapse_self.terminal_index);

    // Write output to global storage
    // TODO
}

fn model_run(neuron_index: u32, terminal_index: u32) {
    // Start with first layer, where all values are fetched from the states


    // We are now at layer1 (layer0 done)
    var previous_layer_size: u32 = model_first_layer_size;
    for ( var layer_index: u32 = 1u; layer_index < model.n_layers; layer_index++ ) {
        let layer_size = model.layer_sizes[layer_index];
        model_process_layer(layer_index, layer_size, previous_layer_size);
        previous_layer_size = layer_size;
        storageBarrier();  // Sync all threads
    }

    // TODO: Handle if 1 layer only

    // Store to output
}

fn model_process_layer(
    layer_index: u32,
    layer_size: u32,
    previous_layer_size: u32
) {
    // Will be 0 or 1
    let current_working_memory_layer = layer_index & 1u;  // Same as layer_index % 2
    let node_range = model_node_ranges[layer_index];
    for ( var node_index: u32 = node_range.start; node_index < node_range.stop; node_index++ ) {
        model_process_node(node_index, previous_layer_size, current_working_memory_layer);
    }
}

fn model_process_node(
    node_index: u32,
    layer_index: u32,
    previous_layer_size: u32,
    current_working_memory_layer: u32
) {
    // Get offsets for bias and weight
    let bias_offset = model_bias_offset[layer_index] + node_index;
    let weight_offset = model_weights_offset[layer_index] + node_index * previous_layer_size;

    // Get offsets for previous and current layer
    let previous_working_memory_layer = current_working_memory_layer ^ 1; // result is 1 if value is 0, or 0 if value is 1
    let previous_working_memory_layer_offset = working_memory_layer_offset + previous_working_memory_layer * model_largest_layer;
    let current_working_memory_layer_offset = working_memory_layer_offset + current_working_memory_layer * model_largest_layer;

    var value: f32 = model_bias[bias_offset];  // Start with bias
    for ( var i: u32 = 0u; i < previous_layer_size; i++ ) {
        let weight = model_weights[weight_offset + i];
        let prev_value = working_memory[previous_working_memory_layer_offset + i];  // TODO: Wrong index, fix
        value += prev_value * weight;
    }

    // Write to working memory
    working_memory[current_working_memory_layer_offset + node_index] = limited_relu(value);
}


// Why initialization makes things faster is that some calculations can be reused,
// and there is no need to recalculate things that are used for all terminals.
// The initialization is the neuronstate self -> connected and connected -> self
// Dendrites could be added as well, but I think they should be seperated, so there will only be terminal <-> dendrite
// The output will be:
// delta-strength - delta-terminal
// Both terminals needs to be updated at the same time, so there are some output chunks in working memory
fn model_init(neuron_index: u32) {
    let initial_range = model_node_ranges[0];
    let first_layer_size = model.layer_sizes[0];
    for ( var node_index: u32 = initial_range.start; node_index < initial_range.stop; node_index++ ) {
        let bias = model_bias[node_index];
        var value_to: f32 = bias;  // self -> connected
        var value_from: f32 = bias;  // connected -> self
        for ( var i: u32 = 0u; i < NEURON_STATE_SIZE; i++ ) {
            let fixed_index = get_shift_and_index(i);

            // This is the value which is going to be pre-calculated
            let value = unpack(
                neuron_states[neuron_index][fixed_index.index],
                fixed_index.shift
            );

            // Order in weights0 is Column major and looks like this:
            // Column1: *neuronstate* - terminal - strength - *connected-neuronstate* - connected-terminal - connected-strength
            // Column2: *neuronstate* - terminal - strength - *connected-neuronstate* - connected-terminal - connected-strength
            // Weight offset is for the next column

            // First values for NEURON_STATE_SIZE is for self -> connected
            let weight_offset = first_layer_size * node_index + i;
            let weight_to = model_weights[weight_offset];

            // Skip terminal and strength
            let weight_from = model_weights[weight_offset + TERMINAL_SIZE + 1u];

            value_to += value * weight_to;
            value_from += value * weight_from;
        }
        // It is f32, but represented as u32
        working_memory[working_memory_worker_offset + node_index] = value_from;
        working_memory[working_memory_worker_offset + node_index + first_layer_size] = value_to;
    }
    storageBarrier();  // Sync all threads
}

// ** Main ** //

@compute
@workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_index) local_invocation_index: u32,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    // Save this for future uses
    thread_index = local_invocation_index;

    // Think of a 3d rectangle, but linearized
    worker_index =
        workgroup_id.x +
        (workgroup_id.y * num_workgroups.x) +
        (workgroup_id.z * num_workgroups.x * num_workgroups.y);

    // Calculate where to work in the model, just once
    model_first_layer_size = model.layer_sizes[0];
    model_output_size = model.layer_sizes[model.n_layers - 1];
    var bias_offset: u32 = 0u;
    var weights_offset: u32 = 0u;
    // Initial size is the input of the model
    var previous_layer_size: u32 = 2u * NEURON_STATE_SIZE + 2u * TERMINAL_SIZE + 2u;
    for ( var layer_index: u32 = 0u; layer_index < model.n_layers; layer_index++ ) {
        let layer_size = model.layer_sizes[layer_index];
        model_node_ranges[layer_index] = get_layer_range(layer_size, BATCH_SIZE);
        model_largest_layer = max(model_largest_layer, layer_size);
        model_bias_offset[layer_index] = bias_offset;
        model_weights_offset[layer_index] = weights_offset;

        bias_offset += layer_size;
        weight_offset += previous_layer_size * layer_size;
        previous_layer_size = layer_size;
    }

    // Calculate how much working memory is needed, then set the offset accordingly depending on how much every workgroup needs
    let init_memory_size = 2u * first_layer_size;  // Both self -> connected and connected -> self
    let layer_memory_size = (2u * model_largest_layer) * BATCH_SIZE;  // Second chunk two chunks with 1 layer each, so it can "flip-flop" between the chunks
    let output_memory_size = (2u * output_size) * BATCH_SIZE;  // Lastly, it should store the output of self and connected

    working_memory_worker_size = init_memory_size + layer_memory_size + output_memory_size;
    working_memory_worker_offset = worker_index * working_memory_worker_size;
    working_memory_init_offset = working_memory_worker_offset;
    working_memory_layer_offset = working_memory_init_offset + init_memory_size;
    working_memory_output_offset = working_memory_layer_offset + layer_memory_size;

    // Calculate which neurons to work with
    let global_range = get_global_range(global_id, num_workgroups, worker_index);

    // Take one neuron at a time
    for ( var neuron_index: u32 = global_range.start; neuron_index < global_range.stop; neuron_index += global_range.step_size ) {
        process_neuron(neuron_index);
    }
}
