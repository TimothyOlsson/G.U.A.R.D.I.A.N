// Used instead of enums (not implemented in wgsl 2023-11-19)
const BLOCK_DISTRIBUTION: u32 = 0u;
const CYCLIC_DISTRIBUTION: u32 = 1u;

// ** Thread localization functions ** //

fn in_range(index: u32, range: Range) -> bool {
    return range.start <= index && index < range.stop;
}

// https://stackoverflow.com/questions/36688900/divide-an-uneven-number-between-threads
// https://stackoverflow.com/a/31116772
fn evenly_distribute_threads(thread_index: u32, range: Range, reminder: u32, distribution: u32) -> Range {
    // TODO: Add step_size
    var adjusted_range: Range = range;
    if reminder == 0u {
        // Nice, it is evenly split up
        return adjusted_range;
    }
    switch ( distribution ) {
        case BLOCK_DISTRIBUTION: {
            // Each worker takes one extra, until we are out
            if thread_index < reminder {
                var offset: u32 = thread_index * range.step_size;
                adjusted_range.start += offset;
                adjusted_range.stop += offset + range.step_size;
            } else {
                // The rest does not take any extra
                var offset: u32 = reminder * range.step_size;
                adjusted_range.start += offset;
                adjusted_range.stop += offset;
            }
        }
        case CYCLIC_DISTRIBUTION: {
            if thread_index <= reminder {
                // The first N threads take 1 more
                var offset: u32 = range.step_size;
                // Start is not touched
                adjusted_range.stop += offset;
            } else {
                // The rest does not take any extra
            }
        }
        default: {
            // noop, wrong input
        }
    }

    return adjusted_range;
}

// Where to work in the global memory
fn get_global_range(
    global_id: vec3<u32>,
    num_workgroups: vec3<u32>,
    worker_index: u32
) -> Range {
    // Possible overflow if the product does not fit in u32. Check before dispatch
    let total_number_of_workers = num_workgroups.x * num_workgroups.y * num_workgroups.z;
    let total_neurons = network_parameters.n_neurons;
    var n_neurons_per_worker: u32 = total_neurons / total_number_of_workers;
    let step_size = total_number_of_workers;  // Every other
    let start = worker_index;
    var stop: u32 = worker_index + step_size * n_neurons_per_worker;
    var range = Range(
        start,
        stop,
        step_size,
    );
    let n_neurons_per_worker_reminder = total_neurons % total_number_of_workers;
    range = evenly_distribute_threads(worker_index, range, n_neurons_per_worker_reminder, CYCLIC_DISTRIBUTION);
    return range;
}

// Where to work in a layer for a batch
fn get_layer_range(layer_size: u32, batch_size: u32) -> Range {
    let total_nodes = (layer_size * batch_size);
    let n_nodes_per_thread = total_nodes / WORKGROUP_SIZE;
    let step_size = 1u;
    var range = Range(
        thread_index * n_nodes_per_thread,
        (thread_index + 1u) * n_nodes_per_thread,
        step_size
    );
    let n_nodes_per_thread_reminder = total_nodes % WORKGROUP_SIZE;
    range = evenly_distribute_threads(thread_index, range, n_nodes_per_thread_reminder, BLOCK_DISTRIBUTION);
    return range;
}

struct Range {
    start: u32,
    stop: u32,
    step_size: u32,
}