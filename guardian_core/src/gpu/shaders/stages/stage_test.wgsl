// Here are the inputs
$PARAMS
$TERMINALS

// Rest of the imports
$RANGES
$UTILS

// Number of threads to use per workgroup
// Depends on the GPU on the possible / optimal setup
const WORKGROUP_SIZE: u32 = 256u;
// How many things to process in a batch. Higher -> more working memory required, but less idle threads per workgroup unit
const BATCH_SIZE: u32 = 1u;

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
}
