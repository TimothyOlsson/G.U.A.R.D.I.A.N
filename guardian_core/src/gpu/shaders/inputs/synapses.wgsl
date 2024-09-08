const ARRAY_SIZE: u32 = $ARRAY_SIZEu;  // Values per array (except the last)
const LAST_ARRAY_SIZE: u32 = $LAST_ARRAY_SIZEu;  // Values per array (except the last)
struct Synapse {
    terminal_index: u32,
    strength: f32,
}
alias Data = Synapse;

// Will add the following here:
// @group(GROUP) @binding(BINDING) var<storage, read> buffer_BUFFER_INDEX: array<DATA, ARRAY_SIZE>;
// ...
// ... LAST_ARRAY_SIZE>;
$BUFFERS

fn get_synapse(index: u32) -> Synapse {
    let buffer_index = index / ARRAY_SIZE;
    let value_index = index % ARRAY_SIZE;
    switch buffer_index {
        default: { return buffer_0[0]; }
        $CASES
    }
}

// ** FUNCTIONS ** //

fn check_if_synapse_connected(terminal_index: u32, synapse: Synapse) {
    if terminal_index != synapse.terminal_index {
        return false;
    } else {
        return true;
    }
}