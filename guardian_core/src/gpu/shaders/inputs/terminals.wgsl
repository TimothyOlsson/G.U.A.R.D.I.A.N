const TERMINALS_PER_Neuron: u32 = $TERMINALS_PER_Neuronu;
const TERMINALS_ARRAY_SIZE: u32 = $TERMINALS_ARRAY_SIZEu;  // Values per array (except the last)
const TERMINALS_LAST_ARRAY_SIZE: u32 = $TERMINALS_LAST_ARRAY_SIZEu;  // Values per array (except the last)
const TERMINAL_SIZE: u32 = $TERMINAL_SIZEu;
alias Terminal = array<u32, (TERMINAL_SIZE / 4u)>;

//!buffers !last @group(GROUP) @binding(BINDING) var<storage, read_write> terminals_BUFFER_INDEX: array<Terminal, TERMINALS_ARRAY_SIZE>;
//!buffers last @group(GROUP) @binding(BINDING) var<storage, read_write> terminals_BUFFER_INDEX: array<Terminal, TERMINALS_LAST_ARRAY_SIZE>;

fn get_terminal_value(locator: DataLocator, value_index: u32) -> f32 {
    let adjusted_index = adjust_index(value_index);
    var packed_value: u32 = 0u;
    switch locator.buffer_index {
        default: { }
        //!cases read case BUFFER_INDEXu: { packed_value = terminals_BUFFER_INDEX[locator.item_index][adjusted_index.index]; }
    }
    let value = unpack(packed_value, adjusted_index.shift);
    return value;
}

fn write_terminal_value(locator: DataLocator, value_index: u32, value: u32) {
    switch locator.buffer_index {
        default: { }  // Do nothing
        //!cases write case BUFFER_INDEXu: { terminals_BUFFER_INDEX[locator.item_index][value_index] = value; }
    }
}