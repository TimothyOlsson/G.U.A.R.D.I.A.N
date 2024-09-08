@group(0)
@binding(0)
var<storage, read> prev_state: array<FakeStruct>;

@group(1)
@binding(0)
var<storage, read> next_state: array<FakeStruct>;

@group(2)
@binding(0)
var<storage, read> working_memory: array<WorkingMemory>;

@group(3)
@binding(0)
var<storage, read> neuron_state: array<NeuronState>;

const MAX_DISPATCH: u32 = 65535u;
//const MAX_BUFFER_SIZE: u32 = 134217728u;  // 128 MiB

struct FakeStruct {
    data: array<u32, 128>,
};

struct WorkingMemory {
    data: array<u32, 128>,
};

struct NeuronState {
    data: array<u32, 1024>,
};

// Could be modified to work as a u64
struct ValueLocator {
    total_index: u32
};

// Pack the value and shift it appropriately
fn pack(value: f32, shift: u32) -> u32 {
    var clamped: u32 = u32(
        normal_round(
            max(
                0.0,
                min(value, 1.0) * 255.0
            )
        )
    );
    var packed: u32 = (clamped & 255u) << (8u * shift);
    return packed;
}

// Division is expensive, multiplication is fast
// Precalculate division and multiply -> fast
const DIVIDE_BY_255: f32 = 1.0 / 255.0;
fn unpack(packed: u32, shift: u32) -> f32 {
    var unpacked: f32 = f32(packed >> (8u * shift) & 255u) * DIVIDE_BY_255;
    return unpacked;
}

fn create_mask(shift: u32) -> u32 {
    // A mask with the lower 8 bits set, shifted to overlay correctly
    // Invert, so we have a "hole"
    var mask: u32 = 0xFFu << shift * 8u;
    mask = ~mask;  // Invert it
    return mask;
}

fn apply_mask(mask: u32, packed: u32, value: u32) -> u32 {
    value &= mask;
    value |= packed;
    return value;
}

// webgpu uses bankers round instead of normal round
// Regular rounds makes more sense and rust uses it
// Emulate normal round here
fn normal_round(value: f32) -> f32 {
    if fract(value) < 0.5 {
        return floor(value);
    }
    return ceil(value);
}

// Ugly stuff due to webgpu is annoying sometimes

fn get_value(locator: ValueLocator) -> u32 {
    // Switch because of this issue, where we cannot have array of buffers:
    // <https://github.com/gpuweb/gpuweb/issues/822>
    // It would be nice to get out buffer then read from value index, but noooo
    // Pointers not allowed as mutable variables
    var value: u32;
    var index: u32 = 0;
    if locator.total_index < arrayLength(prev_state);
    switch locator.buffer_index {
        // {AUTOGENERATE_BUFFER_READ}
        case 0u: {
            value = data[locator.value_index];
        }
        default: {
            // A panic or crash would be best, but not sure how to do it here
            value = 0u;
        }
    };
    return value;
}

// 8 bit
fn write_value(locator: ValueLocator, value: f32) {
    switch locator.buffer_index {
        // {AUTOGENERATE_BUFFER_WRITE}
        case 0u: {
            data[locator.value_index] = value;
        }
        default: {
            // A panic or crash would be best, but not sure how to do it here
            // noop
        }
    };
}

fn get_locator(global_id: vec3<u32>) -> ValueLocator {
    // the variable Z is not currently used, and u64 not implemented
    // this means the maximum amount of values split into the buffers is
    // 4_294_967_295
    // This can be solved by emulating u64, but for now, it is ok
    // Example: if you have arrays with structs with size of 128 bytes,
    // the maximum possible structs you could represent with the index
    // would be 549_755_813_760, which is about ~550Gb, which is sufficient
    // in this case.
    var total_index: u32 = global_id.x + global_id.y * MAX_DISPATCH;
    let locator = ValueLocator(total_index);
    return locator;
}


@compute
@workgroup_size(1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let locator = get_locator(global_id);
}
