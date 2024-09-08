@group(0)
@binding(0)
var<storage, read_write> data: array<atomic<u32>>;
// {AUTOGENERATE_BUFFER_BIND}

const MAX: u32 = 65535u;
// Change this depending on the limits per buffer
const MAX_BUFFER_SIZE: u32 = 134217728u;  // 128 MiB

// Division is expensive, multiplication is fast
// Precalculate division and multiply -> fast
const DIVIDE_BY_255: f32 = 1.0 / 255.0;

struct BufferValueLocator {
    buffer_index: u32,
    value_index: u32
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

fn unpack(packed: u32, shift: u32) -> f32 {
    var unpacked: f32 = f32(packed >> (8u * shift) & 255u) * DIVIDE_BY_255;
    return unpacked;
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

fn get_value(index: u32, shift: u32) -> f32 {
    let locator = get_locator(index);

    // Switch because of this issue, where we cannot have array of buffers:
    // <https://github.com/gpuweb/gpuweb/issues/822>
    // It would be nice to get out buffer then read from value index, but noooo
    // Pointers not allowed as mutable variables
    var value: u32;
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
    var unpacked: f32 = unpack(value, shift);
    return unpacked;
}

// 8 bit
fn write_value(index: u32, value: f32, shift: u32) {
    let locator = get_locator(index);

    // Prepare input
    let packed = pack(value, shift);
    var mask: u32 = 0xFFu << shift * 8u;
    mask = ~mask;  // Invert it

    // See get_value function
    switch locator.buffer_index {
        // {AUTOGENERATE_BUFFER_WRITE}
        case 0u: {
            atomicAnd(&data[locator.value_index], mask);
            atomicOr(&data[locator.value_index], packed);
        }
        default: {
            // A panic or crash would be best, but not sure how to do it here
            // noop
        }
    };
}

fn get_locator(index: u32) -> BufferValueLocator {
    let buffer_index = index % MAX_BUFFER_SIZE;
    let value_index = index - (buffer_index * MAX_BUFFER_SIZE);
    let locator = BufferValueLocator(buffer_index, value_index);
    return locator;
}


@compute
@workgroup_size(1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_index) local_invocation_index: u32
) {
    //data[global_id.x] += 0.1;
    var global_id_fixed: u32 = global_id.x / 4u;
    var shift: u32 = global_id.x % 4u;
    var value: f32 = unpack(data[global_id_fixed], shift);  // No need for atomics
    value += 0.1;  // Modify
    let packed = pack(value, shift);
    // A mask with the lower 8 bits set, shifted to overlay correctly
    // Invert, so we have a "hole"
    var mask: u32 = 0xFFu << shift * 8u;
    mask = ~mask;  // Invert it
    atomicAnd(&data[global_id_fixed], mask);
    atomicOr(&data[global_id_fixed], packed);
    atomicMax(&data[global_id_fixed], 0xFFFFFFFFu);

    //data[global_id_fixed] &= mask;
    //data[global_id_fixed] |= packed;
}
