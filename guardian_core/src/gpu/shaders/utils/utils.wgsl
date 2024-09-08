struct IndexUnpacked {
    index: u32,
    shift: u32
}

struct DataLocator {
    buffer_index: u32,
    item_index: u32
};

// ** Compress and decompress states ** //

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
    var masked: u32 = value;
    masked &= mask;
    masked |= packed;
    return masked;
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

// Activation function similar what packing, but without compression
fn limited_relu(value: f32) -> f32 {
    return clamp(value, 0.0, 1.0);
}

// Function to get out values from compression
fn adjust_index(index: u32) -> IndexUnpacked {
    let value_index = index >> 2u;  // Same as i / 4
    let shift = index & 3u;  // Same as i % 4
    return IndexUnpacked(value_index, shift);
}


// ** Handles buffers ** //

fn get_locator(index: u32, array_size: u32) -> DataLocator {
    let buffer_index = index / array_size;
    let item_index = index % array_size;
    let locator = DataLocator(buffer_index, item_index);
    return locator;
}

