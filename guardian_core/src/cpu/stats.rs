//! Notes: Each core can have a stat struct that sums up all types of things, such as number of connections and such
//! Should be atomic. Should not cost that much in performance

use std::sync::atomic::AtomicU32;

pub struct Stats {
    new_connections: AtomicU32,
    failed_connections: AtomicU32,
    reset_pending_index: AtomicU32,
}