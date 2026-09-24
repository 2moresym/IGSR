//! Vireo backend — feature-gated stub for later. The `GpuBackend` trait
//! keeps the seam stable so GL code paths move over without a rewrite.

use super::{ComputePath, GpuBackend};

pub struct VireoBackend;

impl GpuBackend for VireoBackend {
    fn name(&self) -> &'static str {
        "Vireo (stub)"
    }

    fn compute_path(&self) -> ComputePath {
        ComputePath::Compute
    }
}
