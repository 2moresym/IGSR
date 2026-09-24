//! `GpuBackend` trait: the seam that lets the same C/Rust core run on a
//! throwaway GL backend today and Vireo later without a rewrite.

pub mod gl;
#[cfg(feature = "vireo")]
pub mod vireo;

/// Which shader path a pass should use. HD 4000 / Mesa crocus may not
/// expose compute, so every compute pass needs a fragment fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputePath {
    Compute,
    FragmentFallback,
}

/// Minimal backend interface (stage 1). Grows in stages 3–4 with texture
/// upload / dispatch / timer-query hooks.
pub trait GpuBackend {
    fn name(&self) -> &'static str;
    fn compute_path(&self) -> ComputePath;
    fn supports_compute(&self) -> bool {
        self.compute_path() == ComputePath::Compute
    }
}
