//! Inpainting modules for grid line removal

pub mod directional;
pub mod fast_march;

pub use directional::directional_interpolate;
pub use fast_march::fast_marching_inpaint;
