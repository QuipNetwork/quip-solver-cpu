//! Kernels for structured problem families. Each takes a validated model and
//! the family's motif metadata.

pub mod dimer_weighted;
pub mod dimer_weighted_sampler;
pub mod flow;
pub mod flow_sampler;
pub mod ice_sat;
pub mod ice_sat_sampler;
pub mod otsuka;
pub mod otsuka_sampler;
pub mod routing;
pub mod routing_sampler;
