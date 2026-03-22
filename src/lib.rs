//! Core traits for reinforcement learning environments, policies, and agents.
//!
//! `rl-traits` defines the shared vocabulary used across the ecosystem:
//!
//! - [`ember-rl`]: algorithm implementations using Burn for neural networks
//! - [`bevy-gym`]: Bevy ECS plugin for visualising and parallelising environments
//!
//! # Design goals
//!
//! - **Type-safe by default**: observation and action spaces are associated types,
//!   not runtime objects. The compiler catches mismatches.
//!
//! - **Correct `Terminated` vs `Truncated` distinction**: algorithms that bootstrap
//!   value estimates (PPO, DQN, SAC) need this distinction. It is baked into
//!   [`EpisodeStatus`] from day one.
//!
//! - **Rendering-free**: this crate has no concept of visualisation. That belongs
//!   in `bevy-gym`.
//!
//! - **Bevy-compatible**: `Send + Sync + 'static` bounds on associated types mean
//!   any [`Environment`] implementation can be a Bevy `Component`, enabling
//!   free ECS parallelisation via `Query::par_iter_mut()`.
//!
//! - **Minimal dependencies**: only `rand` for RNG abstractions.
//!
//! # Quick start
//!
//! ```rust
//! use rl_traits::{Environment, StepResult, EpisodeStatus};
//! use rand::Rng;
//!
//! struct MyEnv;
//!
//! impl Environment for MyEnv {
//!     type Observation = f32;
//!     type Action = usize;
//!     type Info = ();
//!
//!     fn step(&mut self, action: usize) -> StepResult<f32, ()> {
//!         StepResult::new(0.0, 1.0, EpisodeStatus::Continuing, ())
//!     }
//!
//!     fn reset(&mut self, _seed: Option<u64>) -> (f32, ()) {
//!         (0.0, ())
//!     }
//!
//!     fn sample_action(&self, rng: &mut impl Rng) -> usize {
//!         rng.gen_range(0..4)
//!     }
//! }
//! ```

pub mod agent;
pub mod buffer;
pub mod environment;
pub mod episode;
pub mod experience;
pub mod policy;
pub mod wrappers;

pub use agent::Agent;
pub use buffer::ReplayBuffer;
pub use environment::Environment;
pub use episode::{EpisodeStatus, StepResult};
pub use experience::Experience;
pub use policy::{Policy, StochasticPolicy};
pub use wrappers::{TimeLimit, Wrapper};
