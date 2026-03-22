use rand::Rng;

use crate::episode::StepResult;

/// The core environment trait.
///
/// Defines the contract that all RL environments must satisfy, regardless
/// of whether they run headless in ember-rl or are visualised via bevy-gym.
///
/// # Design principles
///
/// - **Type-safe observation and action spaces**: `Observation` and `Action`
///   are associated types. The compiler enforces correctness; there are no
///   runtime Box/Discrete/Dict space objects.
///
/// - **Typed `Info`**: auxiliary data is `Self::Info`, not `dict[str, Any]`.
///   If you don't need it, use `()` and get `Default` for free.
///
/// - **No `render()`**: visualisation is entirely bevy-gym's concern.
///   rl-traits knows nothing about rendering.
///
/// - **No `close()`**: implement `Drop` if your environment holds resources.
///
/// - **Bevy-compatible by design**: `Send + Sync + 'static` bounds on
///   associated types mean implementations can be used as Bevy `Component`s
///   directly, enabling free ECS-based parallelisation in bevy-gym via
///   `Query::par_iter_mut()`.
///
/// # Example
///
/// ```rust
/// use rl_traits::{Environment, StepResult, EpisodeStatus};
/// use rand::Rng;
///
/// struct BanditsEnv {
///     arms: [f64; 4],
///     rng: rand::rngs::SmallRng,
/// }
///
/// impl Environment for BanditsEnv {
///     type Observation = ();      // stateless — observation is always ()
///     type Action = usize;        // pull arm 0..3
///     type Info = ();
///
///     fn step(&mut self, action: usize) -> StepResult<(), ()> {
///         let reward = self.rng.gen::<f64>() * self.arms[action];
///         StepResult::new((), reward, EpisodeStatus::Continuing, ())
///     }
///
///     fn reset(&mut self, _seed: Option<u64>) -> ((), ()) {
///         ((), ())
///     }
///
///     fn sample_action(&self, rng: &mut impl Rng) -> usize {
///         rng.gen_range(0..4)
///     }
/// }
/// ```
pub trait Environment {
    /// The observation type produced by `step()` and `reset()`.
    ///
    /// `Send + Sync + 'static` are required for Bevy ECS compatibility.
    type Observation: Clone + Send + Sync + 'static;

    /// The action type consumed by `step()`.
    type Action: Clone + Send + Sync + 'static;

    /// Auxiliary information returned alongside observations.
    ///
    /// Use `()` if you don't need it — `Default` is implemented for `()`.
    type Info: Default + Clone + Send + Sync + 'static;

    /// Advance the environment by one timestep.
    ///
    /// The caller is responsible for checking `StepResult::is_done()` and
    /// calling `reset()` before the next episode.
    fn step(&mut self, action: Self::Action) -> StepResult<Self::Observation, Self::Info>;

    /// Reset the environment to an initial state, starting a new episode.
    ///
    /// If `seed` is `Some(u64)`, the environment should use it to seed its
    /// internal RNG for deterministic reproduction of episodes.
    fn reset(&mut self, seed: Option<u64>) -> (Self::Observation, Self::Info);

    /// Sample a random action from this environment's action space.
    ///
    /// Used by random exploration agents and for initial data collection.
    /// The `rng` is caller-supplied so exploration randomness can be seeded
    /// and tracked independently from environment randomness.
    fn sample_action(&self, rng: &mut impl Rng) -> Self::Action;
}
