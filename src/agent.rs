use crate::environment::Environment;
use crate::experience::Experience;
use crate::policy::Policy;

/// An agent that can act in an environment and learn from experience.
///
/// An `Agent` is a `Policy` that also knows how to update itself from
/// batches of experience. The update logic is algorithm-specific and
/// lives in ember-rl — this trait defines only the interface.
///
/// # Separation of concerns
///
/// - **`Policy`**: pure observation → action mapping. Stateless from the
///   caller's perspective (though the implementation may be stateful internally).
///
/// - **`Agent`**: owns a policy and can improve it. Has mutable state.
///
/// Training loops in ember-rl work entirely in terms of `Agent<E>`:
/// collect experience from `E`, push to a buffer, call `update()`.
///
/// # Generic over the environment
///
/// `Agent<E: Environment>` binds the agent to a specific environment's
/// observation and action types. This prevents accidentally feeding a
/// CartPole observation to a MuJoCo agent at compile time, not runtime.
pub trait Agent<E: Environment>: Policy<E::Observation, E::Action> {
    /// Update the agent's parameters from a batch of experience.
    ///
    /// The batch may be a random sample from a replay buffer (off-policy),
    /// or a full trajectory (on-policy). The agent decides how to use it.
    ///
    /// This is called once per training step in the outer loop.
    fn update(&mut self, batch: &[Experience<E::Observation, E::Action>]);
}
