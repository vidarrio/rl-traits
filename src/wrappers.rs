use rand::Rng;

use crate::environment::Environment;
use crate::episode::{EpisodeStatus, StepResult};

/// A marker trait for environments that wrap another environment.
///
/// Wrappers modify environment behaviour without changing its interface —
/// exactly like Gymnasium's wrapper system, but type-safe.
///
/// # Examples of wrappers (to be implemented in user code or ember-rl)
///
/// - `TimeLimit<E>`: truncate episodes after N steps
/// - `NormalizeObs<E>`: normalize observations to zero mean, unit variance
/// - `ClipReward<E>`: clip rewards to a fixed range
/// - `FrameStack<E>`: stack the last N observations
///
/// # Note on associated types
///
/// A wrapper may change `Observation` or `Action` types (e.g. `FrameStack`
/// changes the observation shape). When types pass through unchanged,
/// use `type Observation = E::Observation` etc.
pub trait Wrapper: Environment {
    type Inner: Environment;

    fn inner(&self) -> &Self::Inner;
    fn inner_mut(&mut self) -> &mut Self::Inner;

    /// Unwrap all layers and return a reference to the base environment.
    fn unwrapped(&self) -> &Self::Inner {
        self.inner()
    }
}

/// Wraps an environment and truncates episodes after `max_steps` steps.
///
/// This is one of the most universally needed wrappers. Without it,
/// environments without natural termination conditions (e.g. locomotion
/// tasks) run forever.
///
/// Episodes truncated by this wrapper emit `EpisodeStatus::Truncated`,
/// not `EpisodeStatus::Terminated`, so algorithms correctly bootstrap
/// the value of the final state.
pub struct TimeLimit<E: Environment> {
    env: E,
    max_steps: usize,
    current_step: usize,
}

impl<E: Environment> TimeLimit<E> {
    pub fn new(env: E, max_steps: usize) -> Self {
        Self {
            env,
            max_steps,
            current_step: 0,
        }
    }

    pub fn elapsed_steps(&self) -> usize {
        self.current_step
    }

    pub fn remaining_steps(&self) -> usize {
        self.max_steps.saturating_sub(self.current_step)
    }
}

impl<E: Environment> Environment for TimeLimit<E> {
    type Observation = E::Observation;
    type Action = E::Action;
    type Info = E::Info;

    fn step(&mut self, action: Self::Action) -> StepResult<Self::Observation, Self::Info> {
        let mut result = self.env.step(action);
        self.current_step += 1;

        // Only truncate if the environment hasn't already terminated naturally.
        // We don't override Terminated with Truncated — natural termination wins.
        if self.current_step >= self.max_steps && result.status == EpisodeStatus::Continuing {
            result.status = EpisodeStatus::Truncated;
        }

        result
    }

    fn reset(&mut self, seed: Option<u64>) -> (Self::Observation, Self::Info) {
        self.current_step = 0;
        self.env.reset(seed)
    }

    fn sample_action(&self, rng: &mut impl Rng) -> Self::Action {
        self.env.sample_action(rng)
    }
}

impl<E: Environment> Wrapper for TimeLimit<E> {
    type Inner = E;
    fn inner(&self) -> &E {
        &self.env
    }
    fn inner_mut(&mut self) -> &mut E {
        &mut self.env
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::episode::{EpisodeStatus, StepResult};

    /// Minimal environment whose termination behaviour is controlled by a
    /// closure, so each test can specify exactly when it terminates.
    struct MockEnv<F: FnMut(usize) -> EpisodeStatus> {
        step_count: usize,
        status_fn: F,
    }

    impl<F: FnMut(usize) -> EpisodeStatus> MockEnv<F> {
        fn new(status_fn: F) -> Self {
            Self { step_count: 0, status_fn }
        }
    }

    impl<F: FnMut(usize) -> EpisodeStatus + Send + Sync + 'static> Environment for MockEnv<F> {
        type Observation = ();
        type Action = ();
        type Info = ();

        fn step(&mut self, _action: ()) -> StepResult<(), ()> {
            self.step_count += 1;
            let status = (self.status_fn)(self.step_count);
            StepResult::new((), 1.0, status, ())
        }

        fn reset(&mut self, _seed: Option<u64>) -> ((), ()) {
            self.step_count = 0;
            ((), ())
        }

        fn sample_action(&self, _rng: &mut impl rand::Rng) -> () {}
    }

    // ── TimeLimit ────────────────────────────────────────────────────────────

    #[test]
    fn steps_below_limit_pass_through_status_unchanged() {
        let mut env = TimeLimit::new(
            MockEnv::new(|_| EpisodeStatus::Continuing),
            5,
        );
        env.reset(None);
        for _ in 0..4 {
            let r = env.step(());
            assert_eq!(r.status, EpisodeStatus::Continuing);
        }
    }

    #[test]
    fn truncates_at_limit_when_inner_is_continuing() {
        let mut env = TimeLimit::new(
            MockEnv::new(|_| EpisodeStatus::Continuing),
            3,
        );
        env.reset(None);
        env.step(());
        env.step(());
        let r = env.step(());
        assert_eq!(r.status, EpisodeStatus::Truncated);
    }

    #[test]
    fn natural_termination_wins_over_time_limit_on_same_step() {
        // If the environment terminates naturally on the exact step that the
        // time limit would fire, the result must be Terminated, not Truncated.
        // Confusing these would cause an algorithm to incorrectly bootstrap the
        // terminal state's value.
        let mut env = TimeLimit::new(
            MockEnv::new(|n| {
                if n >= 3 { EpisodeStatus::Terminated } else { EpisodeStatus::Continuing }
            }),
            3,
        );
        env.reset(None);
        env.step(());
        env.step(());
        let r = env.step(());
        assert_eq!(r.status, EpisodeStatus::Terminated);
    }

    #[test]
    fn reset_restores_step_counter_so_limit_applies_again() {
        let mut env = TimeLimit::new(
            MockEnv::new(|_| EpisodeStatus::Continuing),
            2,
        );
        env.reset(None);
        env.step(());
        let r = env.step(());
        assert_eq!(r.status, EpisodeStatus::Truncated);

        env.reset(None);
        let r = env.step(());
        assert_eq!(r.status, EpisodeStatus::Continuing, "step counter not reset");
    }
}
