/// Whether an episode is ongoing, has naturally ended, or was cut short.
///
/// This distinction is critical for bootstrapping in RL algorithms.
///
/// # Why this matters
///
/// When computing value targets (e.g. TD targets, GAE), the treatment of
/// the terminal state depends on *why* the episode ended:
///
/// - `Terminated`: the agent reached a natural terminal state. The value of
///   the next state is zero — there is no future reward to bootstrap.
///
/// - `Truncated`: the episode was cut short by something external (e.g. a
///   time limit, the agent going out of bounds). The environment has not
///   actually terminated — the agent simply stopped. The value of the next
///   state is *non-zero* and must be bootstrapped from the value function.
///
/// Confusing these two is one of the most common bugs in policy gradient
/// implementations. Gymnasium introduced this distinction in v0.26; we
/// encode it correctly from the start.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EpisodeStatus {
    /// The episode is ongoing.
    Continuing,

    /// The episode reached a natural terminal state (MDP termination).
    ///
    /// Bootstrap target: `r + gamma * 0` — no future value.
    Terminated,

    /// The episode was cut short by an external condition (e.g. time limit).
    ///
    /// Bootstrap target: `r + gamma * V(s')` — future value is non-zero.
    Truncated,
}

impl EpisodeStatus {
    /// Returns `true` if the episode is over for any reason.
    #[inline]
    pub fn is_done(&self) -> bool {
        matches!(self, Self::Terminated | Self::Truncated)
    }

    /// Returns `true` only for natural MDP termination.
    /// Use this to decide whether to bootstrap the next-state value.
    #[inline]
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Terminated)
    }

    /// Returns `true` if the episode was cut short externally.
    #[inline]
    pub fn is_truncated(&self) -> bool {
        matches!(self, Self::Truncated)
    }
}

/// The output of a single environment step.
///
/// Returned by [`crate::Environment::step`]. Contains everything an agent needs
/// to learn: the next observation, the reward signal, whether the episode
/// is done, and any auxiliary info.
#[derive(Debug, Clone)]
pub struct StepResult<O, I> {
    /// The observation after taking the action.
    pub observation: O,

    /// The scalar reward signal.
    pub reward: f64,

    /// Whether the episode continues, terminated, or was truncated.
    pub status: EpisodeStatus,

    /// Auxiliary information (e.g. diagnostics, hidden state, sub-rewards).
    /// Typed — no `HashMap<String, Any>` here.
    pub info: I,
}

impl<O, I> StepResult<O, I> {
    pub fn new(observation: O, reward: f64, status: EpisodeStatus, info: I) -> Self {
        Self {
            observation,
            reward,
            status,
            info,
        }
    }

    /// Convenience: is the episode over for any reason?
    #[inline]
    pub fn is_done(&self) -> bool {
        self.status.is_done()
    }

    /// Map the observation to a different type (useful for wrapper implementations).
    pub fn map_obs<O2>(self, f: impl FnOnce(O) -> O2) -> StepResult<O2, I> {
        StepResult {
            observation: f(self.observation),
            reward: self.reward,
            status: self.status,
            info: self.info,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── EpisodeStatus ────────────────────────────────────────────────────────

    #[test]
    fn continuing_is_not_done() {
        assert!(!EpisodeStatus::Continuing.is_done());
        assert!(!EpisodeStatus::Continuing.is_terminal());
        assert!(!EpisodeStatus::Continuing.is_truncated());
    }

    #[test]
    fn terminated_is_done_and_terminal() {
        assert!(EpisodeStatus::Terminated.is_done());
        assert!(EpisodeStatus::Terminated.is_terminal());
        assert!(!EpisodeStatus::Terminated.is_truncated());
    }

    #[test]
    fn truncated_is_done_but_not_terminal() {
        // This is the critical distinction: Truncated ends the episode but the
        // next-state value is non-zero, so algorithms must NOT zero the bootstrap.
        assert!(EpisodeStatus::Truncated.is_done());
        assert!(!EpisodeStatus::Truncated.is_terminal());
        assert!(EpisodeStatus::Truncated.is_truncated());
    }

    // ── StepResult ───────────────────────────────────────────────────────────

    #[test]
    fn map_obs_transforms_observation_preserves_rest() {
        let result = StepResult::new(2_i32, 1.5, EpisodeStatus::Continuing, "info");
        let mapped = result.map_obs(|o| o * 10);
        assert_eq!(mapped.observation, 20);
        assert_eq!(mapped.reward, 1.5);
        assert_eq!(mapped.status, EpisodeStatus::Continuing);
        assert_eq!(mapped.info, "info");
    }
}
