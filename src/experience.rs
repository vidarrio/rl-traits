use crate::episode::EpisodeStatus;

/// A single transition: `(s, a, r, s', status)`.
///
/// The fundamental unit of experience stored in replay buffers and used
/// for agent updates. Corresponds to one (s, a, r, s', done) tuple in
/// classical RL literature, but with a richer `status` field that
/// distinguishes natural termination from truncation.
#[derive(Debug, Clone)]
pub struct Experience<O, A> {
    /// The observation at the start of this transition.
    pub observation: O,

    /// The action taken.
    pub action: A,

    /// The scalar reward received.
    pub reward: f64,

    /// The observation after taking the action.
    pub next_observation: O,

    /// Whether the episode ended and why.
    ///
    /// Algorithms that bootstrap value estimates (DQN, PPO, SAC) must
    /// inspect this to handle terminal states correctly:
    /// - `Terminated`: bootstrap with zero value
    /// - `Truncated`: bootstrap with V(next_observation)
    /// - `Continuing`: bootstrap with V(next_observation)
    pub status: EpisodeStatus,
}

impl<O, A> Experience<O, A> {
    pub fn new(
        observation: O,
        action: A,
        reward: f64,
        next_observation: O,
        status: EpisodeStatus,
    ) -> Self {
        Self {
            observation,
            action,
            reward,
            next_observation,
            status,
        }
    }

    /// Returns `true` if this transition ends an episode.
    #[inline]
    pub fn is_done(&self) -> bool {
        self.status.is_done()
    }

    /// Returns the bootstrap mask: `1.0` if the episode continues or was
    /// truncated (i.e. the next state has non-zero value), `0.0` if terminated.
    ///
    /// Multiply value estimates by this mask when computing TD targets:
    /// `target = reward + gamma * bootstrap_mask() * V(next_obs)`
    #[inline]
    pub fn bootstrap_mask(&self) -> f64 {
        match self.status {
            EpisodeStatus::Terminated => 0.0,
            EpisodeStatus::Continuing | EpisodeStatus::Truncated => 1.0,
        }
    }

    /// Map the observation to a different type.
    ///
    /// Useful for observation-wrapping layers that preprocess before storage.
    pub fn map_obs<O2>(self, f: impl Fn(O) -> O2) -> Experience<O2, A> {
        Experience {
            observation: f(self.observation),
            action: self.action,
            reward: self.reward,
            next_observation: f(self.next_observation),
            status: self.status,
        }
    }

    /// Map the action to a different type.
    pub fn map_action<A2>(self, f: impl Fn(A) -> A2) -> Experience<O, A2> {
        Experience {
            observation: self.observation,
            action: f(self.action),
            reward: self.reward,
            next_observation: self.next_observation,
            status: self.status,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exp(status: EpisodeStatus) -> Experience<i32, i32> {
        Experience::new(0, 0, 1.0, 1, status)
    }

    // ── bootstrap_mask ───────────────────────────────────────────────────────

    #[test]
    fn bootstrap_mask_is_zero_on_termination() {
        // Terminated: natural end of MDP — next-state value is zero.
        assert_eq!(exp(EpisodeStatus::Terminated).bootstrap_mask(), 0.0);
    }

    #[test]
    fn bootstrap_mask_is_one_when_continuing() {
        assert_eq!(exp(EpisodeStatus::Continuing).bootstrap_mask(), 1.0);
    }

    #[test]
    fn bootstrap_mask_is_one_when_truncated() {
        // Truncated: episode cut short externally — next state still has value.
        // Zeroing this would underestimate returns; algorithms must bootstrap.
        assert_eq!(exp(EpisodeStatus::Truncated).bootstrap_mask(), 1.0);
    }

    // ── map_obs / map_action ─────────────────────────────────────────────────

    #[test]
    fn map_obs_transforms_both_observations() {
        let e = Experience::new(1_i32, 99_i32, 2.0, 3_i32, EpisodeStatus::Continuing);
        let mapped = e.map_obs(|o| o * 10);
        assert_eq!(mapped.observation, 10);
        assert_eq!(mapped.next_observation, 30);
        assert_eq!(mapped.action, 99);
        assert_eq!(mapped.reward, 2.0);
    }

    #[test]
    fn map_action_transforms_action_preserves_observations() {
        let e = Experience::new(5_i32, 2_i32, 0.5, 6_i32, EpisodeStatus::Truncated);
        let mapped = e.map_action(|a| a as f64 * 0.5);
        assert_eq!(mapped.action, 1.0_f64);
        assert_eq!(mapped.observation, 5);
        assert_eq!(mapped.next_observation, 6);
        assert_eq!(mapped.status, EpisodeStatus::Truncated);
    }
}
