use rand::Rng;

/// A deterministic mapping from observations to actions.
///
/// This is the simplest policy interface. Sufficient for value-based methods
/// (DQN with greedy action selection) and imitation learning. Policy gradient
/// methods need `StochasticPolicy` instead.
pub trait Policy<O, A> {
    /// Select an action given the current observation.
    fn act(&self, observation: &O) -> A;
}

/// A stochastic policy that can also return the probability of its choices.
///
/// Required for policy gradient methods (PPO, A2C, SAC). The log probability
/// is used to compute the policy gradient and, in on-policy methods, to detect
/// when the policy has drifted too far from the behaviour policy.
///
/// Implementors must also implement `Policy<O, A>`. The deterministic `act()`
/// should typically sample from the distribution and discard the log prob.
///
/// # Relationship to entropy regularisation
///
/// Many modern RL algorithms (SAC, PPO with entropy bonus) add an entropy
/// term to the objective to encourage exploration. The `entropy()` method
/// provides this directly, enabling clean separation of the exploration
/// coefficient from the policy implementation.
pub trait StochasticPolicy<O, A>: Policy<O, A> {
    /// Sample an action and return its log probability under the current policy.
    ///
    /// The log probability `log π(a|s)` is required for:
    /// - Policy gradient computation: `∇ log π(a|s) * advantage`
    /// - PPO clipping ratio: `π_new(a|s) / π_old(a|s)`
    /// - SAC entropy bonus
    fn act_with_log_prob(&self, observation: &O, rng: &mut impl Rng) -> (A, f64);

    /// Evaluate the log probability of a specific (observation, action) pair.
    ///
    /// Used in off-policy corrections and importance sampling, where actions
    /// were collected under a previous version of the policy.
    fn log_prob(&self, observation: &O, action: &A) -> f64;

    /// Compute the entropy of the action distribution at this observation.
    ///
    /// `H[π(·|s)] = -∑ π(a|s) log π(a|s)` for discrete actions,
    /// or the differential entropy for continuous distributions.
    ///
    /// Used as a regularisation term: maximising entropy encourages
    /// exploration and prevents premature convergence to deterministic policies.
    fn entropy(&self, observation: &O) -> f64;
}
