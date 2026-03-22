use std::collections::HashMap;
use std::hash::Hash;

use rand::Rng;

use crate::episode::{EpisodeStatus, StepResult};

/// A multi-agent environment where all agents act simultaneously each step.
///
/// Mirrors the semantics of PettingZoo's Parallel API, adapted for Rust's
/// type system. Use this when all agents observe and act at every step —
/// cooperative navigation, competitive games, mixed-team tasks.
///
/// # Design principles
///
/// - **`possible_agents` vs `agents`**: `possible_agents()` is the fixed
///   universe of all agent IDs. `agents()` is the live subset for the current
///   episode. After `reset()`, `agents == possible_agents`. Agents are removed
///   from `agents` when their episode ends; the episode is over when `agents`
///   is empty.
///
/// - **Joint step**: `step()` takes exactly one action per agent in `agents()`
///   and returns one [`StepResult`] per active agent. Providing actions for
///   terminated agents or omitting active agents is undefined behaviour.
///
/// - **Homogeneous agents**: all agents share `Observation`, `Action`, and
///   `Info` types. Heterogeneous agents can be modelled with enum wrappers
///   over the per-type variants.
///
/// - **Bevy-compatible by design**: `AgentId: Eq + Hash + Send + Sync +
///   'static` means Bevy `Entity` is a valid agent ID directly, enabling
///   free ECS-based parallelisation across agents in bevy-gym.
///
/// - **No `render()`**: visualisation is bevy-gym's concern.
///
/// - **No `close()`**: implement `Drop` if your environment holds resources.
///
/// # Example
///
/// ```rust
/// use std::collections::HashMap;
/// use rl_traits::{ParallelEnvironment, StepResult, EpisodeStatus};
/// use rand::Rng;
///
/// struct CoopGame {
///     active: Vec<usize>,
/// }
///
/// impl ParallelEnvironment for CoopGame {
///     type AgentId = usize;
///     type Observation = f32;
///     type Action = bool;   // cooperate or defect
///     type Info = ();
///
///     fn possible_agents(&self) -> &[usize] { &[0, 1] }
///     fn agents(&self) -> &[usize] { &self.active }
///
///     fn step(&mut self, _actions: HashMap<usize, bool>)
///         -> HashMap<usize, StepResult<f32, ()>>
///     {
///         self.active.iter()
///             .map(|&id| (id, StepResult::new(0.0_f32, 1.0, EpisodeStatus::Continuing, ())))
///             .collect()
///     }
///
///     fn reset(&mut self, _seed: Option<u64>) -> HashMap<usize, (f32, ())> {
///         self.active = vec![0, 1];
///         self.active.iter().map(|&id| (id, (0.0_f32, ()))).collect()
///     }
///
///     fn sample_action(&self, _agent: &usize, rng: &mut impl Rng) -> bool {
///         rng.gen()
///     }
/// }
/// ```
pub trait ParallelEnvironment {
    /// Identifier for each agent.
    ///
    /// Common choices: `usize` (index), `&'static str` (name), or a Bevy
    /// `Entity` for direct ECS integration without an extra lookup.
    type AgentId: Eq + Hash + Clone + Send + Sync + 'static;

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

    /// The complete, fixed set of agent IDs for this environment.
    ///
    /// Does not change between episodes or as agents terminate mid-episode.
    /// Use `agents()` for the currently live set.
    fn possible_agents(&self) -> &[Self::AgentId];

    /// The agents currently active in this episode.
    ///
    /// Starts equal to `possible_agents()` after `reset()`. Shrinks as agents
    /// terminate or are truncated; never grows. Empty when the episode is over.
    fn agents(&self) -> &[Self::AgentId];

    /// Advance the environment by one step using joint actions.
    ///
    /// `actions` must contain exactly one entry per agent in `self.agents()`.
    /// After this call, agents whose result was done are removed from `agents()`.
    fn step(
        &mut self,
        actions: HashMap<Self::AgentId, Self::Action>,
    ) -> HashMap<Self::AgentId, StepResult<Self::Observation, Self::Info>>;

    /// Reset the environment to an initial state, starting a new episode.
    ///
    /// If `seed` is `Some(u64)`, the environment should use it to seed its
    /// internal RNG for deterministic reproduction of episodes.
    /// Returns the initial observation and info for every agent.
    fn reset(
        &mut self,
        seed: Option<u64>,
    ) -> HashMap<Self::AgentId, (Self::Observation, Self::Info)>;

    /// Sample a random action for the given agent.
    ///
    /// The `rng` is caller-supplied so exploration randomness can be seeded
    /// and tracked independently from environment randomness.
    fn sample_action(&self, agent: &Self::AgentId, rng: &mut impl Rng) -> Self::Action;

    /// A global state observation of the full environment.
    ///
    /// Used by centralised-training / decentralised-execution algorithms
    /// (e.g. MADDPG, QMIX) that condition a centralised critic on the full
    /// state while individual policies see only local observations.
    /// Returns `None` by default; override if your environment supports it.
    fn state(&self) -> Option<Self::Observation> {
        None
    }

    /// Returns `true` when all agents have finished (active set is empty).
    fn is_done(&self) -> bool {
        self.agents().is_empty()
    }

    /// Number of currently active agents.
    fn num_agents(&self) -> usize {
        self.agents().len()
    }

    /// Maximum number of agents that could ever be active simultaneously.
    fn max_num_agents(&self) -> usize {
        self.possible_agents().len()
    }
}

/// A multi-agent environment with Agent Environment Cycle (turn-based) semantics.
///
/// Mirrors the semantics of PettingZoo's AEC API, adapted for Rust's type
/// system. Use this when agents act one at a time — board games, card games,
/// or any domain where simultaneous action is not meaningful.
///
/// # Design principles
///
/// - **Turn-based execution**: one agent acts per `step()` call.
///   `agent_selection()` identifies whose turn it is. After each call,
///   the selection advances to the next active agent.
///
/// - **Persistent state**: the environment tracks each agent's most recent
///   reward, status, and info as mutable state. Read it via `agent_state()`
///   before deciding on an action. `last()` is a convenience that combines
///   `observe()` and `agent_state()` for the current agent.
///
/// - **Cycling out terminated agents**: when `agent_state()` reports a done
///   status for `agent_selection()`, pass `None` to `step()` to advance the
///   turn without applying an action. The type signature makes this contract
///   explicit — passing `Some(action)` for a done agent is undefined behaviour.
///
/// - **Bevy-compatible by design**: same `Send + Sync + 'static` bounds as
///   [`ParallelEnvironment`]. The turn-based nature is inherently sequential,
///   so ECS parallelisation applies less directly than with `ParallelEnvironment`.
///
/// - **No `render()`**: visualisation is bevy-gym's concern.
///
/// - **No `close()`**: implement `Drop` if your environment holds resources.
///
/// # Example
///
/// Typical AEC loop:
///
/// ```rust,ignore
/// env.reset(None);
/// while !env.is_done() {
///     let (obs, _reward, status, _info) = env.last();
///     let action = if status.is_done() {
///         None  // cycle the terminated agent out
///     } else {
///         Some(policy.act(env.agent_selection(), &obs.unwrap()))
///     };
///     env.step(action);
/// }
/// ```
pub trait AecEnvironment {
    /// Identifier for each agent. Same semantics as [`ParallelEnvironment::AgentId`].
    type AgentId: Eq + Hash + Clone + Send + Sync + 'static;

    /// The observation type produced by `observe()`.
    ///
    /// `Send + Sync + 'static` are required for Bevy ECS compatibility.
    type Observation: Clone + Send + Sync + 'static;

    /// The action type consumed by `step()`.
    type Action: Clone + Send + Sync + 'static;

    /// Auxiliary information returned alongside observations.
    ///
    /// Use `()` if you don't need it — `Default` is implemented for `()`.
    type Info: Default + Clone + Send + Sync + 'static;

    /// The complete, fixed set of agent IDs for this environment.
    ///
    /// Does not change between episodes or as agents terminate mid-episode.
    fn possible_agents(&self) -> &[Self::AgentId];

    /// The agents currently active in this episode.
    ///
    /// Starts equal to `possible_agents()` after `reset()`. Shrinks as agents
    /// terminate or are truncated; never grows. Empty when the episode is over.
    fn agents(&self) -> &[Self::AgentId];

    /// The agent whose turn it currently is to act.
    fn agent_selection(&self) -> &Self::AgentId;

    /// Execute the current agent's action and advance to the next agent.
    ///
    /// Pass `None` when `agent_state(agent_selection())` reports a done status,
    /// to cycle the agent out without applying an action. Pass `Some(action)`
    /// otherwise.
    fn step(&mut self, action: Option<Self::Action>);

    /// Reset the environment to an initial state, starting a new episode.
    ///
    /// Unlike [`ParallelEnvironment::reset`], this returns nothing. Retrieve
    /// initial observations via `observe()` after calling `reset()`.
    /// If `seed` is `Some(u64)`, it is used to seed the internal RNG.
    fn reset(&mut self, seed: Option<u64>);

    /// Retrieve the current observation for the given agent.
    ///
    /// Returns `None` if the agent has terminated or been truncated — their
    /// last observation is no longer valid.
    fn observe(&self, agent: &Self::AgentId) -> Option<Self::Observation>;

    /// Retrieve the persistent `(reward, status, info)` for the given agent.
    ///
    /// This state is updated each time the agent acts and persists until its
    /// next turn. It reflects what the agent received as a result of its last
    /// action.
    fn agent_state(&self, agent: &Self::AgentId) -> (f64, EpisodeStatus, Self::Info);

    /// Sample a random action for the given agent.
    ///
    /// The `rng` is caller-supplied so exploration randomness can be seeded
    /// and tracked independently from environment randomness.
    fn sample_action(&self, agent: &Self::AgentId, rng: &mut impl Rng) -> Self::Action;

    /// Returns the full state for the currently selected agent.
    ///
    /// Convenience wrapper around `observe(agent_selection())` and
    /// `agent_state(agent_selection())`. This is the idiomatic way to read the
    /// current agent's situation at the top of the AEC loop.
    fn last(&self) -> (Option<Self::Observation>, f64, EpisodeStatus, Self::Info) {
        let agent = self.agent_selection().clone();
        let obs = self.observe(&agent);
        let (reward, status, info) = self.agent_state(&agent);
        (obs, reward, status, info)
    }

    /// Returns `true` when all agents have finished (active set is empty).
    fn is_done(&self) -> bool {
        self.agents().is_empty()
    }

    /// Number of currently active agents.
    fn num_agents(&self) -> usize {
        self.agents().len()
    }

    /// Maximum number of agents that could ever be active simultaneously.
    fn max_num_agents(&self) -> usize {
        self.possible_agents().len()
    }
}
