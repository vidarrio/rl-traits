use std::collections::HashMap;
use std::hash::Hash;

use rand::Rng;

use crate::episode::{EpisodeStatus, StepResult};

// ── Parallel API ─────────────────────────────────────────────────────────────

/// A multi-agent environment where all agents act simultaneously each step.
///
/// Mirrors the semantics of PettingZoo's [Parallel API], adapted for Rust's
/// type system and Bevy ECS compatibility.
///
/// [Parallel API]: https://pettingzoo.farama.org/api/parallel/
///
/// # Homogeneous agents
///
/// All agents share the same `Observation`, `Action`, and `Info` types. This
/// covers the majority of MARL scenarios (cooperative navigation, competitive
/// games, mixed-team tasks). Heterogeneous agents — where different agent types
/// have structurally different spaces — can be modelled by making `Observation`
/// and `Action` enums that wrap the per-type variants.
///
/// # Bevy compatibility
///
/// Like [`crate::Environment`], all associated types carry `Send + Sync +
/// 'static` bounds. `AgentId: Eq + Hash` enables `HashMap`-keyed per-agent
/// data. Bevy `Entity` satisfies all `AgentId` bounds directly.
///
/// # `possible_agents` vs `agents`
///
/// `possible_agents()` is the fixed universe of all agent IDs the environment
/// could ever use. `agents()` is the live subset currently in the episode.
/// After `reset()`, `agents == possible_agents`. As agents terminate or are
/// truncated, they are removed from `agents`. The episode ends when `agents`
/// is empty.
///
/// # Step contract
///
/// `step()` must receive exactly one action per agent currently in `agents()`.
/// Providing actions for terminated agents or omitting active agents is
/// undefined behaviour. The returned map contains one [`StepResult`] per
/// active agent; agents whose result is done are removed from `agents` before
/// the next call.
///
/// # Example loop
///
/// ```rust,ignore
/// env.reset(None);
/// while !env.is_done() {
///     let actions = env.agents().iter()
///         .map(|id| (id.clone(), policy.act(id, &obs[id])))
///         .collect();
///     let results = env.step(actions);
///     // inspect results, store experience, update obs…
/// }
/// ```
pub trait ParallelEnvironment {
    /// Identifier for each agent.
    ///
    /// Common choices: `usize` (index), `&'static str` (name), or a Bevy
    /// `Entity` for direct ECS integration without an extra lookup.
    type AgentId: Eq + Hash + Clone + Send + Sync + 'static;

    /// Observation type, shared across all agents.
    type Observation: Clone + Send + Sync + 'static;

    /// Action type, shared across all agents.
    type Action: Clone + Send + Sync + 'static;

    /// Auxiliary info type, shared across all agents.
    type Info: Default + Clone + Send + Sync + 'static;

    /// The complete, fixed set of agent IDs for this environment.
    ///
    /// Does not change between episodes or as agents terminate mid-episode.
    fn possible_agents(&self) -> &[Self::AgentId];

    /// The agents currently active in this episode.
    ///
    /// Starts equal to `possible_agents()` after `reset()`. Shrinks (never
    /// grows) as agents terminate or are truncated. Empty when the episode
    /// is over.
    fn agents(&self) -> &[Self::AgentId];

    /// Advance the environment by one step using joint actions.
    ///
    /// `actions` must contain exactly one entry per agent in `self.agents()`.
    /// Returns one [`StepResult`] per active agent. After this call,
    /// `self.agents()` no longer includes agents whose result was done.
    fn step(
        &mut self,
        actions: HashMap<Self::AgentId, Self::Action>,
    ) -> HashMap<Self::AgentId, StepResult<Self::Observation, Self::Info>>;

    /// Reset the environment to an initial state.
    ///
    /// Restores the full `possible_agents()` set as active and returns the
    /// initial observation and info for each agent.
    fn reset(
        &mut self,
        seed: Option<u64>,
    ) -> HashMap<Self::AgentId, (Self::Observation, Self::Info)>;

    /// Sample a random action for the given agent.
    ///
    /// The caller supplies `rng` so exploration randomness is seeded and
    /// tracked independently from environment randomness.
    fn sample_action(&self, agent: &Self::AgentId, rng: &mut impl Rng) -> Self::Action;

    /// A global state observation for the full environment.
    ///
    /// Used by centralised-training / decentralised-execution algorithms
    /// (e.g. MADDPG, QMIX) that condition a centralised critic on the full
    /// environment state while individual policies see only local observations.
    ///
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

// ── AEC API ──────────────────────────────────────────────────────────────────

/// A multi-agent environment with Agent Environment Cycle (turn-based) semantics.
///
/// Mirrors the semantics of PettingZoo's [AEC API], adapted for Rust's type
/// system and Bevy ECS compatibility.
///
/// [AEC API]: https://pettingzoo.farama.org/api/aec/
///
/// # Execution model
///
/// Agents act one at a time. Each call to `step()` advances the
/// `agent_selection` to the next agent. The environment internally maintains
/// the most recent reward, status, and info for each agent as persistent state,
/// readable at any time via `agent_state()`.
///
/// # AEC loop pattern
///
/// ```rust,ignore
/// env.reset(None);
/// while !env.is_done() {
///     let (obs, reward, status, info) = env.last();
///     let action = if status.is_done() {
///         None  // cycle terminated agent out without acting
///     } else {
///         Some(policy.act(env.agent_selection(), &obs.unwrap()))
///     };
///     env.step(action);
/// }
/// ```
///
/// # Terminated agent handling
///
/// When `agent_state()` reports a done status for `agent_selection()`, pass
/// `None` to `step()`. This cycles the agent out and advances the turn without
/// applying an action. Passing `Some(action)` for a done agent is undefined
/// behaviour.
///
/// # Bevy compatibility
///
/// Same `Send + Sync + 'static` bounds as [`ParallelEnvironment`]. The
/// turn-based nature means this API is inherently sequential and does not
/// benefit from ECS parallelism as directly as `ParallelEnvironment`, but
/// it can still be used as a Bevy `Component`.
pub trait AecEnvironment {
    /// Identifier for each agent. Same semantics as [`ParallelEnvironment::AgentId`].
    type AgentId: Eq + Hash + Clone + Send + Sync + 'static;

    /// Observation type, shared across all agents.
    type Observation: Clone + Send + Sync + 'static;

    /// Action type, shared across all agents.
    type Action: Clone + Send + Sync + 'static;

    /// Auxiliary info type, shared across all agents.
    type Info: Default + Clone + Send + Sync + 'static;

    /// The complete, fixed set of agent IDs for this environment.
    fn possible_agents(&self) -> &[Self::AgentId];

    /// The agents currently active in this episode.
    fn agents(&self) -> &[Self::AgentId];

    /// The agent whose turn it currently is to act.
    fn agent_selection(&self) -> &Self::AgentId;

    /// Execute the current agent's action and advance to the next agent.
    ///
    /// Pass `None` when `agent_state(agent_selection())` is done, to cycle
    /// the terminated or truncated agent out without applying an action.
    fn step(&mut self, action: Option<Self::Action>);

    /// Reset the environment to an initial state.
    ///
    /// Unlike [`ParallelEnvironment::reset`], this returns nothing. Retrieve
    /// initial observations via `observe()` after calling `reset()`.
    fn reset(&mut self, seed: Option<u64>);

    /// Retrieve the current observation for the given agent.
    ///
    /// Returns `None` if the agent has terminated or been truncated — their
    /// last observation is no longer valid.
    fn observe(&self, agent: &Self::AgentId) -> Option<Self::Observation>;

    /// Retrieve the persistent (reward, status, info) state for the given agent.
    ///
    /// This state is updated by `step()` and persists until the next time that
    /// agent acts. It reflects what the agent received as a result of its last
    /// action.
    fn agent_state(&self, agent: &Self::AgentId) -> (f64, EpisodeStatus, Self::Info);

    /// Sample a random action for the given agent.
    fn sample_action(&self, agent: &Self::AgentId, rng: &mut impl Rng) -> Self::Action;

    /// Convenience: returns the full state for the currently selected agent.
    ///
    /// Equivalent to calling `observe(agent_selection())` and
    /// `agent_state(agent_selection())`. This is the idiomatic way to read the
    /// current agent's situation before deciding on an action.
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
