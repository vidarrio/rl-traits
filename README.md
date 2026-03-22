# rl-traits

[![crates.io](https://img.shields.io/crates/v/rl-traits.svg)](https://crates.io/crates/rl-traits)
[![docs.rs](https://docs.rs/rl-traits/badge.svg)](https://docs.rs/rl-traits)
[![CI](https://github.com/vidarrio/rl-traits/actions/workflows/ci.yml/badge.svg)](https://github.com/vidarrio/rl-traits/actions/workflows/ci.yml)

Core traits for reinforcement learning environments, policies, and agents.

`rl-traits` defines the shared vocabulary for a Rust RL ecosystem. It is
deliberately small — no algorithms, no neural networks, no rendering. Those
belong in the crates that depend on this one.

## Ecosystem

| Crate | Role |
|---|---|
| **rl-traits** | Shared traits and types (this crate) |
| `ember-rl` *(planned)* | Algorithm implementations (DQN, PPO, SAC) using Burn |
| `bevy-gym` *(planned)* | Bevy ECS plugin for visualising and parallelising environments |

## Design goals

**Type-safe observation and action spaces.** `Observation` and `Action` are
associated types. Feeding a CartPole observation to a MuJoCo agent is a
compile error, not a runtime panic.

**Correct `Terminated` vs `Truncated` distinction.** This is one of the most
common bugs in policy gradient implementations. Bootstrapping algorithms (PPO,
DQN, SAC) must zero the next-state value on natural termination but *not* on
truncation. [`EpisodeStatus`] encodes this from the start.

**Rendering-free.** `Environment` has no `render()` method. Visualisation is
`bevy-gym`'s concern.

**Bevy-compatible.** `Send + Sync + 'static` bounds on associated types mean
any `Environment` implementation can be a Bevy `Component`, enabling free ECS
parallelisation via `Query::par_iter_mut()`.

**Minimal dependencies.** Only `rand` for RNG abstractions.

## Usage

```rust
use rl_traits::{Environment, EpisodeStatus, StepResult};
use rand::Rng;

struct MyEnv;

impl Environment for MyEnv {
    type Observation = f32;
    type Action = usize;
    type Info = ();

    fn step(&mut self, action: usize) -> StepResult<f32, ()> {
        StepResult::new(0.0, 1.0, EpisodeStatus::Continuing, ())
    }

    fn reset(&mut self, _seed: Option<u64>) -> (f32, ()) {
        (0.0, ())
    }

    fn sample_action(&self, rng: &mut impl Rng) -> usize {
        rng.gen_range(0..4)
    }
}
```

Episode step results carry a typed status:

```rust
match result.status {
    EpisodeStatus::Continuing  => { /* keep going */ }
    EpisodeStatus::Terminated  => { /* natural end — bootstrap with zero */ }
    EpisodeStatus::Truncated   => { /* time limit hit — bootstrap with V(s') */ }
}
```

Use `Experience::bootstrap_mask()` to apply this correctly in TD updates:

```rust
let target = exp.reward + gamma * exp.bootstrap_mask() * value_of_next_state;
```

Wrap any environment with `TimeLimit` to truncate episodes after a fixed number
of steps (emitting `Truncated`, not `Terminated`):

```rust
let env = TimeLimit::new(MyEnv, 500);
```

## Multi-agent environments

Two APIs are available, mirroring PettingZoo's split:

**`ParallelEnvironment`** — all agents act simultaneously each step. The
natural fit for cooperative and competitive tasks, and for Bevy since a single
system call produces results for all agents at once.

**`AecEnvironment`** — agents act one at a time (Agent Environment Cycle).
Use this for turn-based domains like board games and card games.

Both APIs share the `possible_agents` / `agents` distinction: `possible_agents`
is the fixed universe of all agent IDs; `agents` is the live subset for the
current episode, shrinking as agents terminate mid-episode.

```rust
// Parallel: step with joint actions, get per-agent results
let actions = env.agents().iter()
    .map(|id| (id.clone(), env.sample_action(id, &mut rng)))
    .collect();
let results = env.step(actions);  // HashMap<AgentId, StepResult<…>>

// AEC: read current agent, act or cycle out if done
let (obs, _reward, status, _info) = env.last();
let action = if status.is_done() { None } else { Some(policy(obs.unwrap())) };
env.step(action);
```

Bevy `Entity` satisfies all `AgentId` bounds directly, so agents can be ECS
entities without any extra indirection.

## Reference examples

`examples/cartpole.rs` implements CartPole-v1 against `Environment`. It serves
as a validation of the single-agent API ergonomics and a reference for how to
implement `Environment`. Run it with:

```
cargo run --example cartpole
```

`examples/pursuit.rs` implements a two-predator cooperative tracking task
against `ParallelEnvironment`. Two predators on a 1-D grid cooperate to catch
a randomly moving prey, demonstrating per-agent observations, joint actions,
and the `Terminated` / `Truncated` distinction across agents. Run it with:

```
cargo run --example pursuit
```

## Development

This crate was developed with the assistance of AI coding tools (Claude by Anthropic).

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT License](LICENSE-MIT) at your option.
