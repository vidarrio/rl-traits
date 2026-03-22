//! Pursuit — a two-predator cooperative tracking task.
//!
//! Two predators on a 1-D grid of length 10 cooperate to catch a randomly
//! moving prey. This example validates the [`rl_traits::ParallelEnvironment`]
//! API: per-agent observations, joint actions, and the `Terminated` /
//! `Truncated` distinction across agents.
//!
//! Run with:
//! ```text
//! cargo run --example pursuit
//! ```

use std::collections::HashMap;

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng as _};
use rl_traits::{EpisodeStatus, ParallelEnvironment, StepResult};

// ── Constants ────────────────────────────────────────────────────────────────

const GRID_LEN: u8 = 10;
const MAX_STEPS: usize = 200;

// ── Observation ──────────────────────────────────────────────────────────────

/// `[predator_pos / (GRID_LEN - 1), prey_pos / (GRID_LEN - 1)]`
///
/// Both values are normalised to `[0.0, 1.0]`.
pub type PursuitObs = [f32; 2];

// ── Environment ──────────────────────────────────────────────────────────────

pub struct Pursuit {
    active: Vec<u8>,      // currently active predator IDs (0 and/or 1)
    pos: [u8; 2],         // predator positions indexed by ID
    prey: u8,             // prey position
    step: usize,
    rng: SmallRng,
}

impl Pursuit {
    pub fn new(seed: u64) -> Self {
        Self {
            active: vec![0, 1],
            pos: [0, GRID_LEN - 1],
            prey: GRID_LEN / 2,
            step: 0,
            rng: SmallRng::seed_from_u64(seed),
        }
    }

    fn obs(&self, id: u8) -> PursuitObs {
        let scale = (GRID_LEN - 1) as f32;
        [self.pos[id as usize] as f32 / scale, self.prey as f32 / scale]
    }

    fn clamp_move(pos: u8, action: u8) -> u8 {
        match action {
            0 => pos.saturating_sub(1),
            _ => (pos + 1).min(GRID_LEN - 1),
        }
    }
}

impl ParallelEnvironment for Pursuit {
    type AgentId = u8;
    type Observation = PursuitObs;
    type Action = u8;   // 0 = move left, 1 = move right
    type Info = ();

    fn possible_agents(&self) -> &[u8] { &[0, 1] }
    fn agents(&self) -> &[u8] { &self.active }

    fn step(&mut self, actions: HashMap<u8, u8>)
        -> HashMap<u8, StepResult<PursuitObs, ()>>
    {
        for (&id, &action) in &actions {
            self.pos[id as usize] = Self::clamp_move(self.pos[id as usize], action);
        }

        // Prey moves randomly, bouncing at the walls.
        self.prey = Self::clamp_move(self.prey, self.rng.gen_range(0..2));
        self.step += 1;

        let caught = self.active.iter().any(|&id| self.pos[id as usize] == self.prey);

        let status = if caught {
            EpisodeStatus::Terminated
        } else if self.step >= MAX_STEPS {
            EpisodeStatus::Truncated
        } else {
            EpisodeStatus::Continuing
        };

        // Build results before mutating the active list.
        let results = self.active.iter().map(|&id| {
            let reward = if caught && self.pos[id as usize] == self.prey { 1.0 } else { 0.0 };
            (id, StepResult::new(self.obs(id), reward, status.clone(), ()))
        }).collect();

        if status.is_done() {
            self.active.clear();
        }

        results
    }

    fn reset(&mut self, seed: Option<u64>) -> HashMap<u8, (PursuitObs, ())> {
        if let Some(s) = seed {
            self.rng = SmallRng::seed_from_u64(s);
        }
        self.active = vec![0, 1];
        self.pos = [0, GRID_LEN - 1];
        self.prey = GRID_LEN / 2;
        self.step = 0;
        [0u8, 1u8].iter().map(|&id| (id, (self.obs(id), ()))).collect()
    }

    fn sample_action(&self, _agent: &u8, rng: &mut impl Rng) -> u8 {
        rng.gen_range(0..2)
    }
}

// ── Demo loop ────────────────────────────────────────────────────────────────

fn run_episode(env: &mut Pursuit, rng: &mut SmallRng) -> ([f64; 2], EpisodeStatus, usize) {
    env.reset(None);
    let mut returns = [0.0_f64; 2];
    let mut steps = 0;
    let mut outcome = EpisodeStatus::Continuing;

    while !env.is_done() {
        let actions = env.agents().iter()
            .map(|&id| (id, env.sample_action(&id, rng)))
            .collect();

        let results = env.step(actions);
        steps += 1;

        for (id, result) in &results {
            returns[*id as usize] += result.reward;
            if result.status.is_done() {
                outcome = result.status.clone();
            }
        }
    }

    (returns, outcome, steps)
}

fn main() {
    const NUM_EPISODES: usize = 10;
    const ENV_SEED: u64 = 42;

    let mut env = Pursuit::new(ENV_SEED);
    let mut rng = SmallRng::seed_from_u64(0);

    println!("Pursuit — random predators, {NUM_EPISODES} episodes\n");
    println!("{:<8} {:>8} {:>8} {:>7} {:>10}", "Episode", "Pred. 0", "Pred. 1", "Steps", "Outcome");
    println!("{}", "-".repeat(46));

    let mut caught = 0;

    for ep in 1..=NUM_EPISODES {
        let (returns, outcome, steps) = run_episode(&mut env, &mut rng);

        let label = match outcome {
            EpisodeStatus::Terminated => { caught += 1; "Caught" }
            EpisodeStatus::Truncated  => "Escaped",
            EpisodeStatus::Continuing => unreachable!(),
        };

        println!("{ep:<8} {:>8.1} {:>8.1} {:>7} {:>10}",
            returns[0], returns[1], steps, label);
    }

    println!("{}", "-".repeat(46));
    println!("Prey caught in {caught}/{NUM_EPISODES} episodes");
}
