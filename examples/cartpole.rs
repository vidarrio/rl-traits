//! CartPole-v1 — a reference implementation of the classic control task.
//!
//! This example exists to validate the [`rl_traits`] API under a realistic
//! environment. It mirrors the semantics of Gymnasium's `CartPole-v1`:
//!
//! - 4-dimensional continuous observation space: `[x, ẋ, θ, θ̇]`
//! - Discrete action space: `0` = push left, `1` = push right
//! - +1 reward every step the pole stays up
//! - **Terminated** when the pole tips past ±12° or the cart leaves ±2.4 m
//! - **Truncated** at 500 steps via [`rl_traits::TimeLimit`]
//!
//! Run with:
//! ```text
//! cargo run --example cartpole
//! ```

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng as _};
use rl_traits::{EpisodeStatus, Environment, StepResult, TimeLimit};

// ── Physical constants (identical to Gymnasium's CartPole-v1) ────────────────

const GRAVITY: f32 = 9.8;
const MASS_CART: f32 = 1.0;
const MASS_POLE: f32 = 0.1;
const TOTAL_MASS: f32 = MASS_CART + MASS_POLE;
const HALF_POLE_LEN: f32 = 0.5; // centre of mass is at the midpoint
const POLE_MASS_LEN: f32 = MASS_POLE * HALF_POLE_LEN;
const FORCE_MAG: f32 = 10.0;
const TAU: f32 = 0.02; // seconds per step

const X_THRESHOLD: f32 = 2.4;
const THETA_THRESHOLD_RAD: f32 = 12.0 * std::f32::consts::PI / 180.0;

const MAX_STEPS: usize = 500;

// ── State ────────────────────────────────────────────────────────────────────

/// `[cart_position, cart_velocity, pole_angle, pole_angular_velocity]`
pub type CartPoleObs = [f32; 4];

// ── Environment ──────────────────────────────────────────────────────────────

pub struct CartPole {
    state: CartPoleObs,
    rng: SmallRng,
}

impl CartPole {
    pub fn new(seed: u64) -> Self {
        Self {
            state: [0.0; 4],
            rng: SmallRng::seed_from_u64(seed),
        }
    }

    fn is_terminal(state: &CartPoleObs) -> bool {
        state[0].abs() > X_THRESHOLD || state[2].abs() > THETA_THRESHOLD_RAD
    }
}

impl Environment for CartPole {
    type Observation = CartPoleObs;
    type Action = usize; // 0 = left, 1 = right
    type Info = ();

    fn step(&mut self, action: usize) -> StepResult<CartPoleObs, ()> {
        let [x, x_dot, theta, theta_dot] = self.state;

        let force = if action == 1 { FORCE_MAG } else { -FORCE_MAG };
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();

        // Equations of motion (Euler integration, same as Gymnasium)
        let temp = (force + POLE_MASS_LEN * theta_dot * theta_dot * sin_theta) / TOTAL_MASS;
        let theta_acc = (GRAVITY * sin_theta - cos_theta * temp)
            / (HALF_POLE_LEN * (4.0 / 3.0 - MASS_POLE * cos_theta * cos_theta / TOTAL_MASS));
        let x_acc = temp - POLE_MASS_LEN * theta_acc * cos_theta / TOTAL_MASS;

        let new_x = x + TAU * x_dot;
        let new_x_dot = x_dot + TAU * x_acc;
        let new_theta = theta + TAU * theta_dot;
        let new_theta_dot = theta_dot + TAU * theta_acc;

        self.state = [new_x, new_x_dot, new_theta, new_theta_dot];

        let status = if Self::is_terminal(&self.state) {
            EpisodeStatus::Terminated
        } else {
            EpisodeStatus::Continuing
        };

        // +1 every step the pole stays up (reward is 0 on the terminal step
        // in Gymnasium; we match that here)
        let reward = if status == EpisodeStatus::Continuing {
            1.0
        } else {
            0.0
        };

        StepResult::new(self.state, reward, status, ())
    }

    fn reset(&mut self, seed: Option<u64>) -> (CartPoleObs, ()) {
        if let Some(s) = seed {
            self.rng = SmallRng::seed_from_u64(s);
        }
        // Uniform [-0.05, 0.05] for all four state variables
        self.state = self.rng.gen::<[f32; 4]>().map(|v| v * 0.1 - 0.05);
        (self.state, ())
    }

    fn sample_action(&self, rng: &mut impl Rng) -> usize {
        rng.gen_range(0..2)
    }
}

// ── Demo loop ────────────────────────────────────────────────────────────────

fn run_episode(env: &mut TimeLimit<CartPole>, rng: &mut SmallRng) -> (f64, EpisodeStatus, usize) {
    env.reset(None);
    let mut total_reward = 0.0;
    let mut steps = 0;

    loop {
        let action = env.sample_action(rng);
        let result = env.step(action);
        total_reward += result.reward;
        steps += 1;
        if result.is_done() {
            return (total_reward, result.status, steps);
        }
    }
}

fn main() {
    const NUM_EPISODES: usize = 10;
    const ENV_SEED: u64 = 42;

    let mut env = TimeLimit::new(CartPole::new(ENV_SEED), MAX_STEPS);
    let mut rng = SmallRng::seed_from_u64(0);

    println!("CartPole-v1 — random agent, {NUM_EPISODES} episodes\n");
    println!("{:<8} {:>8} {:>7} {:>12}", "Episode", "Return", "Steps", "Outcome");
    println!("{}", "-".repeat(40));

    let mut total_return = 0.0;

    for ep in 1..=NUM_EPISODES {
        let (ret, status, steps) = run_episode(&mut env, &mut rng);
        total_return += ret;

        let outcome = match status {
            EpisodeStatus::Terminated => "Terminated",
            EpisodeStatus::Truncated => "Truncated ",
            EpisodeStatus::Continuing => unreachable!(),
        };

        println!("{ep:<8} {ret:>8.1} {steps:>7} {outcome:>12}");
    }

    println!("{}", "-".repeat(40));
    println!(
        "Mean return over {NUM_EPISODES} episodes: {:.1}",
        total_return / NUM_EPISODES as f64
    );
}
