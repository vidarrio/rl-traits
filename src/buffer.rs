use rand::Rng;

use crate::experience::Experience;

/// A buffer that stores past experience for agent training.
///
/// Used primarily by off-policy algorithms (DQN, SAC, TD3) to break
/// temporal correlations between training samples. On-policy algorithms
/// (PPO, A2C) typically collect fixed-length trajectories instead and
/// don't need this trait — they can use a plain `Vec<Experience<O, A>>`.
///
/// # Implementing this trait
///
/// The most common implementation is a circular buffer with a fixed
/// capacity that overwrites the oldest experience when full. Concrete
/// implementations live in ember-rl, not here.
///
/// # Bounds
///
/// `O: Clone + Send + Sync` and `A: Clone + Send + Sync` are required
/// because sampling returns owned `Experience` values (not references),
/// and buffers may be accessed across threads during async training.
pub trait ReplayBuffer<O, A>
where
    O: Clone + Send + Sync,
    A: Clone + Send + Sync,
{
    /// Add a new experience to the buffer.
    ///
    /// If the buffer is at capacity, implementations should overwrite the
    /// oldest experience (FIFO eviction).
    fn push(&mut self, experience: Experience<O, A>);

    /// Sample a random batch of `batch_size` experiences.
    ///
    /// Sampling is done with replacement. The caller supplies the RNG so
    /// sampling randomness can be seeded and controlled independently.
    ///
    /// # Panics
    ///
    /// Implementations may panic if `batch_size > self.len()`.
    fn sample(&self, batch_size: usize, rng: &mut impl Rng) -> Vec<Experience<O, A>>;

    /// The number of experiences currently stored.
    fn len(&self) -> usize;

    /// Returns `true` if the buffer contains no experiences.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The maximum number of experiences the buffer can hold, if bounded.
    ///
    /// Returns `None` for unbounded buffers (e.g. trajectory collectors).
    fn capacity(&self) -> Option<usize>;

    /// Returns `true` if the buffer is at capacity.
    ///
    /// When full, the next `push()` will overwrite the oldest experience.
    fn is_full(&self) -> bool {
        self.capacity().is_some_and(|cap| self.len() >= cap)
    }

    /// Returns `true` if the buffer has enough experience to sample a
    /// batch of the given size.
    ///
    /// Useful for deciding when to start training in the warm-up phase.
    fn ready_for(&self, batch_size: usize) -> bool {
        self.len() >= batch_size
    }
}
