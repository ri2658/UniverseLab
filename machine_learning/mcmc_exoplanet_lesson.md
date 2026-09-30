# MCMC & Parameter Fitting

Since you are already somewhat familiar with the **Metropolis-Hastings (MH)** algorithm and its ties to statistical mechanics (like the Boltzmann distribution), you are in a great spot! 

In physics, MH is often used to sample from the Boltzmann distribution to find the thermodynamic states of a system. But in **computational astrophysics**, we hijack this exact same algorithm for **Bayesian Parameter Fitting**.

Instead of a particle taking a random walk through physical space to find a state of low energy, we have our algorithm take a random walk through *Parameter Space* to find a state of "high probability".

## The Astrophysics Problem: Exoplanet Radial Velocity
Imagine we are observing a star. We can't see the planet orbiting it, but we can see the star "wobbling" back and forth due to the planet's gravity. We measure the star's radial velocity (velocity toward/away from Earth) over several days. 

Because our telescopes aren't perfect, our data is noisy. We want to find the **Mass** of the hidden planet.

We have a simple physics model for the star's velocity $V$ at time $t$ based on the planet's mass $M$:
$$V(t) = M \cdot \sin(2\pi \cdot t / P)$$
*(Where $P$ is the known orbital period)*

## Bayes' Theorem & The Likelihood Function
To use MCMC, we need a way to score how "good" a guessed mass $M$ is. We use Bayes' theorem, but we usually just focus on the **Likelihood** ($L$).

If we guess a mass $M=5$, our physics model generates a perfect, smooth sine wave. We then compare this perfect wave to our noisy telescope data.

Assuming the telescope noise follows a Gaussian (Normal) distribution, the probability of seeing our exact data given our guessed mass is:
$$L \propto \exp\left(-\sum \frac{(Data_i - Model_i)^2}{2 \sigma^2} \right)$$
*(Notice how much this looks like the Boltzmann distribution $e^{-E/kT}$? Here, the "Energy" is just the squared error between our model and the data!)*

## The Metropolis-Hastings Algorithm for Parameter Fitting
Here is how we use MH to find the mass:

1. **Start** with a random guess for the Mass (e.g., $M_{current} = 1.0$).
2. **Calculate** the Likelihood of $M_{current}$.
3. **Propose** a new mass by adding a small random Gaussian jump: $M_{proposed} = M_{current} + \text{random\_step}$.
4. **Calculate** the Likelihood of $M_{proposed}$.
5. **Accept or Reject:**
   - If $\text{Likelihood}(M_{proposed}) > \text{Likelihood}(M_{current})$, we **ACCEPT** the new guess. (It fits the data better!)
   - If it's worse, we don't automatically reject it. We accept it with a probability equal to the ratio: $P_{accept} = \frac{\text{Likelihood}(M_{proposed})}{\text{Likelihood}(M_{current})}$.
6. Repeat this loop 10,000 times!

By allowing the algorithm to occasionally accept "worse" guesses, it prevents it from getting stuck in local minimums (just like thermal fluctuations in a Hamiltonian system!).

After 10,000 steps, if we plot a histogram of all the masses the algorithm visited, it will perfectly map out the **Posterior Probability Distribution** of the true exoplanet mass!

## Why is MCMC Mathematically Unbiased? (Detailed Balance)

It might seem like randomly walking and occasionally accepting bad guesses would give you a biased or messy result. However, MCMC perfectly maps the posterior because the Metropolis-Hastings algorithm mathematically satisfies a condition called **Detailed Balance** (also known as microscopic reversibility).

For a Markov Chain to perfectly sample a target probability distribution (let's call the true posterior probability $\pi(x)$), the transitions between any two states $x$ and $x'$ must balance out. If you have an ensemble of walkers, the flow from $x$ to $x'$ must equal the flow from $x'$ to $x$:

$$ \pi(x) \cdot T(x \to x') = \pi(x') \cdot T(x' \to x) $$

Where $T(x \to x')$ is the probability of the algorithm transitioning from $x$ to $x'$. In Metropolis-Hastings, this transition probability is split into two steps:
1.  **The Proposal ($g$):** The probability of randomly suggesting $x'$. Because we use a symmetrical Gaussian jump, stepping from $x$ to $x'$ is just as likely as stepping from $x'$ to $x$. Thus, $g(x \to x') = g(x' \to x)$.
2.  **The Acceptance ($A$):** The ratio we defined earlier: $\min\left(1, \frac{\pi(x')}{\pi(x)}\right)$.

Let's plug these into the Detailed Balance equation! Assume $x'$ is a *worse* guess than $x$ (so $\pi(x') < \pi(x)$). 
This means jumping to the worse state has an acceptance probability of $A(x \to x') = \frac{\pi(x')}{\pi(x)}$. 
Jumping *back* to the better state is always accepted, so $A(x' \to x) = 1$.

*   **Left Side (Flow $x \to x'$):** $\pi(x) \cdot \left[ g(x \to x') \cdot \frac{\pi(x')}{\pi(x)} \right] = \pi(x') \cdot g(x \to x')$
*   **Right Side (Flow $x' \to x$):** $\pi(x') \cdot \left[ g(x' \to x) \cdot 1 \right] = \pi(x') \cdot g(x \to x')$

**The left side perfectly equals the right side.** Because the MH acceptance ratio artificially chokes the flow into bad states by the *exact mathematical amount* required to balance the equation, the algorithm is strictly unbiased. The fraction of time the walker spends hovering around a mass $M$ is mathematically guaranteed to converge exactly to the true posterior probability of that mass!

## Your Challenge
Let's build this from scratch. I've designed a script where we will:
1. Generate synthetic, noisy telescope data for a fake star.
2. Write the MCMC MH loop to blindly guess the mass of the planet.
3. Plot a histogram of the results to see if the algorithm found the true mass.

Are you ready to create `machine_learning/mcmc_exoplanet.py`?
