# MCMC & Parameter Fitting

In physics, Metropolis-Hastings (MH) is often used to sample from the Boltzmann distribution to find the thermodynamic states of a system. But in **computational astrophysics**, we hijack this exact same algorithm for **Bayesian Parameter Fitting**.

Instead of a particle taking a random walk through physical space, visiting each state with probability $\propto e^{-E/kT}$, our algorithm takes a random walk through *parameter space*, visiting each parameter value in proportion to how probable it is given the data.

## The Astrophysics Problem: Exoplanet Radial Velocity
Imagine we are observing a star. We can't see the planet orbiting it, but we can see the star "wobbling" back and forth due to the planet's gravity. We measure the star's radial velocity (velocity toward/away from Earth) over several days.

Because our telescopes aren't perfect, our data is noisy. We want to learn about the hidden planet's **mass**.

For a planet on a circular orbit, the star's radial velocity is a sinusoid:
$$V(t) = K \sin(2\pi t / P)$$
*(Where $P$ is the known orbital period, and we've assumed we know the phase too.)*

The amplitude $K$ (the **semi-amplitude**, in m/s) is what the data measure directly. It is not the mass itself, but it is set by the mass: for a circular orbit,
$$K = \left(\frac{2\pi G}{P}\right)^{1/3} \frac{m \sin i}{(M_\star + m)^{2/3}},$$
where $m$ is the planet's mass, $M_\star$ the star's, and $i$ the orbit's inclination. So fitting $K$ *is* measuring the planet's (minimum) mass. The Keplerian lesson derives this formula and converts $K$ into kilograms. Here we fit $K$ itself, which is what `mcmc_exoplanet.py` does (true value: $K = 50$ m/s).

## Bayes' Theorem & The Likelihood Function
To use MCMC, we need a way to score how "good" a guessed amplitude $K$ is. Bayes' theorem says

$$\underbrace{p(K \mid \text{data})}_{\text{posterior}} = \frac{\overbrace{p(\text{data} \mid K)}^{\text{likelihood } L(K)}\;\overbrace{p(K)}^{\text{prior}}}{\underbrace{p(\text{data})}_{\text{evidence}}}$$

The evidence $p(\text{data})$ doesn't depend on $K$; it is just the normalizing constant. In this first script we also take a **flat prior** (every $K$ equally plausible beforehand), so the posterior is simply proportional to the likelihood: $p(K \mid \text{data}) \propto L(K)$. (Strictly speaking, a prior that is flat over the entire real line can't be normalized. It's an *improper* prior. It's harmless here because the likelihood falls off fast enough in both directions that the posterior can still be normalized.)

If we guess $K = 5$ m/s, our physics model generates a perfect, smooth sine wave. We then compare this perfect wave to our noisy telescope data.

Assume each measurement $v_i$ has independent Gaussian noise with standard deviation $\sigma_i$ (its error bar). The probability density of the whole data set is the product of the individual Gaussians:
$$L(K) = \prod_i \frac{1}{\sqrt{2\pi\sigma_i^2}} \exp\left(-\frac{(v_i - V(t_i))^2}{2\sigma_i^2}\right) \;\propto\; \exp\left(-\frac{\chi^2}{2}\right), \qquad \chi^2 = \sum_i \frac{(v_i - V(t_i))^2}{\sigma_i^2}$$
The prefactor doesn't depend on $K$, so it drops out. In code we always work with $\ln L = -\chi^2/2 + \text{const}$, because multiplying many small probabilities underflows floating point.

*(Notice how much this looks like the Boltzmann distribution $e^{-E/kT}$? It's the same form with $E = \chi^2/2$ and $kT = 1$. The "energy" is the error-weighted squared misfit between the model and the data.)*

## The Metropolis-Hastings Algorithm for Parameter Fitting
Here is how we use MH to sample $K$:

1. **Start** with a guess (e.g., $K_{current} = 10$ m/s).
2. **Calculate** the likelihood of $K_{current}$.
3. **Propose** a new value by adding a random Gaussian jump: $K_{proposed} = K_{current} + \epsilon$, with $\epsilon \sim \mathcal{N}(0, \text{step}^2)$.
4. **Calculate** the likelihood of $K_{proposed}$.
5. **Accept or Reject:** accept with probability
   $$A = \min\left(1,\; \frac{L(K_{proposed})}{L(K_{current})}\right)$$
   - If the proposal fits better, the ratio is $> 1$ and we **always accept**.
   - If it's worse, we don't automatically reject it. We accept it with probability equal to the ratio.
   - If we reject, the chain **stays where it is, and the current value is recorded again**. Those repeats matter: they are how the chain spends more time in more probable regions.
6. Repeat this loop 10,000 times!

Accepting "worse" guesses is not about escaping local optima, the way it is in simulated annealing. It's the whole point: an optimizer would climb to the single best $K$ and stop. A sampler has to visit less-likely values too, **in proportion to their probability**, so that the collection of visited points reflects our uncertainty.

After discarding an initial **burn-in** (the steps spent walking from the arbitrary starting point into the high-probability region), a histogram of the visited values approximates the **posterior distribution** $p(K \mid \text{data})$. The approximation improves as the chain runs longer. (How long is "long enough" is a real question, answered in the Keplerian lesson with the *autocorrelation time*.)

## Why Does MCMC Target the Right Distribution? (Detailed Balance)

It might seem like randomly walking and occasionally accepting bad guesses would give you a biased or messy result. It doesn't, because the Metropolis-Hastings transition rule satisfies a condition called **detailed balance** (also known as microscopic reversibility).

Call the target distribution (our posterior) $\pi(x)$, and let $T(x \to x')$ be the probability density that one step of the algorithm moves the walker from $x$ to $x'$. Detailed balance requires that, for every pair of states, the probability flow from $x$ to $x'$ equals the flow back:

$$ \pi(x) \, T(x \to x') = \pi(x') \, T(x' \to x) $$

**Why this is enough.** Integrate both sides over $x$:
$$\int \pi(x)\,T(x \to x')\,dx = \pi(x') \int T(x' \to x)\,dx = \pi(x'),$$
because the walker has to go *somewhere* (the transition density integrates to 1). The left side is the distribution of the walker after one step, if it started out distributed as $\pi$. So $\pi$ is a **stationary distribution**: once the walker is distributed according to $\pi$, it stays that way.

In Metropolis-Hastings, the transition from $x$ to a different state $x' \ne x$ happens in two stages, so $T(x \to x') = g(x \to x')\,A(x \to x')$:
1. **The proposal $g$:** the probability density of suggesting $x'$. Our Gaussian jump is symmetric: stepping from $x$ to $x'$ is exactly as likely as stepping from $x'$ to $x$, so $g(x \to x') = g(x' \to x)$.
2. **The acceptance $A$:** $A(x \to x') = \min\left(1, \frac{\pi(x')}{\pi(x)}\right)$.

(Rejections put the walker back at $x$ itself. Detailed balance with $x' = x$ holds trivially, so we only need to check $x' \ne x$.)

Take any $x \ne x'$, and without loss of generality let $x'$ be the *less* probable state, $\pi(x') \le \pi(x)$. (The other case is the same argument with the labels swapped.) Then $A(x \to x') = \pi(x')/\pi(x)$ and $A(x' \to x) = 1$:

*   **Left side (flow $x \to x'$):** $\pi(x) \cdot g(x \to x') \cdot \dfrac{\pi(x')}{\pi(x)} = \pi(x')\, g(x \to x')$
*   **Right side (flow $x' \to x$):** $\pi(x') \cdot g(x' \to x) \cdot 1 = \pi(x')\, g(x \to x')$ (using symmetry of $g$)

**The two sides are equal.** The acceptance rule throttles the flow into less probable states by exactly the amount needed to balance the flow back out.

Two consequences are worth highlighting:

- **Only ratios of $\pi$ appear.** The unknown evidence $p(\text{data})$ cancels, which is why MCMC can sample a posterior we can't normalize. This is the same reason MH works in statistical mechanics without ever computing the partition function $Z$.
- **Asymmetric proposals need a correction.** If $g(x \to x') \ne g(x' \to x)$, the general (Hastings) acceptance is
  $$A(x \to x') = \min\left(1,\; \frac{\pi(x')\,g(x' \to x)}{\pi(x)\,g(x \to x')}\right),$$
  and the same algebra goes through. The Keplerian lesson's stretch move is an example: that's where its mysterious $z^{d-1}$ factor comes from.

**What detailed balance does *not* guarantee.** Stationarity says $\pi$ is a fixed point. To be sure the chain actually *converges* to it from an arbitrary start, it also has to be able to reach every region of non-zero probability (*irreducibility*) and must not get trapped in a deterministic cycle (*aperiodicity*). Our Gaussian proposal satisfies both. When they hold, the ergodic theorem says that averages over the chain, such as the mean or the histogram of $K$, converge to the corresponding posterior quantities as the number of steps goes to infinity.

With a finite chain, the estimates are **consistent** (they approach the right answer as the chain grows), but not exactly unbiased. The start of the chain remembers the arbitrary initial guess, which is why we discard the burn-in. And consecutive samples are correlated, so 10,000 steps carry less information than 10,000 independent draws would.

## Your Challenge
To implement this, in `machine_learning/mcmc_exoplanet.py, we will:

1. Generate synthetic, noisy telescope data for a fake star.
2. Write the MCMC MH loop to blindly infer the semi-amplitude $K$ of the planet's signal.
3. Plot a histogram of the results to see if the algorithm found the true value.
