# Keplerian Orbits & Ensemble MCMC

In `mcmc_exoplanet.py` you recovered a planet signal with Metropolis-Hastings. You had a lot of help, though. The model was

$$V(t) = K \sin(2\pi t / P)$$

and you already knew the period $P$, the phase, and that the orbit was a perfect circle. You only had to fit **one** number.

Real planet hunters get none of that for free. In this lesson we remove every simplification:

| Before | Now |
|---|---|
| Sine wave | Full **Keplerian** orbit (eccentric, with Kepler's equation) |
| 1 parameter | **7 parameters**: period, amplitude, eccentricity, orientation, phase, systemic velocity, jitter |
| Period known | Period **found from the data** (periodogram → optimizer → MCMC) |
| Random-walk MH | **Affine-invariant ensemble sampler** (Goodman & Weare, the algorithm inside `emcee`) |
| "Discard 1000 steps" | Convergence measured with the **autocorrelation time** |
| Fake data from a formula | Data from **your own leapfrog N-body simulation** |

That last row is the best part. You'll generate the "telescope data" with the gravity engine from `gravity/`, then check that an analytic formula reproduces it. Two methods you wrote independently have to agree.

---

## Part 1: The Physics of a Wobbling Star

### Setting up the two-body problem

A star (mass $M$) and a planet (mass $m$) both orbit their shared **centre of mass**. The standard trick is to split the motion into two pieces:

1. The **centre of mass** moves in a straight line (we put it at rest at the origin).
2. The **relative vector** $\vec{r} = \vec{r}_\text{planet} - \vec{r}_\text{star}$ obeys a one-body problem: an ellipse around a fixed mass $M + m$.

Once you know $\vec{r}$, each body's position is a fixed fraction of it:

$$\vec{r}_\text{planet} = +\frac{M}{M+m}\,\vec{r}, \qquad \vec{r}_\text{star} = -\frac{m}{M+m}\,\vec{r}$$

The same split works for velocities. The star traces a **tiny copy** of the planet's orbit, scaled down by $m/(M+m)$ (about 1/500 for a 2 Jupiter-mass planet) and flipped to the opposite side.

To launch an eccentric orbit you need two textbook results for the relative orbit:

- **Kepler's third law** fixes the semi-major axis from the period: $a^3 = \dfrac{G(M+m)P^2}{4\pi^2}$
- **Vis-viva** gives the speed at any distance: $v^2 = G(M+m)\left(\dfrac{2}{r} - \dfrac{1}{a}\right)$

At periastron (closest approach), $r_p = a(1-e)$. Plugging that into vis-viva gives

$$v_p = \sqrt{\frac{G(M+m)}{a}\cdot\frac{1+e}{1-e}}$$

and the velocity there is perpendicular to $\vec{r}$. That's everything you need for **TODO 1**.

### Three anomalies: where is the planet at time $t$?

For a circle, the angle grows uniformly: $\theta = 2\pi t/P$. For an ellipse it doesn't. Kepler's second law says the planet sweeps equal *areas* in equal times, so it races through periastron and dawdles at apoastron. Astronomers describe the position with three angles (for historical reasons called "anomalies"):

| Anomaly | Symbol | Meaning |
|---|---|---|
| **Mean** | $M$ | A fictitious angle that *does* grow uniformly: $M = \frac{2\pi}{P}(t - t_p)$. It's really just "fraction of the orbit elapsed" in radians. |
| **Eccentric** | $E$ | A geometric helper angle, measured on a circle circumscribed around the ellipse. |
| **True** | $\nu$ | The actual angle between the planet and periastron, as seen from the focus. This is the physical one. |

Mean and eccentric anomaly are linked by **Kepler's equation**:

$$M = E - e \sin E$$

Going from $E$ to $M$ is trivial. We need the opposite direction ($M$ is what we know from the time), and that has **no closed-form solution**. This one equation launched centuries of numerical methods; Newton himself worked on it. We use **Newton-Raphson**: to solve $f(E) = E - e\sin E - M = 0$, repeat

$$E_{n+1} = E_n - \frac{f(E_n)}{f'(E_n)} = E_n - \frac{E_n - e\sin E_n - M}{1 - e\cos E_n}$$

Newton-Raphson converges *quadratically*: the number of correct digits roughly doubles each iteration. From the starting guess $E_0 = M + e\sin M$ it typically hits machine precision in 3–5 iterations, even at $e = 0.9$.

Finally, the true anomaly from the eccentric anomaly:

$$\nu = 2\,\operatorname{atan2}\!\left(\sqrt{1+e}\,\sin\tfrac{E}{2},\; \sqrt{1-e}\,\cos\tfrac{E}{2}\right)$$

(`atan2` rather than `arctan` so the quadrant is always right.)

### The radial-velocity equation

Put the observer far away along some axis $\hat{z}$. The star's position along that axis is

$$z = r_\star \sin(\nu + \omega)\sin i$$

where $\omega$ is the **argument of periastron** (the angle from the sky plane to the star's periastron, measured in the orbital plane) and $i$ is the **inclination** ($90°$ = edge-on). Differentiating with respect to time, and using conservation of angular momentum ($r^2\dot\nu = \text{const}$) to eliminate $\dot{r}$ and $\dot{\nu}$, gives the classic result:

$$\boxed{v_r(t) = K\left[\cos(\nu + \omega) + e\cos\omega\right] + \gamma}$$

with the **semi-amplitude**

$$K = \left(\frac{2\pi G}{P}\right)^{1/3} \frac{m \sin i}{(M+m)^{2/3}} \frac{1}{\sqrt{1-e^2}}$$

A few things are worth noticing:

- **$\gamma$** is the whole system's velocity toward or away from us. Every star has one, and it's usually far bigger than $K$.
- **$m$ and $\sin i$ only ever appear as the product $m\sin i$.** An edge-on light planet and a tilted heavy one produce *identical* RV curves. RV alone gives only a **minimum mass**. (Transits or astrometry are needed to break the degeneracy.)
- **Peak-to-peak is exactly $2K$**, whatever $e$ and $\omega$ are. That's a handy check.
- **$\omega$ belongs to the star's orbit.** The star sits opposite the planet, so $\omega_\star = \omega_\text{planet} + \pi$. Getting this wrong flips the RV curve. It's the most common bug in RV code, and TODO 4 makes you think about it.
- For $e = 0$: $v_r = K\cos(\nu + \omega) + \gamma$, a pure sinusoid. That's your original model!

### Why the Keplerian must match your N-body simulation

For **one** planet, the Keplerian orbit is the *exact* solution of Newton's equations. So the curve from `keplerian_rv` and the star's velocity from `leapfrog_step` must agree. The only difference allowed is the integrator's error.

Leapfrog conserves energy beautifully (your `leapfrog_integration_lesson.md`), but it isn't perfect. Its orbital **period** is off by a tiny amount $\propto \Delta t^2$, so the simulated planet slowly drifts ahead of or behind the exact solution, and the mismatch grows linearly with time. Halve $\Delta t$ and the drift drops 4×. The notebook invites you to verify this yourself.

With **two or more** planets, the Keplerian stops being exact: the planets tug on each other, and their orbits precess. A sum of fixed Keplerians can't capture that, but your N-body code can. That's one of the stretch goals.

---

## Part 2: Statistics for a 7-Parameter Posterior

### Choosing *what* to sample

The parameters we sample don't have to be the "natural" physical ones, and a smart choice makes the sampler's job far easier. We sample

$$\theta = \left(\ln P,\; \ln K,\; \sqrt{e}\cos\omega,\; \sqrt{e}\sin\omega,\; M_0,\; \gamma,\; \ln s\right)$$

**Why logs for $P$, $K$ and $s$?** These are *scale* parameters: positive, and a priori we're equally unsure whether $P$ is 3 days or 300. A prior uniform in $\ln P$ spreads belief evenly across orders of magnitude (this is the *Jeffreys prior* for a scale). Sampling in log space also turns multiplicative uncertainty into additive uncertainty, which makes the posterior more Gaussian.

**Why $h = \sqrt{e}\cos\omega$ and $k = \sqrt{e}\sin\omega$?** This fixes two problems at once:

1. **$\omega$ is undefined when $e = 0$.** A circular orbit has no periastron. For nearly circular orbits, $\omega$ swings wildly while the likelihood hardly changes, and the posterior becomes a nasty thin ring. In $(h, k)$ coordinates, the circle $e = 0$ is just the single point $(0,0)$, and nothing blows up.
2. **The square root makes a uniform prior come out right.** $(h, k)$ are polar coordinates with radius $\rho = \sqrt{e}$ and angle $\omega$. The area element is $dh\,dk = \rho\,d\rho\,d\omega$. Since $\rho\,d\rho = \sqrt{e}\cdot\frac{de}{2\sqrt{e}} = \frac{1}{2}de$,
$$dh\,dk = \tfrac{1}{2}\,de\,d\omega$$
so a flat prior over the unit disk in $(h,k)$ is **exactly** a flat prior in $e \in [0,1)$ and $\omega \in [0, 2\pi)$. If we'd used $e\cos\omega$ instead, the implied prior would be $\propto e$, quietly biasing results toward high eccentricity.

**Why $M_0$ at a reference time $t_\text{ref}$ in the middle of the data?** The phase at time $t$ is $\frac{2\pi}{P}(t - t_\text{ref}) + M_0$. If $t_\text{ref}$ sits far from the data, a small change in $P$ must be compensated by a big change in $M_0$: a strong $P$–$M_0$ correlation. Anchoring the phase mid-baseline mostly removes it.

### Jitter, and why the normalization term suddenly matters

Stars aren't perfectly stable. Convection, spots and pulsations add a few m/s of "noise" that the instrument's error bars $\sigma_i$ don't include. We model it as an extra Gaussian term with unknown standard deviation $s$ (the **jitter**) added in quadrature:

$$\ln\mathcal{L} = -\frac{1}{2}\sum_i\left[\frac{(v_i - m_i)^2}{\sigma_i^2 + s^2} + \ln\big(2\pi(\sigma_i^2 + s^2)\big)\right]$$

In your first MCMC you dropped the $\ln(2\pi\sigma^2)$ term, which was fine because $\sigma$ was fixed and the term was a constant. **Now it isn't.** Without it, the likelihood would *always* prefer $s \to \infty$ (a huge variance makes every residual look tiny). The log term is an **Occam penalty**: it charges you for claiming the data are noisier than they are. The two terms balance where $s$ matches the actual excess scatter.

### Priors

All seven priors are uniform ("flat") inside a box, and $-\infty$ outside:

$$\ln p(\theta \mid \text{data}) = \underbrace{\ln \pi(\theta)}_{0 \text{ or } -\infty} + \ln\mathcal{L}(\theta) + \text{const}$$

Two practical rules:

- **Check the prior first, and bail out early.** If $e \ge 1$, Kepler's equation has no elliptical solution and `solve_kepler` produces garbage or NaNs. Never evaluate the likelihood outside the prior.
- **Bounds on angles:** $M_0 \in [0, 2\pi)$ is a hard wall, even though the physics is periodic. It's fine as long as the posterior isn't sitting right on the wall. (Choosing $t_\text{ref}$ sensibly helps here too.)

### Finding the period: periodograms and aliases

The posterior over $P$ is a **forest of narrow spikes**. Every period that makes the data line up tolerably is a local maximum, and the spikes are far narrower than the gaps between them. No random walk will ever hop between them. So we find the right neighbourhood first, with a **periodogram**: for every trial period $P$, fit $a + b\sin(2\pi t/P) + c\cos(2\pi t/P)$ by linear least squares (that's fast, since it's linear in $a, b, c$) and record how much $\chi^2$ drops.

The periodogram shows **aliases**, false peaks created by the *observing schedule* rather than the star. Our star is only observable part of the year, so the sampling itself has a 1-year periodicity. A signal at frequency $f$ observed with a window that repeats at frequency $f_w$ leaks power into $f \pm f_w$:

$$\frac{1}{P_\text{alias}} = \frac{1}{P} \pm \frac{1}{365.25\text{ d}}$$

For $P = 111.4$ d that predicts aliases near **85 d** and **160 d**. Look for them in the notebook. (Ground-based telescopes also have a **1-day alias** because they can only observe at night. It's the reason some famous "planets" turned out to be the wrong period.)

### Optimize, *then* sample

From the periodogram peak, an optimizer climbs to the **MAP** (maximum a posteriori) point, and walkers start in a tiny ball around it. Even the optimizer can get stuck: a nearly-circular orbit with a smaller $K$ is a common trap. So the notebook launches several optimizations with different $\omega$ and keeps the best. Remember the MAP point is **not** the answer. It carries no uncertainty. The answer is the whole posterior.

---

## Part 3: Affine-Invariant Ensemble Sampling

### What's wrong with random-walk MH here?

Your MH sampler proposed $x' = x + \epsilon$ with $\epsilon \sim \mathcal{N}(0, \text{step}^2)$. In 7-D this breaks for two reasons:

1. **Wildly different scales.** In our posterior, $\ln P$ is pinned to about $\pm 3\times10^{-4}$ while $\gamma$ spreads over $\pm 0.5$ m/s. A step size that suits one is useless for the other: either every proposal is rejected, or the walker crawls.
2. **Correlations.** When two parameters trade off (a diagonal ridge in the posterior), axis-aligned steps can only zig-zag slowly along it.

The theoretical fix is to propose with the *posterior's own covariance*, $\epsilon \sim \mathcal{N}(0, \tfrac{2.38^2}{d}\Sigma)$. But $\Sigma$ is what we're trying to find! The notebook's Part 8 shows exactly this: untuned MH gets stuck, while MH given $\Sigma$ ("cheating") is excellent.

### Affine invariance

Suppose you stretch, squash, rotate and shift the parameter space: $Y = AX + b$ for any invertible matrix $A$. A long, thin, tilted posterior becomes a round blob (or vice versa). An **affine-invariant** sampler behaves *identically* in both coordinate systems. Its efficiency doesn't care about scales or linear correlations at all. That's exactly the robustness we're missing.

### The stretch move

Goodman & Weare (2010) achieve this with an **ensemble** of $n$ walkers $\{X_1, \dots, X_n\}$ exploring simultaneously. To update walker $X_k$:

1. Pick a random **other** walker $X_j$.
2. Draw a stretch factor $z \in [1/a, a]$ from $g(z) \propto 1/\sqrt{z}$ (usually $a = 2$).
3. Propose a point on the line through both walkers:
$$Y = X_j + z\,(X_k - X_j)$$
   For $z > 1$ this stretches *away* from $X_j$; for $z < 1$ it contracts toward it.
4. Accept with probability
$$\min\left(1,\; z^{\,d-1}\,\frac{p(Y)}{p(X_k)}\right)$$

Why is this affine invariant? The proposal is built only from *differences between walkers*. If the whole ensemble is squashed by $A$, the differences are squashed identically and the proposals transform exactly like the target. The walkers' spread *is* the step-size matrix, and it adapts automatically.

### Where does $z^{d-1}$ come from? (Detailed balance, again!)

In your last lesson you showed Metropolis-Hastings is unbiased because it satisfies **detailed balance**:

$$\pi(x)\,T(x \to x') = \pi(x')\,T(x' \to x)$$

For a **symmetric** proposal ($g(x\to x') = g(x' \to x)$) the acceptance ratio was just $\pi(x')/\pi(x)$. The stretch move is **not** symmetric, for two reasons:

- **The $z$ distribution.** Going from $X_k$ to $Y$ uses $z$; going back from $Y$ to $X_k$ (around the same $X_j$) needs $1/z$. The choice $g(z) \propto 1/\sqrt{z}$ is special because it satisfies $g(1/z) = z\,g(z)$, which exactly cancels one factor of $z$.
- **Geometry in $d$ dimensions.** The move is along a 1-D line, but it lives in $d$-dimensional space. Scaling the distance from $X_j$ by $z$ maps a thin shell of radius $\rho$ onto a shell of radius $z\rho$, and shell areas grow like (radius)$^{d-1}$. Outward moves land in "more room" than inward moves come from.

Putting these into detailed balance, the proposal densities fail to cancel by precisely $z^{d-1}$, so that factor must go in the acceptance ratio to restore balance. (For the full derivation see Goodman & Weare 2010, Section 3.) Leave it out and the sampler still runs, but it samples the **wrong distribution**. Check 6 in the notebook will catch you.

### Serial updates and the complementary ensemble

Walkers are updated **one at a time**, and each update uses the *current* positions of the others, including walkers already moved in the same sweep. This is what the proof requires. (Stretch goal 5 shows how to split the ensemble in two halves and update each half in parallel, which is how `emcee` actually does it.)

Rules of thumb:

- Use at least $2d$ walkers; $4d$–$5d$ is comfortable. (We use 32 for $d = 7$.)
- A healthy **acceptance fraction** is roughly 0.2–0.5. Near 0: walkers are stuck. Near 1: something's wrong, e.g. the posterior is flat because of a bug.

---

## Part 4: Has It Converged?

### Autocorrelation time

Each MCMC sample is correlated with the ones before it. The **normalized autocorrelation function** measures how similar the chain is to itself $\ell$ steps later:

$$\rho(\ell) = \frac{\operatorname{Cov}(x_t,\, x_{t+\ell})}{\operatorname{Var}(x_t)}$$

The **integrated autocorrelation time** adds it all up:

$$\tau = 1 + 2\sum_{\ell=1}^{\infty}\rho(\ell)$$

Interpretation: **it takes about $\tau$ steps to produce one independent sample.** So

$$N_\text{eff} \approx \frac{N_\text{steps} \times N_\text{walkers}}{\tau}$$

Example: $\tau = 80$ with 10,000 steps × 32 walkers gives $N_\text{eff} \approx 4000$, not 320,000.

### Estimating $\tau$ without fooling yourself

Computing $\rho(\ell)$ directly is $O(N^2)$. The **Wiener–Khinchin theorem** says autocorrelation is the inverse Fourier transform of the power spectrum, so an FFT gets it in $O(N\log N)$ (provided in the notebook).

The subtle part is the sum. At large lags, $\rho(\ell)$ is pure noise, and summing thousands of noisy terms makes $\hat\tau$ wander randomly. **Sokal's adaptive window** stops the sum at the first lag $M$ satisfying

$$M \ge c\,\hat\tau(M), \qquad c \approx 5$$

The window grows with $\tau$ itself: far enough to capture the real correlation, not so far that noise dominates.

For an ensemble, **average $\rho(\ell)$ over walkers first**, then apply the window. That's much less noisy than computing $\tau$ per walker and averaging.

### Practical checklist

- **Run length:** $N_\text{steps} \gtrsim 50\tau$. Below that, even the estimate of $\tau$ isn't trustworthy.
- **Burn-in:** discard the first few $\tau$ (the notebook uses $3\tau$), while the ensemble spreads out from its starting ball.
- **Thinning:** keeping every $\sim\tau/2$-th step loses almost no information and makes the sample set manageable.
- **Look at the trace plots.** A walker stuck far from the others, or a slow trend, means you're not done, whatever the numbers say.

A trick for testing $\tau$: the AR(1) process $x_{n+1} = \phi x_n + \epsilon_n$ has $\rho(\ell) = \phi^\ell$ exactly, and summing the geometric series gives $\tau = \frac{1+\phi}{1-\phi}$. Check 7 uses this.

---

## Part 5: From $K$ to the Planet's Mass

Rearranging the $K$ equation gives the **mass function**:

$$\frac{(m\sin i)^3}{(M + m)^2} = \frac{P K^3 (1-e^2)^{3/2}}{2\pi G}$$

It's a cubic in $m$. Because $m \ll M$, **fixed-point iteration** converges in a few steps:

$$m_{n+1} = K\sqrt{1-e^2}\left(\frac{P}{2\pi G}\right)^{1/3}(M + m_n)^{2/3}, \qquad m_0 = 0$$

The first iteration ($m_1$, using $M + m \approx M$) is already the "textbook" approximation; later iterations fix it up.

The beauty of having posterior **samples**: to get the posterior of *any* derived quantity, just compute it for every sample. Uncertainty propagation, including all correlations between $K$, $P$ and $e$, comes for free, with no error-propagation formulas needed.

---

## Your Challenge

Open **`machine_learning/keplerian_rv_fit.ipynb`**. It has 8 ✏️ TODOs, each followed by a ✅ check cell:

| TODO | What you build | Lesson section |
|---|---|---|
| 1 | Two-body initial conditions at periastron | *Setting up the two-body problem* |
| 2 | Vectorized Newton-Raphson Kepler solver | *Three anomalies* |
| 3 | The Keplerian RV model | *The radial-velocity equation* |
| 4 | $K$, $\omega_\star$ and $M_0$ for the true orbit | *The radial-velocity equation* |
| 5 | Log-prior, log-likelihood with jitter, log-posterior | *Part 2* |
| 6 | The Goodman & Weare stretch-move sampler | *Part 3* |
| 7 | Integrated autocorrelation time with Sokal windowing | *Part 4* |
| 8 | Minimum mass from posterior samples | *Part 5* |

The payoff comes after Check 4, when your analytic model lands on top of your N-body simulation, and at the end, when you recover the planet's mass from noisy data and it matches what you put in.

## Further Reading

- J. Goodman & J. Weare (2010), *Ensemble samplers with affine invariance*, Comm. App. Math. Comp. Sci. 5, 65. The stretch move.
- D. Foreman-Mackey et al. (2013), *emcee: The MCMC Hammer*, PASP 125, 306. A very readable practical guide, including autocorrelation-time advice.
- A. Sokal (1997), *Monte Carlo Methods in Statistical Mechanics: Foundations and New Algorithms*. The adaptive window (and more stat-mech ↔ MCMC connections).
- M. Perryman, *The Exoplanet Handbook*, Chapter 2. Everything about radial velocities.
- M. Mayor & D. Queloz (1995), *A Jupiter-mass companion to a solar-type star*, Nature 378, 355. 51 Peg b, the discovery that started it all.
