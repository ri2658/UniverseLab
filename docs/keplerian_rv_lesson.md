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

A star (mass $M_\star$) and a planet (mass $m$) both orbit their shared **centre of mass**. (We write $M_\star$ rather than $M$, because $M$ is about to mean the *mean anomaly*. The notebook's TODO 1 uses plain $M$ for the star.) The standard trick is to split the motion into two pieces:

1. The **centre of mass** $\vec{R} = (M_\star\vec{r}_\text{star} + m\,\vec{r}_\text{planet})/(M_\star + m)$ feels no net external force, so it moves in a straight line. We put it at rest at the origin.
2. The **relative vector** $\vec{r} = \vec{r}_\text{planet} - \vec{r}_\text{star}$ obeys a one-body problem. Subtracting the two equations of motion gives $\ddot{\vec r} = -G(M_\star + m)\,\vec{r}/r^3$: an orbit around a fixed mass $M_\star + m$.

Setting $\vec{R} = 0$ and solving the two definitions for the positions, each body's position is a fixed fraction of $\vec{r}$:

$$\vec{r}_\text{planet} = +\frac{M_\star}{M_\star+m}\,\vec{r}, \qquad \vec{r}_\text{star} = -\frac{m}{M_\star+m}\,\vec{r}$$

Differentiating gives the same split for the velocities, and then the total momentum $M_\star\dot{\vec r}_\text{star} + m\,\dot{\vec r}_\text{planet}$ is exactly zero. The star traces a **tiny copy** of the planet's orbit, scaled down by $m/M_\star$ (about 1/500 for a 2 Jupiter-mass planet) and flipped to the opposite side.

To launch an eccentric orbit you need two textbook results for the relative orbit:

- **Kepler's third law** fixes the semi-major axis from the period: $a^3 = \dfrac{G(M_\star+m)P^2}{4\pi^2}$
- **Vis-viva** gives the speed at any distance: $v^2 = G(M_\star+m)\left(\dfrac{2}{r} - \dfrac{1}{a}\right)$

At periastron (closest approach), $r_p = a(1-e)$. Plugging that into vis-viva gives

$$v_p^2 = \frac{G(M_\star+m)}{a}\left(\frac{2}{1-e} - 1\right) = \frac{G(M_\star+m)}{a}\cdot\frac{1+e}{1-e}$$

Periastron is a minimum of $r$, so $\dot r = 0$ there, and the velocity is perpendicular to $\vec{r}$. That's everything you need for **TODO 1**.

### Three anomalies: where is the planet at time $t$?

For a circle, the angle grows uniformly: $\theta = 2\pi t/P$. For an ellipse it doesn't. Kepler's second law says the planet sweeps equal *areas* in equal times, so it races through periastron and dawdles at apoastron. Astronomers describe the position with three angles (for historical reasons called "anomalies"):

| Anomaly | Symbol | Meaning |
|---|---|---|
| **Mean** | $M$ | A fictitious angle that *does* grow uniformly: $M = \frac{2\pi}{P}(t - t_p)$. It's really just "fraction of the orbit elapsed" in radians. |
| **Eccentric** | $E$ | A geometric helper angle. Draw the circle of radius $a$ centred on the ellipse's *centre*, project the planet perpendicular to the major axis onto that circle, and measure the angle of that point from the centre. |
| **True** | $\nu$ | The actual angle between the planet and periastron, as seen from the focus. This is the physical one. |

In terms of $E$, with the focus at the origin and periastron along $+x$, the planet sits at

$$x = a(\cos E - e), \qquad y = a\sqrt{1-e^2}\,\sin E, \qquad r = a(1 - e\cos E)$$

Mean and eccentric anomaly are linked by **Kepler's equation**:

$$M = E - e \sin E$$

**Where it comes from.** Kepler's second law says the area swept out from the focus grows linearly in time. Since the whole ellipse (area $\pi a b$, with $b = a\sqrt{1-e^2}$) is swept in one period, the area swept since periastron is $\pi a b \cdot \frac{t - t_p}{P} = \tfrac{1}{2}ab\,M$. Now compute the same area geometrically. Squashing the auxiliary circle by $b/a$ turns it into the ellipse. On the circle, the region swept from the focus is a circular sector from the centre (area $\tfrac12 a^2 E$) minus a triangle with base $ae$ (the centre-to-focus distance) and height $a\sin E$ (area $\tfrac12 a^2 e\sin E$). Squashing multiplies every area by $b/a$, so the ellipse's area is $\tfrac12 ab\,(E - e\sin E)$. Setting the two areas equal gives Kepler's equation.

Going from $E$ to $M$ is trivial. We need the opposite direction ($M$ is what we know from the time), and that has **no closed-form solution** in elementary functions. This one equation launched centuries of numerical methods; Newton himself worked on it.

First, a solution always exists and is unique. $f(E) = E - e\sin E - M$ has derivative $f'(E) = 1 - e\cos E \ge 1 - e > 0$ for $e < 1$, so $f$ is strictly increasing and runs from $-\infty$ to $+\infty$. It crosses zero exactly once. We find that crossing with **Newton-Raphson**: repeat

$$E_{n+1} = E_n - \frac{f(E_n)}{f'(E_n)} = E_n - \frac{E_n - e\sin E_n - M}{1 - e\cos E_n}$$

Near the root, Newton-Raphson converges *quadratically*: the error obeys $\varepsilon_{n+1} \approx \frac{f''}{2f'}\,\varepsilon_n^2$, so the number of correct digits roughly doubles each iteration. From the starting guess $E_0 = M + e\sin M$ it typically reaches machine precision in 4–5 iterations. The slow case is high $e$ near periastron ($M \approx 0$), where $f' \approx 1 - e$ is small and the first steps overshoot. It still converges, but the worst case grows to about 7 iterations at $e = 0.9$ and 10 at $e = 0.99$. That's why the solver stops on a tolerance, not after a fixed number of iterations.

Finally, the true anomaly. Using $\tan\frac{\nu}{2} = \sqrt{\frac{1+e}{1-e}}\,\tan\frac{E}{2}$ (which follows from the $x, y$ above), write it with both signs kept:

$$\nu = 2\,\operatorname{atan2}\!\left(\sqrt{1+e}\,\sin\tfrac{E}{2},\; \sqrt{1-e}\,\cos\tfrac{E}{2}\right)$$

(`atan2` rather than `arctan` so the quadrant is always right. $E/2$ and $\nu/2$ always lie in the same half-turn, so doubling the result recovers the correct $\nu$.)

### The radial-velocity equation

Put the observer far away and let $\hat{z}$ point **away** from them, so that positive $v_r$ means receding (redshift). The orbital plane is tilted by the **inclination** $i$ relative to the sky plane ($i = 90°$ is edge-on). It crosses the sky plane along the *line of nodes*. The **ascending node** is the crossing where the star moves away from us, and the **argument of periastron** $\omega$ is the angle, measured in the orbital plane in the direction of motion, from the ascending node to the star's periastron. The star's angle from the node is then $\nu + \omega$, and its distance along the line of sight is

$$z = r_\star \sin(\nu + \omega)\sin i$$

Here $r_\star$ is the star's distance from the centre of mass. It traces an ellipse with the same $e$, $P$ and $\nu$ as the relative orbit, but with semi-major axis $a_\star = \frac{m}{M_\star+m}\,a$.

**Differentiating.** With $i$ and $\omega$ constant,

$$\dot z = \sin i\left[\dot r_\star\sin(\nu+\omega) + r_\star\dot\nu\cos(\nu+\omega)\right]$$

We need $\dot r_\star$ and $r_\star\dot\nu$. Two facts about Kepler orbits supply them. The orbit equation is $r_\star = \dfrac{p}{1 + e\cos\nu}$, with *semi-latus rectum* $p = a_\star(1-e^2)$. Conservation of angular momentum gives $r_\star^2\dot\nu = h$, a constant. Then

$$r_\star\dot\nu = \frac{h}{r_\star} = \frac{h}{p}(1 + e\cos\nu), \qquad \dot r_\star = \frac{p\,e\sin\nu\,\dot\nu}{(1+e\cos\nu)^2} = \frac{h}{p}\,e\sin\nu$$

Substitute these, and use $\cos\nu\cos(\nu+\omega) + \sin\nu\sin(\nu+\omega) = \cos\omega$:

$$\dot z = \frac{h\sin i}{p}\left[\cos(\nu+\omega) + e\left(\cos\nu\cos(\nu+\omega) + \sin\nu\sin(\nu+\omega)\right)\right] = \frac{h\sin i}{p}\left[\cos(\nu+\omega) + e\cos\omega\right]$$

Adding the centre of mass's own velocity $\gamma$ gives the classic result:

$$\boxed{v_r(t) = K\left[\cos(\nu + \omega) + e\cos\omega\right] + \gamma}, \qquad K = \frac{h\sin i}{p}$$

**The semi-amplitude.** The swept-area rate is $h/2$, so a whole ellipse takes $P = 2\pi a_\star b_\star/h$, i.e. $h = 2\pi a_\star^2\sqrt{1-e^2}/P$. Therefore

$$K = \frac{2\pi a_\star\sin i}{P\sqrt{1-e^2}}$$

Now eliminate $a_\star = \frac{m}{M_\star+m}a$ using Kepler's third law, $a = \left(G(M_\star+m)P^2/4\pi^2\right)^{1/3}$:

$$K = \left(\frac{2\pi G}{P}\right)^{1/3} \frac{m \sin i}{(M_\star+m)^{2/3}} \frac{1}{\sqrt{1-e^2}}$$

A few things are worth noticing:

- **$\gamma$** is the whole system's velocity toward or away from us. Every star has one, and it's usually far bigger than $K$.
- **$m$ and $\sin i$ only ever appear as the product $m\sin i$.** An edge-on light planet and a tilted heavy one produce *identical* RV curves. RV alone gives only a **minimum mass**. (Transits or astrometry are needed to break the degeneracy.)
- **Peak-to-peak is exactly $2K$**, whatever $e$ and $\omega$ are. That's a handy check.
- **$\omega$ belongs to the star's orbit.** The star sits opposite the planet, so $\omega_\star = \omega_\text{planet} + \pi$. Getting this wrong flips the RV curve. It's the most common bug in RV code, and TODO 4 makes you think about it.
- For $e = 0$: $E = M = \nu$, so $v_r = K\cos(M + \omega) + \gamma$, a pure sinusoid in time. That's your original model, with a phase and an offset added. (For a circle $\omega$ isn't defined on its own; only the combination $M_0 + \omega$ matters. This hints at the $M_0$–$\omega$ correlation you'll see in the corner plot.)

### Why the Keplerian must match your N-body simulation

For **one** planet, the Keplerian orbit is the *exact* solution of Newton's equations. So the curve from `keplerian_rv` and the star's velocity from `leapfrog_step` must agree. The only difference allowed is the integrator's error.

Leapfrog conserves energy beautifully (your `leapfrog_integration_lesson.md`), but it isn't perfect. Its orbital **period** is off by a tiny amount $\propto \Delta t^2$, so the simulated planet slowly drifts ahead of or behind the exact solution, and the mismatch grows linearly with time. Halve $\Delta t$ and the drift drops 4×. The notebook invites you to verify this yourself.

With **two or more** planets, the Keplerian stops being exact: the planets tug on each other, and their orbits precess. A sum of fixed Keplerians can't capture that, but your N-body code can. That's one of the stretch goals.

---

## Part 2: Statistics for a 7-Parameter Posterior

### Choosing *what* to sample

The parameters we sample don't have to be the "natural" physical ones, and a smart choice makes the sampler's job far easier. We sample

$$\theta = \left(\ln P,\; \ln K,\; \sqrt{e}\cos\omega,\; \sqrt{e}\sin\omega,\; M_0,\; \gamma,\; \ln s\right)$$

**Why logs for $P$, $K$ and $s$?** These are *scale* parameters: positive, and a priori we're equally unsure whether $P$ is 3 days or 300. A prior uniform in $\ln P$ spreads belief evenly across orders of magnitude. By the change-of-variables rule $p(P) = p(\ln P)\left|\frac{d\ln P}{dP}\right| \propto 1/P$, so every factor-of-10 interval gets equal probability (this is the *Jeffreys prior* for a scale parameter, and the only prior that doesn't change when you switch units). Sampling in log space also turns multiplicative uncertainty into additive uncertainty, which makes the posterior more Gaussian.

**Why $h = \sqrt{e}\cos\omega$ and $k = \sqrt{e}\sin\omega$?** This fixes two problems at once:

1. **$\omega$ is undefined when $e = 0$.** A circular orbit has no periastron. For nearly circular orbits, $\omega$ swings wildly while the likelihood hardly changes, and the posterior becomes a nasty thin ring. In $(h, k)$ coordinates, the circle $e = 0$ is just the single point $(0,0)$, and nothing blows up.
2. **The square root makes a uniform prior come out right.** $(h, k)$ are polar coordinates with radius $\rho = \sqrt{e}$ and angle $\omega$. The area element is $dh\,dk = \rho\,d\rho\,d\omega$. Since $\rho\,d\rho = \sqrt{e}\cdot\frac{de}{2\sqrt{e}} = \frac{1}{2}de$,
$$dh\,dk = \tfrac{1}{2}\,de\,d\omega$$
A density transforms with the Jacobian, $p(e,\omega) = p(h,k)\left|\frac{\partial(h,k)}{\partial(e,\omega)}\right| = \tfrac12\,p(h,k)$. So a flat prior over the unit disk in $(h,k)$ is **exactly** a flat prior in $e \in [0,1)$ and $\omega \in [0, 2\pi)$. If we'd used $(e\cos\omega, e\sin\omega)$ instead, the area element would be $e\,de\,d\omega$ and the implied prior would be $p(e) \propto e$, quietly biasing results toward high eccentricity.

**Why $M_0$ at a reference time $t_\text{ref}$ in the middle of the data?** The mean anomaly at time $t$ is $M(t) = \frac{2\pi}{P}(t - t_\text{ref}) + M_0$. The data pin down the phase near the *middle* of the observations, $t_c$. Holding $M(t_c)$ fixed while changing $P$ by $\delta P$ requires
$$\delta M_0 = \frac{2\pi\,(t_c - t_\text{ref})}{P^2}\,\delta P$$
If $t_\text{ref}$ sits far from the data, a tiny change in $P$ must be compensated by a big change in $M_0$: a strong $P$–$M_0$ correlation. With $t_\text{ref} \approx t_c$ the coefficient is roughly zero, which mostly removes it.

### Jitter, and why the normalization term suddenly matters

Stars aren't perfectly stable. Convection, spots and pulsations add a few m/s of "noise" that the instrument's error bars $\sigma_i$ don't include. We model it as an extra, independent Gaussian noise term with unknown standard deviation $s$ (the **jitter**). Variances of independent Gaussians add, so each point's total variance is $\sigma_i^2 + s^2$ (the errors add "in quadrature"). The log of the product of Gaussians is

$$\ln\mathcal{L} = -\frac{1}{2}\sum_i\left[\frac{(v_i - \mu_i)^2}{\sigma_i^2 + s^2} + \ln\big(2\pi(\sigma_i^2 + s^2)\big)\right]$$

where $\mu_i$ is the model's prediction at time $t_i$. (We use $\mu$ rather than $m$ to avoid a clash with the planet mass.)

In your first MCMC you dropped the $\ln(2\pi\sigma^2)$ term, which was fine because $\sigma$ was fixed and the term was a constant. **Now it isn't.** Without it, $\ln\mathcal L$ would increase monotonically toward 0 as $s \to \infty$, so the fit would always prefer infinite jitter (a huge variance makes every residual look tiny). The log term is an **Occam penalty**: it charges you for claiming the data are noisier than they are.

You can see exactly where the balance falls. Write $V_i = \sigma_i^2 + s^2$ and $r_i = v_i - \mu_i$, and differentiate with respect to $s^2$:

$$\frac{\partial \ln\mathcal L}{\partial (s^2)} = \frac12\sum_i \frac{r_i^2 - V_i}{V_i^2}$$

This is zero when the squared residuals match their predicted variance *on average* (weighted by $1/V_i^2$). If all the $\sigma_i$ were equal, it would give exactly $s^2 = \overline{r^2} - \sigma^2$: the excess scatter. If the residuals scatter *less* than the error bars predict, the derivative is negative for every $s$, and the likelihood peaks at $s = 0$.

### Priors

All seven priors are uniform ("flat") in the sampled coordinates $\theta$, inside a box (plus the disk $h^2 + k^2 < 1$), and zero outside. A flat prior's density is a constant inside its support, and that constant just joins the other normalizing constants, so we can take its log to be $0$ inside and $-\infty$ outside:

$$\ln p(\theta \mid \text{data}) = \underbrace{\ln p(\theta)}_{0 \text{ or } -\infty} + \ln\mathcal{L}(\theta) + \text{const}$$

Two practical rules:

- **Check the prior first, and bail out early.** At $e \ge 1$ the orbit is no longer an ellipse. $f'(E) = 1 - e\cos E$ can now reach zero, so Newton's method can divide by zero, and $\sqrt{1-e}$ in the true-anomaly formula becomes NaN. Never evaluate the likelihood outside the prior.
- **Bounds on angles:** $M_0 \in [0, 2\pi)$ is a hard wall, even though the physics is periodic. It's fine as long as the posterior isn't sitting right on the wall. (Choosing $t_\text{ref}$ sensibly helps here too.)

### Finding the period: periodograms and aliases

The posterior over $P$ is a **forest of narrow spikes**. Every period that makes the data line up tolerably is a local maximum. Over a baseline $T$, two trial frequencies that differ by $\delta f$ drift apart in phase by $2\pi\,\delta f\,T$, so each spike has a width of about $\delta f \sim 1/T$, or $\delta P \sim P^2/T$ in period (about 11 days for $P = 111$ d and $T = 1100$ d, and much narrower at the *posterior* level, where the noise is small compared with $K$). The spikes are far narrower than the gaps between them, and no random walk will ever hop between them. So we find the right neighbourhood first, with a **periodogram**: for every trial period $P$, fit $a + b\sin(2\pi t/P) + c\cos(2\pi t/P)$ by weighted linear least squares (that's fast, since it's linear in $a, b, c$) and record the fractional drop in $\chi^2$. (This is the *generalized Lomb–Scargle* periodogram. It's the best-fit $e = 0$ Keplerian at each period, which is why it works well for moderate $e$.)

The periodogram shows **aliases**, false peaks created by the *observing schedule* rather than the star. Our star is only observable part of the year, so the sampling pattern (the *window function*) itself has a 1-year periodicity. Sampling multiplies the signal by the window, and multiplication in time is convolution in frequency. So the observed spectrum is the true spectrum convolved with the window's spectrum, which has a peak at $f_w = 1/\text{yr}$. A signal at frequency $f$ therefore leaks power into $f \pm f_w$ (and, since real signals have power at $-f$ too, into $|f - f_w|$):

$$\frac{1}{P_\text{alias}} = \left|\frac{1}{P} \pm \frac{1}{365.25\text{ d}}\right|$$

For $P = 111.4$ d that predicts aliases near **85 d** and **160 d**. Look for them in the notebook. (Ground-based telescopes also have a **1-day alias**, because they can only observe at night. The next subsection explains where aliases come from at a deeper level and why they are dangerous.)

### Nyquist, aliases, and how much we can observe

**Evenly spaced samples have a hard limit.** Suppose we observe at $t_k = k\Delta$. A sinusoid of frequency $f$ gives the samples $\sin(2\pi f k\Delta + \varphi)$, and for any integer $m$

$$\sin\!\big(2\pi (f - m/\Delta)\,k\Delta + \varphi\big) = \sin\!\big(2\pi f k\Delta + \varphi - 2\pi m k\big) = \sin(2\pi f k\Delta + \varphi).$$

So $f$ and $f - m/\Delta$ produce *identical* data. Only a band of width $1/\Delta$ can be told apart, conventionally $|f| \le f_N = 1/(2\Delta)$, the **Nyquist frequency**. Everything above it is folded back into that band. (The sampling theorem is the converse: a signal with nothing above $f_N$ is completely determined by its samples.)

Here is what that means for us. Fifty observations spread *evenly* over our 1095-day baseline are 21.9 d apart, so the Nyquist limit is a period of 43.8 d. A 20-day planet ($f = 0.0500\ \text{d}^{-1}$) then looks exactly like a signal at $f - 1/\Delta = 0.0043\ \text{d}^{-1}$, a 231-day period. The two periodogram peaks tie, so the data cannot choose. (In a quick simulation with $K = 50$ m/s and $\sigma = 10$ m/s, both peaks had power 0.95.)

**Irregular times break the tie.** With irregular times there is no single $\Delta$ that makes $f$ and $f - m/\Delta$ coincide at every sample, so the aliases stop matching the data and sink into a low noise floor. The same simulation with 50 *random* times gave power 0.94 at the true 20 d and 0.01 at 231 d. So for uneven sampling there is no hard Nyquist limit at the *average* spacing. (Roughly speaking, the smallest spacings set the limit instead.) That is why a Lomb–Scargle periodogram stays useful at frequencies far above $N/2T$.

**Real schedules are irregular, but not random.** Telescopes work at night, and stars are only up for part of the year, so the window has *structure*. That structure is what puts the sharp aliases at $1/P \pm 1/\text{yr}$ and $1/P \pm 1/\text{day}$. For inference this has a nasty consequence: each alias is a separate narrow mode of the posterior, and an ensemble started near one of them stays there. Start at, or converge to, the wrong alias and you report a confident wrong period. This has happened to real planets. 55 Cancri e was first reported with $P \approx 2.8$ d and later shown to have $P = 0.7365$ d (Dawson & Fabrycky 2010), which a later transit detection confirmed. With one observation per day, a 0.7365 d signal is aliased to $1/|1/0.7365 - 1| = 2.795$ d.

**What limits the number of observations?** The limit is rarely the raw count. It is *when* and *how well* we can observe:

- **The Earth.** Night only (the 1-day alias), seasonal visibility (the yearly alias), weather.
- **Photons.** A point's precision comes from the light collected, $\sigma_i^2 \propto 1/t_i$ for an exposure of length $t_i$. So $\sum_i 1/\sigma_i^2 \propto \sum_i t_i$: for white noise, the information depends on the *total* exposure time, and splitting it into many short exposures gains nothing. Overheads (slewing, detector readout) make short exposures strictly worse.
- **The star.** Jitter sets a precision floor. With the jitter model of Part 2, each point has variance $\sigma_i^2 + s^2$, so once $\sigma_i \ll s$, better precision *per point* no longer helps. Only more points, spread over time, do.
- **The baseline.** You can't speed up time. A period longer than the campaign can't be found.
- **Cost and stability.** Telescope time is competitive, and the spectrograph must stay stable for years.

What the observations buy you is quantified by the exact posterior from your first MCMC lesson: $\sigma_K = \big(\sum_i \sin^2(2\pi t_i/P)/\sigma_i^2\big)^{-1/2}$. With well-spread phases, $\sum_i \sin^2 \approx N/2$, so

$$\sigma_K \approx \sigma\sqrt{\frac{2}{N}} \quad\longrightarrow\quad \sigma_K \gtrsim s\sqrt{\frac{2}{N}} \text{ once jitter dominates.}$$

(The MCMC notebook's 50 points give $\sigma_K = 2.1$ m/s.) Four times as many observations only halve the uncertainty, which is why the cost of each extra point matters.

**What engineers do about it.** This is the standard toolbox of signal processing, and most of it has a counterpart in radial-velocity work:

| Technique | Idea | In radial-velocity work |
|---|---|---|
| **Anti-alias filter** | Remove the frequencies you can't sample *before* sampling | An exposure of length $\tau$ is a boxcar average, whose response $\sin(\pi f\tau)/(\pi f\tau)$ suppresses frequencies above $\sim 1/\tau$. Exposures of roughly 10–15 minutes average over a Sun-like star's few-minute oscillations |
| **Oversampling** | Sample faster than needed, then filter and decimate | Take a few exposures per night and bin them |
| **Random sampling (dithering)** | Irregular times turn coherent aliases into a noise floor | Don't observe at the same hour every night. Use Lomb–Scargle, which handles uneven times |
| **Interleaved samplers** | Offset samplers fill each other's gaps | Telescopes at different longitudes, or space missions (Kepler, TESS) that observe continuously |
| **Window analysis** | Compute the window's spectrum, check whether a peak sits at $f \pm f_w$, and remove it (CLEAN, prewhitening) | Compare your periodogram with the spectral window of your schedule |
| **Adaptive design** | Choose the next sample to maximize the information gained | Schedule the next observation where the competing aliases predict the most different velocities |
| **Independent data** | A different measurement breaks the degeneracy | Transits or astrometry give the period directly |
| **Multimodal inference** | Keep every mode and weigh them | Parallel tempering, nested sampling, or comparing the Bayesian evidence of each alias |

The last row matters for us. The ensemble sampler of Part 3 explores *one* mode well and will not tell you about the others, so the periodogram, and a look at its aliases, is how we decide where to start.

### Optimize, *then* sample

From the periodogram peak, an optimizer climbs to the **MAP** (maximum a posteriori) point, and walkers start in a tiny ball around it. Even the optimizer can get stuck: a nearly-circular orbit with a smaller $K$ is a common trap. So the notebook launches several optimizations with different $\omega$ and keeps the best. Remember the MAP point is **not** the answer. It carries no uncertainty, and it isn't even invariant: the mode of a density moves when you change variables, because of the Jacobian factor, so the MAP in $(\ln P, h, k, \dots)$ is generally not the MAP in $(P, e, \omega, \dots)$. The answer is the whole posterior. Posterior medians and percentile intervals *are* invariant under monotonic reparametrization, which is one reason we report them.

---

## Part 3: Affine-Invariant Ensemble Sampling

### What's wrong with random-walk MH here?

Your MH sampler proposed $x' = x + \epsilon$ with $\epsilon \sim \mathcal{N}(0, \text{step}^2)$. In 7-D this breaks for two reasons:

1. **Wildly different scales.** In our posterior, $\ln P$ is pinned to about $\pm 3\times10^{-4}$ while $\gamma$ spreads over $\pm 0.5$ m/s. A step size that suits one is useless for the other: either every proposal is rejected, or the walker crawls.
2. **Correlations.** When two parameters trade off (a diagonal ridge in the posterior), axis-aligned steps can only zig-zag slowly along it.

The theoretical fix is to propose with the *posterior's own covariance*, $\epsilon \sim \mathcal{N}(0, \tfrac{2.38^2}{d}\Sigma)$. (Roberts, Gelman & Gilks 1997 proved this scaling is optimal for Gaussian-like targets as $d \to \infty$, where it gives an acceptance rate of 0.234. It's a good rule of thumb well beyond that setting.) But $\Sigma$ is what we're trying to find! The notebook's Part 8 shows exactly this: untuned MH gets stuck, while MH given $\Sigma$ ("cheating") is excellent.

### Affine invariance

Suppose you stretch, squash, rotate and shift the parameter space: $Y = AX + b$ for any invertible matrix $A$. A long, thin, tilted posterior $p(x)$ becomes a round blob $p_A(y) \propto p\big(A^{-1}(y - b)\big)$ (or vice versa). A sampler is **affine invariant** if running it on $p_A$ from the transformed starting ensemble produces exactly the transformed chain: $Y(t) = AX(t) + b$ at every step, given the same random numbers. Then its efficiency (acceptance rate, autocorrelation time) is the *same* for $p$ and $p_A$. It doesn't care about scales or linear correlations at all, which is exactly the robustness we're missing.

### The stretch move

Goodman & Weare (2010) achieve this with an **ensemble** of $n$ walkers $\{X_1, \dots, X_n\}$ exploring simultaneously. To update walker $X_k$:

1. Pick a random **other** walker $X_j$.
2. Draw a stretch factor $z \in [1/a, a]$ from $g(z) \propto 1/\sqrt{z}$ (usually $a = 2$).
3. Propose a point on the line through both walkers:
$$Y = X_j + z\,(X_k - X_j)$$
   For $z > 1$ this stretches *away* from $X_j$; for $z < 1$ it contracts toward it.
4. Accept with probability
$$\min\left(1,\; z^{\,d-1}\,\frac{p(Y)}{p(X_k)}\right)$$

Why is this affine invariant? Apply $x \mapsto Ax + b$ to every walker. Then
$$AX_j + b + z\big((AX_k + b) - (AX_j + b)\big) = A\big(X_j + z(X_k - X_j)\big) + b = AY + b,$$
so the proposal transforms exactly like the walkers. The acceptance ratio is unchanged too, since $p_A(AY+b)/p_A(AX_k+b) = p(Y)/p(X_k)$ (the Jacobian $|\det A|^{-1}$ cancels in the ratio). The walkers' spread *is* the step-size matrix, and it adapts automatically.

The ensemble needs to span the whole space, though. Every proposal lies in the affine hull of the current walkers (the smallest flat subspace containing them), so if the walkers start inside a lower-dimensional subspace, they stay there forever. That requires $n \ge d + 1$ walkers in general position. In practice you want many more (see the rules of thumb below).

### Where does $z^{d-1}$ come from? (Detailed balance, again!)

In your last lesson you showed Metropolis-Hastings leaves the target $\pi$ invariant because it satisfies **detailed balance**:

$$\pi(x)\,T(x \to x') = \pi(x')\,T(x' \to x)$$

For a **symmetric** proposal the acceptance ratio was just $\pi(x')/\pi(x)$. For an asymmetric proposal density $q$, it becomes the Hastings ratio

$$\frac{\pi(x')\,q(x' \to x)}{\pi(x)\,q(x \to x')}$$

The stretch move is asymmetric, so let's compute $q$ explicitly.

**Setting up.** Hold the partner $c = X_j$ fixed. (It isn't changed by this update, so we may condition on it.) Write the current walker as $x = c + r\,u$, where $u$ is a unit vector and $r > 0$. The proposal $y = c + z r\,u$ stays on the same ray from $c$, with new distance $r' = zr$. (Since $z > 0$, it never crosses to the other side of $c$.) The reverse move from $y$ back to $x$ uses the same ray and needs $z' = 1/z$.

**Two ingredients.**

1. **The 1-D proposal density.** If $z$ has density $g$, then $r' = zr$ has density $q(r \to r') = g(r'/r)/r$ (change of variables, $dz = dr'/r$). Likewise $q(r' \to r) = g(r/r')/r'$.
2. **The $d$-dimensional volume element.** In polar coordinates around $c$, $d^dx = r^{d-1}\,dr\,d\Omega$. So the target restricted to the ray has density $\pi(c + r u)\,r^{d-1}$ in $r$, not just $\pi(c + ru)$. There's more volume at larger radius: a shell's area grows like $r^{d-1}$.

**The Hastings ratio on the ray** is therefore

$$\frac{\pi(y)\,r'^{\,d-1}\;q(r'\to r)}{\pi(x)\,r^{d-1}\;q(r\to r')} = \frac{\pi(y)}{\pi(x)}\cdot z^{d-1}\cdot\frac{g(1/z)/r'}{g(z)/r} = \frac{\pi(y)}{\pi(x)}\cdot z^{d-1}\cdot\frac{g(1/z)}{z\,g(z)}$$

The choice $g(z) \propto 1/\sqrt{z}$ on $[1/a, a]$ is made precisely so that $g(1/z) = \sqrt{z} = z\,g(z)$. The last fraction is then exactly 1, leaving

$$A = \min\left(1,\; z^{\,d-1}\,\frac{\pi(Y)}{\pi(X_k)}\right)$$

(The interval $[1/a, a]$ is symmetric under $z \to 1/z$, so every move has its reverse available.) Leave out $z^{d-1}$ and the sampler still runs, but it over-favours contracting moves and samples the **wrong distribution**. Check 6 in the notebook will catch you. For the original treatment, see Goodman & Weare (2010), Section 3.

### Serial updates and the complementary ensemble

The ensemble samples the **joint** distribution $\Pi(X_1, \dots, X_n) = \pi(X_1)\cdots\pi(X_n)$: each walker is an independent draw from $\pi$ once converged. The derivation above is a valid MH update of $X_k$ with the others held fixed, i.e. it targets $\pi(X_k)$ *given* the current $X_{j \ne k}$. So it leaves $\Pi$ invariant, just as each sweep of a Gibbs sampler does.

That's why walkers are updated **one at a time**, each using the *current* positions of the others, including walkers already moved in the same sweep. Moving all walkers simultaneously, each using the others' *old* positions, is not a sequence of such conditional updates, and the proof no longer applies. The fix `emcee` uses (stretch goal 5) is to split the walkers into two halves and update all of one half at once using only the other half as partners. Within a half, the partners really are held fixed, so each update is again a valid conditional move.

Rules of thumb:

- The hard minimum is $n \ge d + 1$ (to span the space). Use at least $2d$ walkers; $4d$–$5d$ is comfortable. (We use 32 for $d = 7$.)
- A healthy **acceptance fraction** is roughly 0.2–0.5. Near 0: walkers are stuck. Near 1: something's wrong, e.g. the posterior is flat because of a bug.

---

## Part 4: Has It Converged?

### Autocorrelation time

Each MCMC sample is correlated with the ones before it. For a chain that has reached stationarity, the **normalized autocorrelation function** measures how similar the chain is to itself $\ell$ steps later:

$$\rho(\ell) = \frac{\operatorname{Cov}(x_t,\, x_{t+\ell})}{\operatorname{Var}(x_t)}$$

(Stationarity means this depends only on the lag $\ell$, not on $t$.) The **integrated autocorrelation time** adds it all up:

$$\tau = 1 + 2\sum_{\ell=1}^{\infty}\rho(\ell)$$

**Where it comes from.** Estimate the posterior mean with the chain average $\bar x = \frac1N\sum_t x_t$, and write $\sigma^2 = \operatorname{Var}(x_t)$. Expanding the variance of a sum,

$$\operatorname{Var}(\bar x) = \frac{1}{N^2}\sum_{s,t}\operatorname{Cov}(x_s, x_t) = \frac{\sigma^2}{N}\sum_{|\ell| < N}\left(1 - \frac{|\ell|}{N}\right)\rho(\ell) \;\xrightarrow{N \gg \tau}\; \frac{\sigma^2}{N}\,\tau$$

For independent samples it would be $\sigma^2/N$. So the chain behaves like $N/\tau$ independent samples: **it takes about $\tau$ steps to produce one independent sample.** Treating the walkers as independent chains (a good approximation once they've spread out),

$$N_\text{eff} \approx \frac{N_\text{steps} \times N_\text{walkers}}{\tau}$$

Example: $\tau = 80$ with 10,000 steps × 32 walkers gives $N_\text{eff} \approx 4000$, not 320,000.

### Estimating $\tau$ without fooling yourself

Computing $\rho(\ell)$ directly is $O(N^2)$. The **Wiener–Khinchin theorem** says autocorrelation is the inverse Fourier transform of the power spectrum, so an FFT gets it in $O(N\log N)$ (provided in the notebook). The FFT treats the series as periodic, so the notebook zero-pads it to at least $2N$ first. Otherwise the end of the chain would wrap around and correlate with its beginning.

The subtle part is the sum. At large lags, each $\hat\rho(\ell)$ is estimated from few effectively independent pairs, and is mostly noise. Summing all of them makes $\hat\tau$'s variance grow in proportion to the number of terms, so $\hat\tau$ wanders randomly. **Sokal's adaptive window** stops the sum at the first lag $W$ satisfying

$$W \ge c\,\hat\tau(W), \qquad \hat\tau(W) = 1 + 2\sum_{\ell=1}^{W}\hat\rho(\ell), \qquad c \approx 5$$

(The notebook calls the window $M$; we use $W$ here because $M$ is the mean anomaly.) The window grows with $\tau$ itself. If $\rho$ decays roughly like $e^{-\ell/\tau}$, the terms dropped beyond $5\tau$ are about $e^{-5} \approx 0.7\%$ of the sum: a small bias, traded for a large cut in noise.

For an ensemble, **average $\rho(\ell)$ over walkers first**, then apply the window. That's much less noisy than computing $\tau$ per walker and averaging.

### Practical checklist

- **Run length:** $N_\text{steps} \gtrsim 50\tau$. Below that, even the estimate of $\tau$ isn't trustworthy.
- **Burn-in:** discard the first few $\tau$ (the notebook uses $3\tau$), while the ensemble spreads out from its starting ball.
- **Thinning:** keeping every $\sim\tau/2$-th step loses almost no information and makes the sample set manageable. It's purely a storage convenience: thinning never makes an estimate *more* precise than using every sample.
- **Look at the trace plots.** A walker stuck far from the others, or a slow trend, means you're not done, whatever the numbers say. No diagnostic can prove convergence. It can only reveal non-convergence.

A trick for testing $\tau$: the AR(1) process $x_{n+1} = \phi x_n + \epsilon_n$ with independent noise $\epsilon_n$ and $|\phi| < 1$. Once it's stationary, multiplying by $x_{n+1-\ell}$ and taking expectations gives $\operatorname{Cov}(x_{n+1}, x_{n+1-\ell}) = \phi\,\operatorname{Cov}(x_n, x_{n+1-\ell})$, so $\rho(\ell) = \phi^\ell$ exactly. Summing the geometric series gives $\tau = 1 + \frac{2\phi}{1-\phi} = \frac{1+\phi}{1-\phi}$. Check 7 uses this.

---

## Part 5: From $K$ to the Planet's Mass

Rearranging the $K$ equation gives the **mass function**:

$$\frac{(m\sin i)^3}{(M_\star + m)^2} = \frac{P K^3 (1-e^2)^{3/2}}{2\pi G}$$

The right side is measured. The left side still contains the unknown $i$, inside $M_\star + m$. The standard convention for the **minimum mass** is to set $\sin i = 1$ in that total-mass term, which makes the equation a cubic in the single unknown $m$. (For $m \ll M_\star$ the difference is negligible: it changes $m\sin i$ by a fraction of at most about $\frac23\frac{m}{M_\star}$.) Rather than use the cubic formula, solve it by **fixed-point iteration**:

$$m_{n+1} = \phi(m_n) = C\,(M_\star + m_n)^{2/3}, \qquad C = K\sqrt{1-e^2}\left(\frac{P}{2\pi G}\right)^{1/3}, \qquad m_0 = 0$$

**Why it converges so fast.** Near the solution $m^*$, the error shrinks by a factor of $|\phi'(m^*)|$ per iteration, and
$$\phi'(m) = \frac23\,C\,(M_\star + m)^{-1/3} = \frac23\,\frac{\phi(m)}{M_\star + m} \;\Rightarrow\; \phi'(m^*) = \frac23\,\frac{m^*}{M_\star + m^*}$$
For our planet that's about $1.3\times10^{-3}$: each iteration gains roughly three correct digits, so five or six iterations reach machine precision. (Convergence from $m_0 = 0$ is guaranteed. $\phi$ is increasing, and $\phi(0) > 0 = m_0$, so the iterates increase monotonically. They can never overshoot the fixed point, because $m_n < m^*$ implies $m_{n+1} = \phi(m_n) < \phi(m^*) = m^*$. A bounded increasing sequence converges, and its limit must be the fixed point.)

The first iteration ($m_1$, using $M_\star + m \approx M_\star$) is already the "textbook" approximation; later iterations fix it up.

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
- J. VanderPlas (2018), *Understanding the Lomb–Scargle Periodogram*, ApJS 236, 16. Uneven sampling, window functions and aliases, explained carefully.
- R. Dawson & D. Fabrycky (2010), *Radial velocity planets de-aliased: a new, short period for super-Earth 55 Cnc e*, ApJ 722, 937. A real planet with the wrong period.
- M. Mayor & D. Queloz (1995), *A Jupiter-mass companion to a solar-type star*, Nature 378, 355. 51 Peg b, the discovery that started it all.
