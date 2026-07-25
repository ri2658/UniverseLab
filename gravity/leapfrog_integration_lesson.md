# Numerical Methods: Fixing "Leaky" Physics

Welcome to your first lesson in numerical methods! We are going to look at the math behind how computers simulate continuous physics (like gravity) in discrete time steps.

## The Problem: The Euler Method
In `earth_sun_moon_simulation.py`, you currently calculate orbits using the **Forward Euler Method**. It's the most intuitive way to simulate physics.

Here is the core logic you currently use:
1. Calculate the current force (acceleration) on a planet.
2. Update its velocity: $v_{new} = v_{old} + a \cdot \Delta t$
3. Update its position: $x_{new} = x_{old} + v_{new} \cdot \Delta t$

### Why is this "leaky"?
In calculus, $\Delta t$ is infinitely small. But in a computer simulation, $\Delta t$ is a discrete chunk (e.g., 2 hours in your simulation). 
Because gravity gets stronger as planets get closer, if Earth moves toward the Sun for 2 whole hours based *only* on its starting acceleration, it will "overshoot" its true curved path. 

This creates a small error in every single frame. The error specifically **adds artificial energy** to the system. Over thousands of years, the Earth will slowly spiral outward into deep space because the math itself is "leaking" energy into the orbit!

## The Solution: Leapfrog Integration (A Symplectic Integrator)
In computational astrophysics, we use **Symplectic Integrators**. "Symplectic" is a fancy math term which means the algorithm is specifically designed to conserve the total energy (Hamiltonian) of the system over long periods.

The most famous beginner-friendly symplectic integrator is the **Leapfrog Method**.

### How Leapfrog Works
Instead of calculating position and velocity at the exact same time, we calculate them offset by a **half-step** ($\Delta t / 2$). They "leapfrog" over each other!

1. **Kick (Half-step velocity):** Update velocity for a half-step using current acceleration.
   $$v_{t+1/2} = v_{t} + a_t \cdot \frac{\Delta t}{2}$$
2. **Drift (Full-step position):** Update position for a full step using the *new* half-step velocity.
   $$x_{t+1} = x_{t} + v_{t+1/2} \cdot \Delta t$$
3. **Calculate New Acceleration:** Find the new forces at the new position $x_{t+1}$.
4. **Kick (Half-step velocity):** Finish updating the velocity for the remaining half-step.
   $$v_{t+1} = v_{t+1/2} + a_{t+1} \cdot \frac{\Delta t}{2}$$

Because the position is updated using the velocity from the *middle* of the time step, it averages out the "overshoot" error perfectly. Energy fluctuates slightly frame-to-frame, but it never spirals out of control!

## Your Challenge
Let's upgrade your code! Open up `earth_sun_moon_simulation.py`. 

Right now, your `simulate_n_body` function does everything in one big loop. To implement Leapfrog, we need to separate the "calculate acceleration" math from the "update position/velocity" math.

Are you ready to refactor `simulate_n_body` to use the **Kick-Drift-Kick** Leapfrog algorithm? I can either guide you step-by-step, or I can write the code and explain it!
