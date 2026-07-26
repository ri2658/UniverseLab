# UniverseLab

This repository now includes a starter n-body gravity simulation for the **Sun, Earth, and Moon**.

## Run the simulation

```bash
python gravity/earth_sun_moon_simulation.py
```

Outputs:

- `datasets/earth_sun_moon_trajectories.csv`
- `visualizations/earth_sun_moon_simulation.svg`

To generate the animated version, run:

```bash
python gravity/animate_simulation.py
```

## Sample Visualizations

### Static orbit plot

![Earth-Sun-Moon simulation SVG](visualizations/earth_sun_moon_simulation.svg)

### Animated orbit preview

![Earth-Sun-Moon simulation GIF](visualizations/earth_sun_moon_3d.gif)

### Additional example output

![MCMC posterior example](visualizations/mcmc_posterior.svg)

## Directory skeleton

- `gravity/`
- `stars/`
- `galaxies/`
- `blackholes/`
- `cosmology/`
- `machine_learning/`
- `physics/`
- `datasets/`
- `visualizations/`
- `experiments/`
