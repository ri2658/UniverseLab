import math
import tempfile
import unittest
from pathlib import Path

from gravity.earth_sun_moon_simulation import (
    create_earth_sun_moon_system,
    simulate_n_body,
    total_energy,
    write_trajectories_svg,
)


class EarthSunMoonSimulationTests(unittest.TestCase):
    def test_initial_system_has_three_bodies(self):
        bodies = create_earth_sun_moon_system()
        self.assertEqual([body.name for body in bodies], ["Sun", "Earth", "Moon"])

    def test_simulation_keeps_values_finite(self):
        bodies = create_earth_sun_moon_system()
        trajectories = simulate_n_body(bodies, dt_seconds=3600, steps=24)
        self.assertEqual(set(trajectories.keys()), {"Sun", "Earth", "Moon"})

        for points in trajectories.values():
            self.assertEqual(len(points), 25)
            for x, y in points:
                self.assertTrue(math.isfinite(x))
                self.assertTrue(math.isfinite(y))

    def test_svg_writer_outputs_labeled_paths(self):
        bodies = create_earth_sun_moon_system()
        trajectories = simulate_n_body(bodies, dt_seconds=3600, steps=4)

        with tempfile.TemporaryDirectory() as tmp_dir:
            svg_path = Path(tmp_dir) / "simulation.svg"
            write_trajectories_svg(trajectories, svg_path, days=4 / 24, step_hours=1.0)
            content = svg_path.read_text(encoding="utf-8")

        self.assertIn("<svg", content)
        self.assertIn("Sun", content)
        self.assertIn("Earth", content)
        self.assertIn("Moon", content)

    def test_leapfrog_conserves_energy_over_one_year(self):
        # Leapfrog is symplectic: energy oscillates around the true value but must not drift.
        bodies = create_earth_sun_moon_system()
        initial_energy = total_energy(bodies)
        simulate_n_body(bodies, dt_seconds=3600, steps=365 * 24)
        relative_error = abs((total_energy(bodies) - initial_energy) / initial_energy)
        self.assertLess(relative_error, 1e-6)


if __name__ == "__main__":
    unittest.main()
