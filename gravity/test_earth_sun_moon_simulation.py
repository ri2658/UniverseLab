import math
import tempfile
import unittest
from pathlib import Path

from gravity.earth_sun_moon_simulation import (
    create_earth_sun_moon_system,
    simulate_n_body,
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
            write_trajectories_svg(trajectories, svg_path)
            content = svg_path.read_text(encoding="utf-8")

        self.assertIn("<svg", content)
        self.assertIn("Sun", content)
        self.assertIn("Earth", content)
        self.assertIn("Moon", content)


if __name__ == "__main__":
    unittest.main()
