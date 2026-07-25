from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

G = 6.67430e-11  # m^3 kg^-1 s^-2


@dataclass
class Body:
    name: str
    mass: float
    x: float
    y: float
    vx: float
    vy: float
    color: str


def create_earth_sun_moon_system() -> list[Body]:
    sun = Body("Sun", 1.98847e30, 0.0, 0.0, 0.0, 0.0, "#f6c431")
    earth = Body("Earth", 5.9722e24, 1.495978707e11, 0.0, 0.0, 29_780.0, "#3b82f6")
    moon = Body(
        "Moon",
        7.34767309e22,
        earth.x + 384_400_000.0,
        earth.y,
        earth.vx,
        earth.vy + 1_022.0,
        "#9ca3af",
    )
    return [sun, earth, moon]


def simulate_n_body(bodies: list[Body], dt_seconds: float, steps: int) -> dict[str, list[tuple[float, float]]]:
    trajectories = {body.name: [(body.x, body.y)] for body in bodies}

    for _ in range(steps):
        # 1. Calculate INITIAL accelerations (if first step) or use previous step's final accelerations.
        # For simplicity, we calculate them fresh here, though it can be optimized.
        accelerations = []
        for i, body in enumerate(bodies):
            ax = 0.0
            ay = 0.0
            for j, other in enumerate(bodies):
                if i == j:
                    continue
                dx = other.x - body.x
                dy = other.y - body.y
                dist_sq = dx * dx + dy * dy
                dist = dist_sq**0.5
                if dist == 0.0:
                    continue
                force_per_mass = G * other.mass / dist_sq
                ax += force_per_mass * dx / dist
                ay += force_per_mass * dy / dist
            accelerations.append((ax, ay))

        # 2. First KICK (half-step velocity) & DRIFT (full-step position)
        for i, (body, (ax, ay)) in enumerate(zip(bodies, accelerations)):
            body.vx += 0.5 * ax * dt_seconds
            body.vy += 0.5 * ay * dt_seconds
            body.x += body.vx * dt_seconds
            body.y += body.vy * dt_seconds

        # 3. Recalculate NEW accelerations based on NEW positions
        new_accelerations = []
        for i, body in enumerate(bodies):
            ax = 0.0
            ay = 0.0
            for j, other in enumerate(bodies):
                if i == j:
                    continue
                dx = other.x - body.x
                dy = other.y - body.y
                dist_sq = dx * dx + dy * dy
                dist = dist_sq**0.5
                if dist == 0.0:
                    continue
                force_per_mass = G * other.mass / dist_sq
                ax += force_per_mass * dx / dist
                ay += force_per_mass * dy / dist
            new_accelerations.append((ax, ay))

        # 4. Second KICK (half-step velocity) with NEW accelerations
        for i, (body, (ax, ay)) in enumerate(zip(bodies, new_accelerations)):
            body.vx += 0.5 * ax * dt_seconds
            body.vy += 0.5 * ay * dt_seconds
            trajectories[body.name].append((body.x, body.y))

    return trajectories


def write_trajectories_csv(trajectories: dict[str, list[tuple[float, float]]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["body", "step", "x_m", "y_m"])
        for body_name, points in trajectories.items():
            for step, (x, y) in enumerate(points):
                writer.writerow([body_name, step, x, y])


def write_trajectories_svg(trajectories: dict[str, list[tuple[float, float]]], output_path: Path, days: float, step_hours: float) -> None:
    all_points = [point for points in trajectories.values() for point in points]
    min_x = min(x for x, _ in all_points)
    max_x = max(x for x, _ in all_points)
    min_y = min(y for _, y in all_points)
    max_y = max(y for _, y in all_points)

    width = 900
    height = 900
    padding = 60

    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)

    def to_viewport(x: float, y: float) -> tuple[float, float]:
        norm_x = (x - min_x) / span_x
        norm_y = (y - min_y) / span_y
        view_x = padding + norm_x * (width - 2 * padding)
        view_y = height - (padding + norm_y * (height - 2 * padding))
        return view_x, view_y

    palette = {
        "Sun": "#f6c431",
        "Earth": "#3b82f6",
        "Moon": "#9ca3af",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("<svg xmlns='http://www.w3.org/2000/svg' width='900' height='900' viewBox='0 0 900 900'>\n")
        handle.write("  <rect width='100%' height='100%' fill='#050b18'/>\n")
        handle.write(f"  <text x='20' y='35' fill='white' font-size='22'>Earth-Sun-Moon n-body simulation ({days} days, {step_hours} hour intervals)</text>\n")
        for body_name, points in trajectories.items():
            converted = [to_viewport(x, y) for x, y in points]
            path_points = " ".join(f"{x:.2f},{y:.2f}" for x, y in converted)
            color = palette.get(body_name, "#ffffff")
            handle.write(f"  <polyline points='{path_points}' fill='none' stroke='{color}' stroke-width='1.6'/>\n")
            final_x, final_y = converted[-1]
            handle.write(f"  <circle cx='{final_x:.2f}' cy='{final_y:.2f}' r='4' fill='{color}'/>\n")
            handle.write(
                f"  <text x='{final_x + 8:.2f}' y='{final_y - 8:.2f}' fill='{color}' font-size='14'>{body_name}</text>\n"
            )

        handle.write("</svg>\n")


def run_simulation(days: float, step_hours: float) -> tuple[Path, Path]:
    dt_seconds = step_hours * 3600
    steps = int((days * 24) / step_hours)

    bodies = create_earth_sun_moon_system()
    trajectories = simulate_n_body(bodies, dt_seconds=dt_seconds, steps=steps)

    repo_root = Path(__file__).resolve().parents[1]
    dataset_path = repo_root / "datasets" / "earth_sun_moon_trajectories.csv"
    visualization_path = repo_root / "visualizations" / "earth_sun_moon_simulation.svg"

    write_trajectories_csv(trajectories, dataset_path)
    write_trajectories_svg(trajectories, visualization_path, days, step_hours)
    return dataset_path, visualization_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Earth-Sun-Moon n-body simulation outputs.")
    parser.add_argument("--days", type=float, default=30.0, help="Total simulated days (default: 30).")
    parser.add_argument("--step-hours", type=float, default=2.0, help="Simulation time step in hours (default: 2).")
    args = parser.parse_args()

    dataset_path, visualization_path = run_simulation(days=args.days, step_hours=args.step_hours)
    print(f"Wrote trajectories to: {dataset_path}")
    print(f"Wrote visualization to: {visualization_path}")


if __name__ == "__main__":
    main()
