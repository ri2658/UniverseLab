import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from earth_sun_moon_simulation import create_earth_sun_moon_system, simulate_n_body
from pathlib import Path

def generate_animation(_days: float = 365.0, _step_hours: float = 24.0):
    print("Running simulation physics...")
    # Simulate a full year so we can see the Earth orbit the Sun
    # Step size of 24 hours to keep the number of frames (365) reasonable for a GIF
    days = _days
    step_hours = _step_hours
    dt_seconds = step_hours * 3600
    steps = int((days * 24) / step_hours)

    bodies = create_earth_sun_moon_system()
    trajectories = simulate_n_body(bodies, dt_seconds=dt_seconds, steps=steps)

    print("Setting up 3D Matplotlib canvas...")
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(8, 8))
    # We use a 3D projection even though our physics is coplanar (z=0)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#050b18')
    fig.patch.set_facecolor('#050b18')

    # Calculate bounding box for the solar system based on Earth's orbit
    all_points = [p for name in ["Sun", "Earth"] for p in trajectories[name]]
    min_x = min(x for x, _ in all_points)
    max_x = max(x for x, _ in all_points)
    min_y = min(y for _, y in all_points)
    max_y = max(y for _, y in all_points)
    
    span = max(max_x - min_x, max_y - min_y) / 2
    cx = (max_x + min_x) / 2
    cy = (max_y + min_y) / 2
    
    ax.set_xlim(cx - span*1.1, cx + span*1.1)
    ax.set_ylim(cy - span*1.1, cy + span*1.1)
    ax.set_zlim(-span*1.1, span*1.1) # Create a cubic 3D volume
    
    ax.axis('off') # Hide grid and axes for a cinematic space look
    ax.set_title(f"3D Project of Earth-Sun-Moon System ({days} days, {step_hours} hour intervals)", color='white', pad=20, fontsize=14)
    time_text = ax.text2D(0.02, 0.96, "", transform=ax.transAxes, color='white', fontsize=12)

    colors = {"Sun": "#f6c431", "Earth": "#3b82f6", "Moon": "#9ca3af"}
    sizes = {"Sun": 300, "Earth": 80, "Moon": 20}
    
    lines = {}
    points = {}
    
    # Initialize empty plot elements
    for name in trajectories.keys():
        line, = ax.plot([], [], [], color=colors[name], alpha=0.5, linewidth=1.5)
        # 3D scatter
        point = ax.scatter([], [], [], color=colors[name], s=sizes[name], edgecolors='white', linewidths=0.5)
        lines[name] = line
        points[name] = point

    def update(frame):
        ex, ey = trajectories["Earth"][frame]
        
        for name in trajectories.keys():
            x, y = trajectories[name][frame]
            
            # VISUALIZATION HACK:
            # The Earth-Moon distance is ~400x smaller than the Sun-Earth distance.
            # If we plot them 1:1, the Moon is inside the Earth's pixels.
            # So, purely for visualization, we artificially scale the Moon's distance from Earth by 40x.
            if name == "Moon":
                x = ex + (x - ex) * 40
                y = ey + (y - ey) * 40

            # Draw the full orbital path from the beginning of the simulation
            start_idx = 0
            
            if name == "Moon":
                tail_x = []
                tail_y = []
                for i in range(start_idx, frame+1):
                    tx, ty = trajectories["Moon"][i]
                    tex, tey = trajectories["Earth"][i]
                    tail_x.append(tex + (tx - tex) * 40)
                    tail_y.append(tey + (ty - tey) * 40)
            else:
                tail_x = [p[0] for p in trajectories[name][start_idx:frame+1]]
                tail_y = [p[1] for p in trajectories[name][start_idx:frame+1]]
                
            tail_z = np.zeros_like(tail_x)
            
            # Update path
            lines[name].set_data(tail_x, tail_y)
            lines[name].set_3d_properties(tail_z)
            
            # Update planet position
            points[name]._offsets3d = ([x], [y], [0])
            
        # Keep the camera stationary but with a nice cinematic isometric angle
        ax.view_init(elev=35, azim=45)
        sim_hours = frame * step_hours
        sim_days = int(sim_hours // 24)
        remaining_hours = int(sim_hours % 24)
        time_text.set_text(f"Day {sim_days}, Hour {remaining_hours:02d}")
        
        return list(lines.values()) + list(points.values())

    print(f"Rendering {steps} frames...")
    anim = FuncAnimation(fig, update, frames=steps, interval=30, blit=False)
    
    out_path = Path(__file__).resolve().parents[1] / "visualizations" / "earth_sun_moon_3d.gif"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out_path, writer=PillowWriter(fps=30))
    print(f"Saved animation to {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate Earth-Sun-Moon n-body simulation outputs.")
    parser.add_argument("--days", type=float, default=365.0, help="Total simulated days (default: 363).")
    parser.add_argument("--step-hours", type=float, default=24.0, help="Simulation time step in hours (default: 24).")
    args = parser.parse_args()
    generate_animation(_days=args.days, _step_hours=args.step_hours)

if __name__ == "__main__":
    main()
