import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Optional, Tuple

def get_position_at_time( ID_pedestrian: int = 0, t: float = 0.0, off_x: float = 15.0, off_y: float = -2.0, csv_filename: str = "crowds_trajectories.csv") -> Optional[Tuple[float, float, float, float]]:    
    """
    Returns x, y, and heading for a given pedestrian at specified time t.

    Parameters:
        ID_pedestrian (int): The ID of the pedestrian.
        t (float): Time at which to get the position.
        off_x (float): Offset to add to x coordinate.
        off_y (float): Offset to add to y coordinate.
        csv_filename (str): The CSV file containing trajectory data.

    Returns:
        tuple: (x, y, heading) at time t (heading in radians)
        None: if no data is available for interpolation.
    """
    # Load the CSV data
    df = pd.read_csv(csv_filename)
    
    # Filter for the specified pedestrian and sort by time
    ped_df = df[df["agent_id"] == ID_pedestrian].sort_values("time")
    
    if ped_df.empty:
        return None
    
    # Extract the time, position, heading, and velocity columns.
    times = ped_df["time"].values.astype(float)
    x_data = ped_df["pos_x"].values.astype(float)
    y_data = ped_df["pos_y"].values.astype(float)
    heading_data = ped_df["heading"].values.astype(float) if "heading" in ped_df.columns else np.arctan2(np.gradient(y_data), np.gradient(x_data))
    velocity_data = ped_df["velocity"].values.astype(float) if "velocity" in ped_df.columns else np.sqrt(np.gradient(x_data)**2 + np.gradient(y_data)**2)

    # Check if t is within the time range
    if t < times[0] or t > times[-1]:
        return None
    
    # Interpolate x, y, heading, and velocity at time t
    x = np.interp(t, times, x_data) + off_x
    y = np.interp(t, times, y_data) + off_y
    heading = np.interp(t, times, heading_data)
    velocity = np.interp(t, times, velocity_data)
    
    return x, y, heading, velocity

# ----------------------------
# Main plotting with animation using get_position_at_time
# ----------------------------
if __name__ == "__main__":
    csv_filename = "crowds_trajectories.csv"
    
    # Load the CSV once to determine global time range and pedestrian IDs.
    df = pd.read_csv(csv_filename)
    if "time" not in df.columns and "frame_id" in df.columns:
        original_fps = 25
        df["time"] = df["frame_id"] / original_fps

    pedestrian_ids = df["agent_id"].unique()
    selected_pedestrians = pedestrian_ids[:15]

    colors = plt.cm.rainbow(np.linspace(0, 1, len(selected_pedestrians)))

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlabel("Posizione X (metri)")
    ax.set_ylabel("Posizione Y (metri)")
    ax.set_title(f"Animazione delle traiettorie di {len(selected_pedestrians)} pedoni")
    ax.grid(True)

    # Create dictionaries to store trajectory data and matplotlib objects for each pedestrian.
    traj_data = {ped: {"x": [], "y": []} for ped in selected_pedestrians}
    lines = {}
    quivers = {}
    arrow_length = 0.5  # Adjust arrow length as needed

    for ped, color in zip(selected_pedestrians, colors):
        # Create an empty line for the trajectory.
        line, = ax.plot([], [], 'o-', lw=2, color=color, label=f'Pedone {ped}')
        lines[ped] = line
        # Create an initial quiver arrow (will be updated later).
        q = ax.quiver(0, 0, 0, 0, angles='xy', scale_units='xy', scale=1, color=color)
        quivers[ped] = q

    # Determine global start and end times from the CSV data.
    global_start_time = df["time"].min()
    global_end_time = df["time"].max()
    dt = 0.1  # time step in seconds
    num_steps = int((global_end_time - global_start_time) / dt) + 1

    # Set plot limits (applying the same offsets as in get_position_at_time).
    all_x = df["pos_x"] + 15
    all_y = df["pos_y"] - 2
    ax.set_xlim(all_x.min() - 1, all_x.max() + 1)
    ax.set_ylim(all_y.min() - 1, all_y.max() + 1)
    ax.legend()

    def init():
        for ped in selected_pedestrians:
            lines[ped].set_data([], [])
            quivers[ped].set_offsets(np.array([[0, 0]]))
        return list(lines.values()) + list(quivers.values())

    def update(frame_idx):
        current_time = global_start_time + frame_idx * dt
        for ped in selected_pedestrians:
            pos = get_position_at_time(ped, current_time, csv_filename=csv_filename)
            if pos is not None:
                x, y, heading = pos
                # Append current position to the trajectory.
                traj_data[ped]["x"].append(x)
                traj_data[ped]["y"].append(y)
                # Update the line (trajectory) for this pedestrian.
                lines[ped].set_data(traj_data[ped]["x"], traj_data[ped]["y"])
                # Update the arrow: set its new position and direction.
                quivers[ped].set_offsets(np.array([[x, y]]))
                quivers[ped].set_UVC(np.cos(heading) * arrow_length, np.sin(heading) * arrow_length)
        return list(lines.values()) + list(quivers.values())

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=num_steps,
        init_func=init,
        interval=100,  # 100 ms per frame
        blit=True,
        repeat=False
    )

    plt.show()
