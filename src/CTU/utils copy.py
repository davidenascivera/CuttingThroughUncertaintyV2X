import numpy as np
import json, random
from shapely.geometry import Polygon, Point
from typing import Tuple
from SME_mine.utils import vertices_meas_range_bearing
from constants import MAP_MIN_X, MAP_MIN_Y, MAP_MAX_X, MAP_MAX_Y, color_dict
from typing import Optional, Tuple
import pandas as pd 
import logging
logger = logging.getLogger(__name__)

def ray_intersects_rectangle(ray_origin, ray_direction, rect_pos, rect_width, rect_height):
    """
    Same intersection function from your code ...
    """
    rect_min = np.array(rect_pos)
    rect_max = rect_min + np.array([rect_width, rect_height])
    
    # Add check to prevent division by zero
    ray_direction = np.array(ray_direction)
    # Replace zeros with very small values
    ray_direction = np.where(ray_direction == 0, 1e-10, ray_direction)
    
    inv_dir = 1.0 / ray_direction
    t1 = (rect_min - ray_origin)*inv_dir
    t2 = (rect_max - ray_origin)*inv_dir
    tmin = np.max(np.minimum(t1, t2))
    tmax = np.min(np.maximum(t1, t2))
    if tmax < 0 or tmin > tmax:
        return False, None
    return True, ray_origin + tmin*ray_direction

def lin_velocity(pos, start_time, time_steps, velocity):
    """
    Same velocity function from your code ...
    """
    x, y = pos
    if time_steps >= start_time:
        vel_meter_per_sec = velocity * 1000 / 3600
        x_next = x + vel_meter_per_sec*0.1


    return (x_next,y)

def random_point_in_polygon(polygon: Polygon) -> Point:
    minx, miny, maxx, maxy = polygon.bounds
    while True:
        x = random.normal(minx, maxx)
        y = random.normal(miny, maxy)
        p = Point(x, y)
        if polygon.contains(p):
            return p
          
def check_point_in_polygon(poly_set: np.ndarray, point: Tuple[float,float]) -> bool:
  meas_polygon = Polygon(poly_set[:-1] if np.allclose(poly_set[0], poly_set[-1]) else poly_set)
  vru_point = Point(point)
  is_contained = meas_polygon.contains(vru_point)
  return is_contained


def get_sensor_list(sensor_pos: Tuple[float, float], sorted_hits: list[dict[str, Tuple[np.float64, np.float64]]], dR: float, dA: float, id_sens:int, 
                    renderer:any, dt:float,percent_faulty:float, measurements = {}, plot_set: bool=True, counter:int = 0) -> None:
  '''
  Function that helps generating the random poits for a sensor given the range and bearing uncertainty
  '''
  if sorted_hits != []: # Check what the RSU sees 
    for vru in sorted_hits: # For each pedestrian seen by the RSU
      
      # Get the pedestrian ID 
      ped_id = vru['id']   # Get pedestrian , give 2 _ ID for the RSU ID
      ped_pos = vru['pos']
      ped_head = vru['head']
      ped_vel = vru['vel']
      orgin_sensor = sensor_pos
      ped_r = np.linalg.norm(np.array(ped_pos) - np.array(orgin_sensor))
      ped_a = np.degrees(np.arctan2(ped_pos[1]-orgin_sensor[1], ped_pos[0]-orgin_sensor[0]))
      noisy_r = ped_r + np.random.uniform(-dR/2, dR/2)*0.96
      noisy_a = ped_a + np.random.uniform(-dA/2, dA/2)*0.96
      
      # Add the sensor position to properly position in global coordinates
      noisy_x = orgin_sensor[0] + noisy_r * np.cos(np.radians(noisy_a))
      noisy_y = orgin_sensor[1] + noisy_r * np.sin(np.radians(noisy_a))
      
    #   logger.critical(f"The distance of measurement from gt is {np.linalg.norm(np.array(ped_pos) - np.array((noisy_x, noisy_y)))}")
      meas_set = vertices_meas_range_bearing(dR=dR, dA=dA, pos_sens=orgin_sensor, pos_obj=(noisy_x, noisy_y))
      poly = Polygon(meas_set)

      # Check if the pedestrian is inside the polygon
      if check_point_in_polygon(meas_set, vru['pos']) is False:
        raise ValueError(f"Pedestrain  P{vru['id']} is outside the polygon {ped_id}")
      
      if np.random.rand()< percent_faulty or dt<0.4: # 50% chance to add noise
        
        if plot_set:
        #   renderer.ax.scatter(noisy_x, noisy_y, color=color_dict[id_sens], label='Target Path', zorder=2000, s=5)
          renderer.ax.plot(meas_set[:, 0], meas_set[:, 1], color=color_dict[id_sens], label='Target Path', zorder=2000, linewidth=1, ls ='--')
        #   logger.critical(f"the area of sensor {id_sens} polygon is {poly.area:.4f} m^2")
        #   renderer.ax.text( noisy_x + 0.2, noisy_y + 0.2, str(counter + id_sens), fontsize=8, color='black', zorder=3000   , ha='center', va='center') 
          
        measurements[counter + id_sens] = {
            'pos': ped_pos,
            'meas_id': counter + id_sens,
            'head': ped_head,
            'vel': ped_vel,
            'meas_set': meas_set,
            'noisy_x': noisy_x,
            'noisy_y': noisy_y,
            'r': noisy_r,
            'a': noisy_a,
            'dt':dt,
            'gt_id': ped_id,
            'sensor_id': id_sens,
            'track_id': None,
        }
      counter += 1
  return counter


def generate_clutter(dt: float, id_sens: int, measurements: dict, counter: int, lambda_clutter: float = 3.0, renderer: any = None, 
                     v_max_clutter: float = 1.5, ray_cfg: dict = None,sensor_pos: Tuple[float, float] = None, plot_set:bool =False) -> int:
    """Genera clutter in un settore (definito da ray_cfg) o in tutta la mappa se ray_cfg è None."""
    
    if ray_cfg:
        ox, oy = ray_cfg['origin']
        R_max = ray_cfg['ray_length']
        half_fan = ray_cfg['fan_deg'] / 2.0
        theta_min, theta_max = ray_cfg['heading'] - half_fan, ray_cfg['heading'] + half_fan
    else:
        ox, oy = sensor_pos
        R_max = max(MAP_MAX_X - MAP_MIN_X, MAP_MAX_Y - MAP_MIN_Y) * 1.5
        theta_min, theta_max = -180.0, 180.0

    for _ in range(np.random.poisson(lambda_clutter)):
        u = np.random.rand()
        r = R_max * np.sqrt(u)
        theta = np.random.uniform(theta_min, theta_max)
        x, y = ox + r * np.cos(np.radians(theta)), oy + r * np.sin(np.radians(theta))

        if ray_cfg is None and not (MAP_MIN_X <= x <= MAP_MAX_X and MAP_MIN_Y <= y <= MAP_MAX_Y):
            continue

        head = np.random.uniform(0.0, 360.0)
        vel = np.random.uniform(0.0, v_max_clutter)
        meas_set = vertices_meas_range_bearing(dR=1, dA=5, pos_sens=sensor_pos, pos_obj=(x, y))

        measurements[counter + id_sens] = {
            'pos': (x, y), 'head': head, 'vel': vel, 'meas_set': meas_set, 'meas_id': counter + id_sens,
            'noisy_x': x, 'noisy_y': y, 'r': r, 'a': theta, 'dt': dt,
            'gt_id': -1000+counter, 'sensor_id': id_sens, 'track_id': None
        }
        counter += 1

        if plot_set:
        #   renderer.ax.scatter(x, y, color=color_dict[id_sens], label='Target Path', zorder=2000, s=5)
          renderer.ax.plot(meas_set[:, 0], meas_set[:, 1], color=color_dict[id_sens], label='Target Path', zorder=2000, linewidth=.5, ls ='--')
          print(f"Clutter point generated at ({x}, {y}) with heading {head} and velocity {vel}")
    return counter

# After performing data association and before visualizing
def check_pedestrians_in_intersections(vru_list, ped_seen, ass_list, SME_dict, measurements,tot_error,dt):
    # Track which pedestrians are covered by intersections
    covered_pedestrians = set()
    mature1 = False
    mature2 = False
    # For each association pair, check if any pedestrians are in their intersection
    for pair in ass_list:
        # Get the polygons from the tracks
        poly1 = Polygon(SME_dict[pair[0]].Xc)
        mature1 = SME_dict[pair[0]].mature_track
        poly2 = Polygon(SME_dict[pair[1]].Xc)
        mature2 = SME_dict[pair[1]].mature_track
        
        # Calculate the intersection
        intersection_poly = poly1.intersection(poly2)
        
        # Skip if the intersection is empty
        if intersection_poly.is_empty or intersection_poly.area < 0.01:
            continue
            
        # Check which pedestrians are in this intersection
        for ped in vru_list:
            if ped['id'] in ped_seen:
                ped_point = Point(ped['pos'])
                if intersection_poly.contains(ped_point):
                    covered_pedestrians.add(ped['id'])
                    # logger.info(f"Pedestrian {ped['id']} is in the intersection between tracks {track_id1} and {track_id2}")
    
    # Now check which pedestrians were seen but not covered by any intersection
    seen_but_not_covered = set(ped_seen) - covered_pedestrians
    current_meas = { key: value for key, value in measurements.items() if value['dt'] == dt}

    for ped in seen_but_not_covered:
        count = sum(1 for m in current_meas.values() if m['gt_id'] == ped)
        if count>1 and mature1 and mature2:
            logger.warning(f"Pedestrians {seen_but_not_covered} were seen but not covered by any intersection, count: {count}")
            tot_error += 1
            logger.debug(f"Total error count: {tot_error}")
        
        
    return tot_error

def get_position_at_time(
    ID_pedestrian: int = 0,
    t: float = 0.0,
    off_x: float = 15.0,
    off_y: float = -2.0,
    csv_filename: str = "crowds_trajectories.csv",
) -> Optional[Tuple[float, float, float, float]]:
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
    heading_data = (
        ped_df["heading"].values.astype(float)
        if "heading" in ped_df.columns
        else np.arctan2(np.gradient(y_data), np.gradient(x_data))
    )
    velocity_data = (
        ped_df["velocity"].values.astype(float)
        if "velocity" in ped_df.columns
        else np.sqrt(np.gradient(x_data) ** 2 + np.gradient(y_data) ** 2)
    )

    # Check if t is within the time range
    if t < times[0] or t > times[-1]:
        return None

    # Interpolate x, y, heading, and velocity at time t
    x = np.interp(t, times, x_data) + off_x
    y = np.interp(t, times, y_data) + off_y
    heading = np.interp(t, times, heading_data)
    velocity = np.interp(t, times, velocity_data)

    return x, y, heading, velocity


def get_full_trajectory(
    ID_pedestrian: int,
    off_x: float = 15.0,
    off_y: float = -2.0,
    csv_filename: str = "crowds_trajectories.csv",
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Returns the full trajectory (x, y coordinates) for a given pedestrian.

    Parameters:
        ID_pedestrian (int): The ID of the pedestrian.
        off_x (float): Offset to add to x coordinate.
        off_y (float): Offset to add to y coordinate.
        csv_filename (str): The CSV file containing trajectory data.

    Returns:
        tuple: (x_coords, y_coords) arrays of the full trajectory
        None: if no data is available for the pedestrian.
    """
    # Load the CSV data
    df = pd.read_csv(csv_filename)

    # Filter for the specified pedestrian and sort by time
    ped_df = df[df["agent_id"] == ID_pedestrian].sort_values("time")

    if ped_df.empty:
        return None

    # Extract the position data
    x_data = ped_df["pos_x"].values.astype(float) + off_x
    y_data = ped_df["pos_y"].values.astype(float) + off_y

    return x_data, y_data


def get_trajectory_segments(
    ID_pedestrian: int,
    current_time: float,
    future_duration: float = 0.8,
    off_x: float = 15.0,
    off_y: float = -2.0,
    csv_filename: str = "crowds_trajectories.csv",
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Returns past and future trajectory segments for a given pedestrian at a specific time.

    Parameters:
        ID_pedestrian (int): The ID of the pedestrian.
        current_time (float): Current simulation time.
        future_duration (float): Duration in seconds to look ahead for future trajectory.
        off_x (float): Offset to add to x coordinate.
        off_y (float): Offset to add to y coordinate.
        csv_filename (str): The CSV file containing trajectory data.

    Returns:
        tuple: (past_x, past_y, future_x, future_y) arrays of trajectory segments
        None: if no data is available for the pedestrian.
    """
    # Load the CSV data
    df = pd.read_csv(csv_filename)

    # Filter for the specified pedestrian and sort by time
    ped_df = df[df["agent_id"] == ID_pedestrian].sort_values("time")

    if ped_df.empty:
        return None

    # Extract time and position data
    times = ped_df["time"].values.astype(float)
    x_data = ped_df["pos_x"].values.astype(float) + off_x
    y_data = ped_df["pos_y"].values.astype(float) + off_y

    # Split into past and future based on current time
    past_mask = times <= current_time
    future_mask = (times >= current_time) & (times <= current_time + future_duration)

    past_x = x_data[past_mask]
    past_y = y_data[past_mask]
    future_x = x_data[future_mask]
    future_y = y_data[future_mask]

    # If there's no exact match for current_time, interpolate the current position
    # and ensure it's included in both segments for continuity
    if current_time not in times:
        # Get current position by interpolation
        if current_time >= times[0] and current_time <= times[-1]:
            current_x = np.interp(current_time, times, x_data)
            current_y = np.interp(current_time, times, y_data)

            # Add current position to the end of past trajectory if not already there
            if len(past_x) == 0 or past_x[-1] != current_x or past_y[-1] != current_y:
                past_x = np.append(past_x, current_x)
                past_y = np.append(past_y, current_y)

            # Add current position to the beginning of future trajectory if not already there
            if (
                len(future_x) == 0
                or future_x[0] != current_x
                or future_y[0] != current_y
            ):
                future_x = np.insert(future_x, 0, current_x)
                future_y = np.insert(future_y, 0, current_y)

    return past_x, past_y, future_x, future_y
# Call the function after association

if __name__ == "__main__":
    pass