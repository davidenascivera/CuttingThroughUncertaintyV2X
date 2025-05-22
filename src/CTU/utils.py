import numpy as np
import json, random
from shapely.geometry import Polygon, Point
from typing import Tuple, List
from matplotlib.patches import Ellipse
import os
from SME_mine.RangeBearingSensor.RangeBearing import vertices_meas_range_bearing
from src.CTU.constants import MAP_MIN_X, MAP_MIN_Y, MAP_MAX_X, MAP_MAX_Y

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
                    renderer:any, dt:float,percent_faulty:float, measurements = {}, plot_set: bool=False, counter:int = 0, 
                    colors:dict[int,str]= {} ) -> None:
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
      
      
      meas_set = vertices_meas_range_bearing(dR=dR, dA=dA, pos_sens=orgin_sensor, pos_obj=(noisy_x, noisy_y))
      
      # Check if the pedestrian is inside the polygon
      if check_point_in_polygon(meas_set, vru['pos']) is False:
        raise ValueError(f"Pedestrain  P{vru['id']} is outside the polygon {ped_id}")
      
      if np.random.rand()< percent_faulty or dt<0.4: # 50% chance to add noise
        
        if plot_set:
          renderer.ax.scatter(noisy_x, noisy_y, color=colors[id_sens], label='Target Path', zorder=2000, s=5)
          renderer.ax.plot(meas_set[:, 0], meas_set[:, 1], color=colors[id_sens], label='Target Path', zorder=2000, linewidth=.5, ls ='--')
          # renderer.ax.text( noisy_x + 0.2, noisy_y + 0.2, str(ped_id), fontsize=8, color='black', zorder=3000   , ha='center', va='center') 
          
        measurements[counter + id_sens] = {
            'pos': ped_pos,
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


def generate_clutter(dt: float, id_sens: int, measurements: dict, counter: int, lambda_clutter: float = 3.0, 
                     v_max_clutter: float = 1.5, ray_cfg: dict = None,sensor_pos: Tuple[float, float] = None) -> int:
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
            'pos': (x, y), 'head': head, 'vel': vel, 'meas_set': meas_set,
            'noisy_x': x, 'noisy_y': y, 'r': r, 'a': theta, 'dt': dt,
            'gt_id': -1000+counter, 'sensor_id': id_sens, 'track_id': None
        }
        counter += 1

    return counter

# After performing data association and before visualizing
def check_pedestrians_in_intersections(vru_list, ped_seen, ass_list, SME_dict, measurements,tot_error):
    # Track which pedestrians are covered by intersections
    covered_pedestrians = set()
    
    # For each association pair, check if any pedestrians are in their intersection
    for pair in ass_list:
        # Get the polygons from the tracks
        poly1 = Polygon(SME_dict[pair[0]].Xc)
        poly2 = Polygon(SME_dict[pair[1]].Xc)
        
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
    if seen_but_not_covered:
        tot_error += len(seen_but_not_covered)
        logger.warning(f"Pedestrians {seen_but_not_covered} were seen but not covered by any intersection")
        logger.debug(f"Total error count: {tot_error}")

        
    return tot_error

# Call the function after association

if __name__ == "__main__":
    pass