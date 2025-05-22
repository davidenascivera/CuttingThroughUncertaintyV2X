# 1. IMPORTS AND SETUP
# --------------------
import sys, time, random, logging
from typing import Tuple

import numpy as np
import pandas as pd
from shapely.geometry import Polygon, Point

from src.CTU.constants import *
from SME_mine.SME_module import SMEModule
from map_renderer import MapRenderer
from TrackManager import TrackManager
from dataset.pedestrian_loader import get_position_at_time

from utils import (
    lin_velocity, get_sensor_list, check_point_in_polygon,
    generate_clutter, check_pedestrians_in_intersections
)

# Data association method: Hungarian + blacklist heuristic
# from DataAssociationGreedy import DataAssociationManager
from DataAssociation import DataAssociationManager

# Logging configuration
logging.basicConfig(
    level=logging.DEBUG,
    handlers=[logging.FileHandler('log.txt', mode='w'),
              logging.StreamHandler(sys.stdout)]
)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def Main():
  # random.seed(42)  # Set a seed for reproducibility
  '''
  2. SIMULATION CONFIGURATION
  
  '''

  # -------------------------
  # 2.1. Time Configuration
  time_steps:np.NDArray[np.float64] = np.linspace(0, 11, 111)  # 15 seconds simulation with 151 steps
  EV_present:bool = False 
  alpha_ray:float = 0.08
  tot_error:int = 0

  # 2.2. Visualization Settings
  RECORDING: bool = False        # Enable frame recording
  log_data: bool = False         # Log data to CSV file
  RENDER_PLOT: bool = True           # Render the frames
  time_sleep = 0.01        # Inter-frame delay
  manual_mode = -1     # Frame-by-frame mode (-1 for auto)
  windows_scale = .35     # Scale the window size

  dR1 = 1  # Note that the error is expressed in r_0+-dR1 and a_0 +-dA1
  dA1 = 6
  blacklist = set()
  counter:int = 0

  peds = [1,2,3,4,5,6,7,8,9]

  # ------------------------------------------------
  # Create Renderer
  # ------------------------------------------------
  # Dopo aver creato il renderer:
  renderer = MapRenderer()
  id_counter = 0

  # Add frame timing variable
  list_ev_hits: list[dict[str, tuple[np.float64, np.float64]]] = [] # list of hitted pedestrian, dict compoes of [{'pos': (np.float64(x), np.float64(y)), 'id': 4}]
  list_rsu_hits: list[dict[str, tuple[np.float64, np.float64]]] = []
  SME_dict: dict[int, SMEModule] = {}

  measurements = {}
  frame_duration:float = 0
  color_dict = {1000:'green', 2000:'yellow', 3000:'blue'}
  '''
  # 3. SIMULATION LOOP
  '''
  EV_pos =   (5,    -2.5)
  # ----------------
  for dt in time_steps:
    
    
    # Start timing new frame
    frame_start_time: time = time.time()
    
    dt = np.round(dt,2)
    # 3.1. Environment Setup
    logger.info(f"--------------------dt:{dt} (previous frame: {frame_duration:.3f}s)--------------------")
    
    renderer.clear_scene()
    y_top_edge, y_bottom_edge = renderer.draw_road(ROAD_START_X, ROAD_START_Y, 
                                                  ROAD_LENGTH, LANE_WIDTH, 
                                                  NUM_LANES, SHOULDER_WIDTH)

    # 3.2. Static Object Rendering
    renderer.draw_vehicle(PV_POS, VEH_LENGTH, VEH_WIDTH, 'lightblue', 'PV (2)', label_color='black')
    
    renderer.draw_rsu(RSU_POS, 2, 1, 'darkgreen', 'RSU(1)')
    renderer.draw_rsu(RSU_POS2, 2, 1, 'blue', 'RSU(3)')

    
    offset = [(19, -10), (30, -10), (19, -10), (19, -10), (19, -10), (19, -10), (19, -10), (24, 1), (19, -8.5), (30, -10), (19, -10), (16, -10), (19, -8.5)]
    offset = [(19, -10)] * 13
    # offset[1]= (19, -10.4)

    vru_list = []
    # for i in [1,2,3,5]:  # Loop per gli ID dei pedoni da 1 a 4
    for i in peds:
      pos_data = get_position_at_time(i, dt, off_x=offset[i-1][0], off_y=offset[i-1][1])
      # if pos_data is None:
      #     raise ValueError(f"Pedestrian {i} has missing data at time {dt}")
      if pos_data is not None:
        pos = pos_data[:2]  # Estrae solo (x, y)
        heading, velocity = pos_data[2:] 
        vru_list.append({'pos': pos, 
                            'id': i, 
                            'head': heading, 
                            'vel':velocity })
        if i ==10:
          print(f"Pedestrian {i} position: {pos}, heading: {heading}, velocity: {velocity}")
        renderer.draw_pedestrian(pos, 'orange', f'P{i}')
        
  
    '''
      RAYCASTING
    '''

    # 3.4. Sensor Simulation
    # 3.4.1. EV Sensor Processing
    if EV_present:  
      EV_pos = lin_velocity(EV_pos,0,dt, 15) #(pos, start, time_steps, velocity)
      # EV_pos =   (10,    -2.5)
      renderer.draw_vehicle(EV_pos, VEH_LENGTH, VEH_WIDTH, 'yellow', 'EV',label_color='black')
      list_ev_hits = renderer.draw_rays(ray_origin=EV_pos, ray_length=20, n_rays=300, heading=0, fan_deg=200, 
                                        vehicles=vehicles_ev_see, color='yellow', vru_objects=vru_list, alpha=alpha_ray,vru_occlusion_shadow=False)


    list_rsu_hits = renderer.draw_rays(ray_cfg=RSU1_RAY_CONFIG, vehicles=vehicles_rsu_see, vru_objects=vru_list, alpha=alpha_ray,vru_occlusion_shadow=False)
    
    list_rsu2_hits = renderer.draw_rays(ray_cfg=RSU3_RAY_CONFIG, vehicles=vehicles_rsu_see, vru_objects=vru_list, alpha=alpha_ray,vru_occlusion_shadow=False)
    
    
    # Getting the IDs of the pedestrians seen by the RSU and CV
    sorted_rsu_hits = sorted(list_rsu_hits, key=lambda hit: hit['id']) 
    ped_seen_rsu = [k['id'] for k in sorted_rsu_hits]
    sorted_rsu2_hits = sorted(list_rsu2_hits, key=lambda hit: hit['id']) 
    ped_seen_rsu2 = [k['id'] for k in sorted_rsu2_hits]
    ped_seen_both = sorted(set(ped_seen_rsu) & set(ped_seen_rsu2))

    '''
      SENSORS UPDATE
    '''
    # Get the sensor list for RSU1 
    # We are adding to the measurement dict the measurement from the RSU. 
    counter = get_sensor_list(sensor_pos=RSU_POS, sorted_hits=sorted_rsu_hits, dR=dR1, dA=dA1, id_sens=1000, renderer=renderer, dt=dt, 
                    measurements = measurements, percent_faulty=85, plot_set=True,colors= color_dict,counter=counter)
    

    # Get the sensor list for EV
    # get_sensor_list(sensor_pos=EV_pos, sorted_hits=sorted_cv_hits, dR=dR1, dA=dA1, id_sens=2000, renderer=renderer, dt=dt,
    #                 measurements = measurements, percent_faulty=2,plot_set=True, color_set='yellow')
    # Get the sensor list for RSU2
    counter = get_sensor_list(sensor_pos=RSU_POS2, sorted_hits=sorted_rsu2_hits, dR=dR1, dA=dA1, id_sens=3000, renderer=renderer, dt=dt, 
                     measurements = measurements, percent_faulty=85, plot_set=True, colors=color_dict,counter=counter)

    counter = generate_clutter(dt=dt, id_sens=3000, measurements=measurements, counter=counter,sensor_pos=RSU_POS2, 
                               lambda_clutter=1, v_max_clutter=1.5,ray_cfg=RSU3_RAY_CONFIG)
    
    counter = generate_clutter(dt=dt, id_sens=1000, measurements=measurements, counter=counter,sensor_pos=RSU_POS2, 
                               lambda_clutter=1, v_max_clutter=1.5,ray_cfg=RSU3_RAY_CONFIG)
        
    track_manager = TrackManager(dA1=dA1, dR1=dR1,measurements=measurements, sme_dict=SME_dict, 
                                 verbose=False, renderer=renderer, id_counter=id_counter, color_dict=color_dict)
    
    SME_dict, measurements, id_counter = track_manager.process_measurements(dt)

    # Check that the pedestrian is inside the polygon
    for ped in vru_list:
        # Only check pedestrians that are in the ped_seen_both list
        if ped['id'] in ped_seen_both:
            if not any(check_point_in_polygon(est.Xp, ped['pos']) for est in SME_dict.values()):
                logging.error(f"Pedestrian {ped['id']} is outside all polygons")

    mature_tracks = {(k,v) for k,v in SME_dict.items() if v.mature_track}
    
    # for (key,val) in mature_tracks:
    #   print(f"Track id: {key} - Sensor id: {val.sensor_id} - Ped id: {val.id} ")


    # print(f"tentative tracks: {[v.id for v in tentative_tracks]}")

    '''
      DATA ASSOCIATION
    '''
    mature_track_keys = [k for k, v in SME_dict.items() if v.mature_track]
    assoc = DataAssociationManager(id_sens_1=1000,id_sens_2=3000, measurements=measurements, sme_dict=SME_dict, blacklist=blacklist, mature_tracks_id=mature_track_keys, dt=dt)
    blacklist = assoc.compute_cost_matrix(verbose=False)
    # logger.debug(f"Blacklist: {[tuple(sorted(fs)) for fs in blacklist]}")

    ass_list1 = assoc.solve(verbose=False)
    assoc.validate()
    # print(f"the ass list is {ass_list1}")
    
    # assoc2 = DataAssociationManager(id_sens_1=1,id_sens_2=3, measurements=measurements, sme_dict=SME_dict, blacklist=blacklist)
    # assoc2.compute_cost_matrix(verbose=False)
    # ass_list2 = assoc2.solve(verbose=False)
    # assoc2.validate()
    # ass_list = [] 
    ass_list = ass_list1 
    
    # Add this line:
    if dt >0.2:
      tot_error = check_pedestrians_in_intersections(vru_list, ped_seen_both, ass_list, SME_dict, measurements,tot_error)

    '''
      VISUALISE THE ASSOCIATION. 
    '''
    for pair in ass_list:
      poly1 = Polygon(SME_dict[pair[0]].Xc)
      poly2 = Polygon(SME_dict[pair[1]].Xc)
      intersection = np.array(poly1.intersection(poly2).exterior.coords)
      renderer.ax.plot(intersection[:, 0], intersection[:, 1], color='orange', label='Target Path', zorder=100000, linewidth=2)
      # data.append(pair[0])
      # data.append(poly1.intersection(poly2).area)
      
    
    frame_duration: time = time.time() - frame_start_time
    
    # renderer.log_to_csv(filename, data) 
    if RENDER_PLOT:
      renderer.render_frame(  
          # x_limits=(-2, 44),
          # y_limits=(-10, 10),
          x_limits=(10, 35 ),
          y_limits=(-10,-3),
          title = f"V2X Collaborative Perception Scenario {dt}",
          xlabel ='x [Global]',
          ylabel ='y [Global]',
          scale=windows_scale,
          recording=RECORDING,
          dt=dt,
          screen_dt=7
      )
    
    # Calculate previous frame duration
    
    # 10) Pause for a short while
    if dt > manual_mode and manual_mode != -1:
      input(f"Press enter for the next step... {dt}")
    else:
      # print(f"Time: {dt}")
      time_sleep = 0 if RENDER_PLOT == False else time_sleep
      
      time.sleep(time_sleep)
    #
  # Done animating
  # renderer.finalize()
  
  logger.debug(f"total number of errors: {tot_error}")
  return tot_error

if __name__ == "__main__":
    Main()
