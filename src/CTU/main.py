# 1. IMPORTS AND SETUP
# --------------------
import sys, time, random, logging
import numpy as np
import os
from constants import *
from SBE_module.SME_module import SMEModule
from map_renderer import MapRenderer
from TrackManager import TrackManager
from PostAssociation import PostAssociation
from utils import (
    get_sensor_list, check_point_in_polygon, generate_clutter,
    check_pedestrians_in_intersections, get_position_at_time, get_trajectory_segments
)

# Data association method: Hungarian + blacklist heuristic
from DataAssociationGreedy import DataAssociationManager as DataAssociationManagerGreedy
from DataAssociation import DataAssociationManager as DataAssociationManagerNew

# Logging configuration
logging.basicConfig(
    level=logging.DEBUG,
    handlers=[
        logging.FileHandler("log.txt", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
random_var = 2
random.seed(random_var)  # Set a seed for reproducibility
np.random.seed(random_var)  # Set a seed for reproducibility


def Main(greedy: bool = False, heading_error_degrees: float = 8) -> int:
    # random.seed(42)  # Set a seed for reproducibility
    """
    2. SIMULATION CONFIGURATION
    """

    # -------------------------
    # 2.1. Time Configuration
    time_steps: np.NDArray[np.float64] = np.linspace(
        0, 8, 81
    )  # 15 seconds simulation with 151 steps
    tot_error: int = 0

    # 2.2. Visualization Settings
    RECORDING: bool = False  # Enable frame recording
    log_data: bool = False  # Log data to CSV file
    RENDER_PLOT: bool = True  # Render the frames
    time_sleep = 0.01  # Inter-frame delay
    manual_mode = 9  # Frame-by-frame mode (-1 for auto)
    windows_scale = 0.6  # Scale the window size

    lambda_clutter = 0  # Clutter rate

    dR1 = 1  # Note that the error is expressed in r_0+-dR1 and a_0 +-dA1
    dA1 = 8
    perc_working = 2
    heading_error_degrees = 10
    velocity_error_speed = 1

    blacklist = set()
    counter: int = 0

    peds = [1, 2, 3, 4, 5, 6, 9]
    # peds = [2, 3,3,4,5]  # The one not crossing the intersection and realistic

    # ------------------------------------------------
    # Create Renderer
    # ------------------------------------------------
    # Dopo aver creato il renderer:
    renderer = MapRenderer()
    filename = renderer.init_csv("sim_log.csv", log_data=log_data)
    id_counter = 0

    # Add frame timing variable
    list_ev_hits: list[dict[str, tuple[np.float64, np.float64]]] = (
        []
    )  # list of hitted pedestrian, dict compoes of [{'pos': (np.float64(x), np.float64(y)), 'id': 4}]
    list_rsu_hits: list[dict[str, tuple[np.float64, np.float64]]] = []
    SME_dict: dict[int, SMEModule] = {}

    measurements = {}
    frame_duration: float = 0
    
    csv_traj = os.path.join(os.path.dirname(__file__), 'crowds_trajectories.csv')
    print(f"file path: {csv_traj}")
    
    """
    3. SIMULATION LOOP
    """
    EV_pos = (5, -2.5)
    use_greedy: bool = greedy

    # ----------------
    for dt in time_steps:
      dt = np.round(dt, 2)
      # 3.1. Environment Setup
      logger.info(
          f"--------------------dt:{dt} (previous frame: {frame_duration:.3f}s)--------------------"
      )
      print(dt)
      renderer.clear_scene()
      renderer.draw_road(ROAD_START_X, ROAD_START_Y, ROAD_LENGTH, LANE_WIDTH, NUM_LANES, SHOULDER_WIDTH)

      # 3.2. Static Object Rendering
      renderer.draw_vehicle(PV_POS, VEH_LENGTH, VEH_WIDTH, "lightgray", "PV", label_color="black")
      renderer.draw_rsu(RSU_POS, 2, 1, "#01bf62", "RSU1")
      renderer.draw_rsu(RSU_POS2, 2, 1, "#ff914d", "RSU2")

      offset = [(18.5, -9.5)] * 13
      vru_list = []

      # Draw pedestrian trajectories (past as dotted, future 0.8s as solid) in black
      for i in peds:
          trajectory_segments = get_trajectory_segments( i, dt, future_duration=0.8, off_x=offset[i - 1][0], 
                                                        off_y=offset[i - 1][1],csv_filename = csv_traj)
          if trajectory_segments is not None:
              past_x, past_y, future_x, future_y = trajectory_segments
              renderer.draw_trajectory_segments( past_x, past_y, future_x, future_y, color="black", 
                                                linewidth=1.5, alpha=0.7)

      # loop for drawing pedestrians and collecting their data
      for i in peds:
          pos_data = get_position_at_time(i, dt, off_x=offset[i - 1][0], off_y=offset[i - 1][1], csv_filename=csv_traj)  
          if pos_data is not None:
              pos = pos_data[:2]  # Estrae solo (x, y)
              heading, velocity = pos_data[2:]
              vru_list.append({"pos": pos, "id": i, "head": heading, "vel": velocity})
              if i == 10:
                  print(
                      f"Pedestrian {i} position: {pos}, heading: {heading}, velocity: {velocity}"
                  )
              renderer.draw_pedestrian(pos, "red", f"P{i-1}")

      """
      RAYCASTING
      """
      renderer.draw_vehicle(EV_pos, VEH_LENGTH, VEH_WIDTH, "lightblue", "EV", label_color="black")

      list_rsu_hits = renderer.draw_rays(ray_cfg=RSU1_RAY_CONFIG, vru_objects=vru_list)  
      list_rsu2_hits = renderer.draw_rays(ray_cfg=RSU3_RAY_CONFIG, vru_objects=vru_list)

      # Getting the IDs of the pedestrians seen by the RSU and CV
      sorted_rsu_hits = sorted(list_rsu_hits, key=lambda hit: hit["id"])
      ped_seen_rsu = [k["id"] for k in sorted_rsu_hits]
      sorted_rsu2_hits = sorted(list_rsu2_hits, key=lambda hit: hit["id"])
      ped_seen_rsu2 = [k["id"] for k in sorted_rsu2_hits]
      ped_seen_both = sorted(set(ped_seen_rsu) & set(ped_seen_rsu2))


      """
      SENSORS UPDATE
      """
      # Get the sensor list for RSU1
      # We are adding to the measurement dict the measurement from the RSU.
      counter = get_sensor_list(
          sensor_pos=RSU_POS, sorted_hits=sorted_rsu_hits,
          dR=dR1, dA=dA1, id_sens=1000, renderer=renderer, dt=dt,
          measurements=measurements, percent_faulty=perc_working,
          plot_set=False, counter=counter)

      # Get the sensor list for RSU2
      counter = get_sensor_list(
          sensor_pos=RSU_POS2, sorted_hits=sorted_rsu2_hits,
          dR=dR1, dA=dA1, id_sens=3000, renderer=renderer, dt=dt,
          measurements=measurements, percent_faulty=perc_working,
          plot_set=False, counter=counter)    

      counter = generate_clutter(
          dt=dt, id_sens=3000, measurements=measurements,
          counter=counter, sensor_pos=RSU_POS2,
          lambda_clutter=lambda_clutter,
          v_max_clutter=1.5, ray_cfg=RSU3_RAY_CONFIG,
          plot_set=True, renderer=renderer)
       

      counter = generate_clutter(
          dt=dt, id_sens=1000, measurements=measurements,
          counter=counter, sensor_pos=RSU_POS,
          lambda_clutter=lambda_clutter,
          v_max_clutter=1.5,ray_cfg=RSU1_RAY_CONFIG,
          plot_set=True,renderer=renderer)

      track_manager = TrackManager(
          dA1=dA1, dR1=dR1, measurements=measurements,
          sme_dict=SME_dict, verbose=False,
          renderer=renderer, id_counter=id_counter,
          color_dict=color_dict, err_vel=velocity_error_speed,
          err_heading=heading_error_degrees,
      )

      SME_dict, measurements, id_counter = track_manager.process_measurements(dt)

      # Check that the pedestrian is inside the polygon
      for ped in vru_list:
          # Only check pedestrians that are in the ped_seen_both list
          if ped["id"] in ped_seen_both:
              if not any(
                  check_point_in_polygon(est.Xp, ped["pos"])
                  for est in SME_dict.values()
              ):
                  logging.error(f"Pedestrian {ped['id']} is outside all polygons")

      mature_tracks = {(k, v) for k, v in SME_dict.items() if v.mature_track}


      """
      DATA ASSOCIATION
      """
      mature_track_keys = [k for k, v in SME_dict.items() if v.mature_track]
      if use_greedy:
          assoc = DataAssociationManagerGreedy(
              id_sens_1=1000,
              id_sens_2=3000,
              measurements=measurements,
              sme_dict=SME_dict,
              blacklist=blacklist,
              mature_tracks_id=mature_track_keys,
              dt=dt,
          )
      else:
          assoc = DataAssociationManagerNew(
              id_sens_1=1000,
              id_sens_2=3000,
              measurements=measurements,
              sme_dict=SME_dict,
              blacklist=blacklist,
              mature_tracks_id=mature_track_keys,
              dt=dt,
          )

      blacklist = assoc.compute_cost_matrix(verbose=False)

      logger.debug(f"Blacklist: {[tuple(sorted(fs)) for fs in blacklist]}")

      ass_list = assoc.solve(verbose=False)

      # Add this line:
      if dt > 0.2:
          tot_error = check_pedestrians_in_intersections(
              vru_list, ped_seen_both, ass_list, SME_dict, measurements, tot_error, dt
          )

      """
      VISUALISE THE ASSOCIATION. 
      """
      # instantiate once per frame
      post = PostAssociation(
          SME_dict=SME_dict,
          renderer=renderer,
          dA=dA1,
          dR=dR1,
          max_acc=0.5,
          max_vel=1.8,
      )

      # draw raw intersections
      post.visualize_intersections(ass_list)
      
      
      # draw reachable‐set expansions
      post.visualize_predictions(ass_list, dt)

      """
      RENDER THE GLOBAL PREDICTION
      """

      # renderer.log_to_csv(filename, data)
      if RENDER_PLOT:
          renderer.render_frame(
              x_limits=(18, 34),
              y_limits=(-9, 0),
              title=f"{dt}",
              xlabel="$X [m]$",
              ylabel="$Y [m]$",
              scale=windows_scale,
              recording=RECORDING,
              save_axes=True,
              dt=dt,
              screen_dt=7,
          )

      # 10) Pause for a short while
      if dt > manual_mode and manual_mode != -1:
          input(f"Press enter for the next step... {dt}")
      else:
          # print(f"Time: {dt}")
          time_sleep = 0 if RENDER_PLOT == False else time_sleep

          time.sleep(time_sleep)
        #
        
    # Done animating
    renderer.finalize()

    logger.debug(f"total number of errors: {tot_error}")
    return tot_error


if __name__ == "__main__":
    Main()
