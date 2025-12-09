# Parameters:
alpha_ray: float = 0.0
import numpy as np

color_dict = {1000: "#01bf62", 2000: "yellow", 3000: "#ff914d"}


# 1. ROAD GEOMETRY PARAMETERS
# -------------------------
LANE_WIDTH     = 3.75    # Width of each lane in meters
SHOULDER_WIDTH = 1.5     # Width of road shoulder in meters
NUM_LANES      = 2       # Number of lanes
ROAD_START_X   = 0       # Starting x-coordinate of road
ROAD_START_Y   = 0       # Starting y-coordinate of road
ROAD_LENGTH    = 40      # Length of road in meters

# 2. VEHICLE DIMENSIONS
# ------------------
VEH_LENGTH     = 8       # Length of vehicles in meters
VEH_WIDTH      = 3       # Width of vehicles in meters

MAP_MIN_X = 0
MAP_MAX_X = 40
MAP_MIN_Y = -10
MAP_MAX_Y = 10





# 3. STATIC OBJECT POSITIONS
# -----------------------
# Format: (x, y) coordinates in meters
CV_pos   = (33,    2.5)  # Connected Vehicle position
PV_POS   = (15,  -5.85)  # Passenger Vehicle position
VRU_pos1 = (19.5,-6.45) 
VRU_pos2 = (23.5,-2)  
VRU_pos3 = (23.5,   -8)
EV_pos =   (5,    -2.5)
RSU_POS  = (20,    5.5)
RSU_POS2  = (22,    -9)

dR1 = 1   # Note that the error is expressed in r_0+-dR1 and a_0 +-dA1
dA1 = 7
  
# 4. RAYCASTING PARAMETERS
# ---------------------
# 4.1 EV Raycasting Configuration
# Vehicles the EV sees
vehicles_ev_see = [
    # { 'pos': (CV_pos[0] - VEH_LENGTH/2, CV_pos[1] - VEH_WIDTH/2), 'width': VEH_LENGTH, 'height': VEH_WIDTH },
    { 'pos': (PV_POS[0] - VEH_LENGTH/2, PV_POS[1] - VEH_WIDTH/2), 'width': VEH_LENGTH, 'height': VEH_WIDTH },
]


# Vehicles the CV sees
vehicles_rsu_see = [
    { 'pos': (PV_POS[0] - VEH_LENGTH/2, PV_POS[1] - VEH_WIDTH/2), 'width': VEH_LENGTH, 'height': VEH_WIDTH },
    { 'pos': (EV_pos[0] - VEH_LENGTH/2, EV_pos[1] - VEH_WIDTH/2),'width': VEH_LENGTH, 'height': VEH_WIDTH }
]

RSU1_RAY_CONFIG = {
  'origin':  RSU_POS + np.array([1, 0]),
  'ray_length': 18,
  'n_rays':  180,
  'heading': 270,
  'fan_deg': 100,
  'color': '#01bf62',
}

RSU3_RAY_CONFIG = {
  'origin':  RSU_POS2 + np.array([1, 0.5]),
  'ray_length': 18,
  'n_rays':  200,
  'heading': 90 ,
  'fan_deg': 150,
  'color': '#ff914d',
}
