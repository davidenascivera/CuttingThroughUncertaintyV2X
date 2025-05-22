import os 
os.environ["USE_NUMBA"] = "0"  # o "1"

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time


from utils import minkowski, minkowski_safe # We have 2 functions for the minkowski sum. "Minkowski" uses the O(m) alg, Minkowski_safe uses the O(m n) alg.
from utils import plot_poly, regular_polygon, reorder_polygon, get_annular_sector_points
from SoutherLand.vanilla_numba import sutherland_hodgman
# warmup_suth_numba() # Warmup the numba function

from SoutherLand.vanilla import sutherland_hodgman
from CuttingConvex.convex_int_numba import convex_intersect

def intersection_convex(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    '''
    Function that given the 2 polygons P and Q, returns the intersection of the two polygons.
    '''
    print("P")
    print(P)
    print("Q")
    print(Q)
    # Mulitpling by 1_000_000 because the function works with integers
    P_raw = P * 1_000_000
    Q_raw = Q * 1_000_000

    P = P_raw.flatten().astype(np.int64)
    Q = Q_raw.flatten().astype(np.int64)
    # Nota bene. Potrebbe essere che questa parte contenga un errore. in caso provare a rimuoreve il //2
    nP = len(P) // 2
    nQ = len(Q) // 2
    maxOut = 40  # maximum expected points
    outPoints_numba = np.zeros(2 * maxOut, dtype=np.float64)
    outPoints_numba, numPoints_numba = convex_intersect(P, nP, Q, nQ, maxOut)
    
    pts_numba = outPoints_numba[:2*numPoints_numba].reshape((numPoints_numba, 2))
    if numPoints_numba > 0:
        return pts_numba[2:] / 1_000_000
    else:
        print("No intersection found")
        return np.array([0,0])
    



# -----------------------------
# Define the two sets (polygons)
# --------------------- --------

# Parameters for the annular sector.
r_0: float = 1.5
delta_r: float = 0.6
delta_a: int = 15  # degrees
plot_intermediate: bool = False 
v_max:float = 1.8 # m/s

acceleration: float = 0.5 # m/s^2
dt: float = 0.1
future_steps: int = 5
time_horizon: float = future_steps * dt

bounded_vel = get_annular_sector_points(a_0=0, r_0=r_0, delta_a=delta_a, delta_r=delta_r, future_steps=time_horizon)

# Define the convex set square (vertices in counter-clockwise order).
X0_square: np.ndarray = np.array([
    [-0.5, 0.5],
    [-0.5, -0.5],
    [0.5, -0.5],
    [0.5, 0.5]
])

acc_displacement = 0.5 * time_horizon ** 2

# Define the acceleration displacement square.
acc_square: np.ndarray = np.array([
    [-acc_displacement / 2, acc_displacement / 2],
    [-acc_displacement / 2, -acc_displacement / 2],
    [acc_displacement / 2, -acc_displacement / 2],
    [acc_displacement / 2, acc_displacement / 2]
])


acc_square = regular_polygon(4, acc_displacement/2, np.array([0, 0]))
max_vel_radius = regular_polygon(6, v_max * time_horizon , np.array([0, 0]))

# -----------------------------
# Compute and plot the Minkowski Sum.
# -----------------------------
start = time.perf_counter()

# Step 1 - Compute the reachable set of the velocity
result_vel = minkowski(bounded_vel, X0_square, plot_intermediate = plot_intermediate)

# Step 2 - Compute the reachable set of the acceleration    
result = minkowski(result_vel, acc_square, plot_intermediate = plot_intermediate)

# Step 3 - Compute the reachable set of the max position given max velocity
max_vel = minkowski(max_vel_radius, X0_square, plot_intermediate = plot_intermediate)


end = time.perf_counter()

final = intersection_convex(result, max_vel)
final_sout = sutherland_hodgman(result, max_vel)

plt.figure(figsize=(8, 6))  # larghezza=8 pollici, altezza=6 pollic

plot_poly(max_vel_radius, 'g--', 2, 'Bounded Velocity')
# plot_poly(acc_square, 'p--', 2, 'Bounded Velocity')
plot_poly(max_vel, 'y-', 2, 'MaxVel Constraint')
plot_poly(final, 'r-', 2, 'Final Constraint',zorder = 2)
plot_poly(X0_square, 'r-', 2, 'X0_square')
plot_poly(result_vel, 'b-', 2, 'Velocity')
plot_poly(result, 'b-', 2, 'vel + acc')

plt.title("Final Minkowski Sum")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.axis('equal')
# plt.legend()
plt.show()

elapsed = end - start
print(f"High Resolution Elapsed Time: {elapsed*1_000} milliseconds")

X1_square = final

result_vel1 = minkowski(bounded_vel, X1_square, plot_intermediate = plot_intermediate)

result1 = minkowski(result_vel1, acc_square, plot_intermediate = plot_intermediate)

max_vel = minkowski(max_vel_radius, X1_square, plot_intermediate = plot_intermediate)

final1 = intersection_convex(result1, max_vel)



plt.figure(figsize=(10, 6))  # larghezza=8 pollici, altezza=6 pollic

plot_poly(max_vel, 'y-', 2, 'MaxVel Constraint')
plot_poly(final1, 'r-', 2, 'Final Constraint',zorder = 2)
plot_poly(X1_square, 'r-', 2, 'X0_square')
plot_poly(result_vel1, 'b-', 2, 'Velocity')
plot_poly(result1, 'b-', 2, 'vel + acc')

plt.title("Final Minkowski Sum")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.show()



## COMPARING




# -----------------------------
# Compute and plot the Minkowski Sum.
# -----------------------------





time_horizon = 2 * time_horizon

# Define polygon A-B-F-E using np.array.
bounded_vel: np.ndarray = np.array([
    [A[0] * time_horizon, A[1] * time_horizon],
    [B[0] * time_horizon, B[1] * time_horizon],
    [F[0] * time_horizon, F[1] * time_horizon],
    [E[0] * time_horizon, E[1] * time_horizon]
])

# Define the convex set square (vertices in counter-clockwise order).
X0_square: np.ndarray = np.array([
    [-0.5, 0.5],
    [-0.5, -0.5],
    [0.5, -0.5],
    [0.5, 0.5]
])

acc_displacement = 0.5 * time_horizon ** 2
# Define the acceleration displacement square.
acc_square: np.ndarray = np.array([
    [-acc_displacement / 2, acc_displacement / 2],
    [-acc_displacement / 2, -acc_displacement / 2],
    [acc_displacement / 2, -acc_displacement / 2],
    [acc_displacement / 2, acc_displacement / 2]
])


acc_square = regular_polygon(4, acc_displacement/2, np.array([0, 0]))
max_vel_radius = regular_polygon(10, v_max * time_horizon , np.array([0, 0]))

result_vel = minkowski(bounded_vel, X0_square, plot_intermediate = plot_intermediate)

result = minkowski(result_vel, acc_square, plot_intermediate = plot_intermediate)

max_vel = minkowski(max_vel_radius, X0_square, plot_intermediate = plot_intermediate)

final_og = intersection_convex(result, max_vel)
plt.figure(figsize=(10, 6))  # larghezza=8 pollici, altezza=6 pollic

plot_poly(X0_square, 'k-', 2, 'Initial Position Set')
plot_poly(final, 'k--', 2, 'Reachability of 0.5 second',zorder = 2)
plot_poly(final1, 'k--', 2, 'Reachability of 1 second (performed twice)',zorder = 2)
plot_poly(final_og, 'k-', 2, 'Reachability of 1 second')


plt.title("Final Minkowski Sum")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.show()

elapsed = end - start
print(f"High Resolution Elapsed Time: {elapsed*1_000} milliseconds")




