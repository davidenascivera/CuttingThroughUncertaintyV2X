import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Optional
from scipy.spatial import ConvexHull


# -----------------------------
# Helper functions for the Minkowski sum algorithm
# -----------------------------
def reorder_polygon(P: np.ndarray) -> np.ndarray:
    # Find the vertex with the minimum y (and minimum x in case of a tie)
    pos = np.lexsort((P[:, 0], P[:, 1]))[0]  # Sort by (y, x), get the first index

    # Rotate the array to start from the found position
    return np.roll(P, -pos, axis=0)

def plot_polygons(P: np.ndarray, Q: np.ndarray, result: Optional[np.ndarray], title: str):
    plt.clf()
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')

    # Plot polygon P (blue)
    patch_P = patches.Polygon(P, closed=True, fill=False, edgecolor='blue', linewidth=2, label='P')
    ax.add_patch(patch_P)

    # Plot polygon Q (green)
    patch_Q = patches.Polygon(Q, closed=True, fill=False, edgecolor='green', linewidth=2, label='Q')
    ax.add_patch(patch_Q)

    # Plot the current Minkowski sum (result) in red (dashed)
    if result is not None and result.size > 0:
        patch_R = patches.Polygon(result, closed=True, fill=False, edgecolor='red', linestyle='--', linewidth=2, label='P+Q (current)')
        ax.add_patch(patch_R)

        # Extract x, y coordinates for point markers
        rx, ry = result[:, 0], result[:, 1]
        plt.plot(rx, ry, 'ro')

    plt.title(title)
    plt.grid(True)
    plt.pause(0.5)

def minkowski(P: np.ndarray, Q: np.ndarray, plot_intermediate: bool = False) -> np.ndarray:
    # Reorder polygons so that the vertex with the lowest y is first.
    P = reorder_polygon(P.copy())
    Q = reorder_polygon(Q.copy())
    
    # Append the first two vertices at the end for cyclic handling.
    P = np.vstack((P, P[:2]))
    Q = np.vstack((Q, Q[:2]))
    
    # Initialize result as an empty NumPy array
    result = np.empty((0, 2), dtype=np.float64)
    
    i, j = 0, 0

    if plot_intermediate:
        plt.figure(figsize=(6,6))
        plot_polygons(P[:-2], Q[:-2], result, "Start: Polygons P and Q")
    
    # Main loop: step through the edges of both polygons.
    while i < len(P) - 2 or j < len(Q) - 2:
        # Addition of np.array objects works element-wise.
        new_point = P[i] + Q[j]
        result = np.vstack((result, new_point))  # Efficiently stack points
        
        title = f"Iteration: i={i}, j={j}, added: ({new_point[0]:.2f}, {new_point[1]:.2f})"
        if plot_intermediate:
            plot_polygons(P[:-2], Q[:-2], result, title)
        
        # Compute cross product of the current edge vectors using np.cross.
        cross_val = np.cross(P[i+1] - P[i], Q[j+1] - Q[j])
        if cross_val >= 0 and i < len(P) - 2:
            i += 1
        if cross_val <= 0 and j < len(Q) - 2:
            j += 1

    if plot_intermediate:
        plot_polygons(P[:-2], Q[:-2], result, "Final Minkowski Sum")
        plt.show()

    return result

def minkowski_safe(P: np.ndarray, Q: np.ndarray, plot_intermediate: bool = False) -> np.ndarray:
    """
    Compute the Minkowski sum of two sets of points (polygons) by computing all pairwise sums
    and taking the convex hull of the resulting points.
    
    Parameters:
        P (np.ndarray): An (n,2) array of vertices for polygon P.
        Q (np.ndarray): An (m,2) array of vertices for polygon Q.
        plot_intermediate (bool): If True, plot the Minkowski sum result.
    
    Returns:
        np.ndarray: An array of vertices representing the convex hull of the Minkowski sum.
    """
    # Compute all pairwise sums of vertices from P and Q.
    points = np.array([p + q for p in P for q in Q])
    
    # Compute the convex hull of the summed points.
    hull = ConvexHull(points)
    result = points[hull.vertices]

    return result

def plot_poly(poly: List[np.ndarray] = None, cl='r-', lw=2, name='') -> None:
    if poly is None:
        return
    poly_coords = [(pt[0], pt[1]) for pt in poly]
    poly_coords.append(poly_coords[0])
    x_point, y_point = zip(*poly_coords)
    plt.plot(x_point, y_point, cl, marker='o', linewidth=lw, label=name)

def regular_polygon(n: int, radius: float = 1.0, center: np.ndarray = None) -> np.ndarray:
    if center is None:
        center = np.array([0.0, 0.0])
    cx, cy = center
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = cx + radius * np.cos(angles)
    y = cy + radius * np.sin(angles)
    return np.stack((x, y), axis=1)  # shape (n, 2)
def plot_poly(poly: Optional[np.ndarray] = None, cl='r-', lw=2, name='',zorder = 1) -> None:
    if poly is None or len(poly) == 0:
        return

    # Ensure poly is a NumPy array and close the polygon
    poly_closed = np.vstack((poly, poly[0]))  # Append first point at the end

    # Extract x and y coordinates using NumPy slicing
    x_points, y_points = poly_closed[:, 0], poly_closed[:, 1]

    # Plot using NumPy-optimized values
    plt.plot(x_points, y_points, cl, marker='o', linewidth=lw, label=name, zorder=zorder)



def get_annular_sector_points(a_0: float, r_0: float, delta_a: float, delta_r: float, future_steps: float) -> np.ndarray:
    """
    Generate 4 points representing the polygon of an annular sector (A-B-F-E), scaled and rotated.
    
    Parameters:
        a_0 (float): Center angle of the sector in degrees (rotation around origin).
        r_0 (float): Mean radius of the sector.
        delta_a (float): Half angular width of the sector in degrees.
        delta_r (float): Radial thickness of the sector.
        future_steps (float): Time horizon to scale the polygon.
        
    Returns:
        np.ndarray: 4x2 array of (x, y) points in the order A-B-F-E.
    """
    # Inner and outer radii
    inner_radius = np.maximum(r_0 - delta_r, 0)
    outer_radius = np.maximum(r_0 + delta_r, 0) 

    # Angles for points A and B (inner arc)
    angle_A = delta_a
    angle_B = -delta_a

    # Coordinates in local (non-rotated) frame
    A = (inner_radius * np.cos(np.radians(angle_A)), inner_radius * np.sin(np.radians(angle_A)))
    B = (inner_radius * np.cos(np.radians(angle_B)), inner_radius * np.sin(np.radians(angle_B)))
    E = (outer_radius, outer_radius * np.tan(np.radians(angle_A)))
    F = (outer_radius, -outer_radius * np.tan(np.radians(angle_A)))

    # Stack into array and scale
    polygon = np.array([A, B, F, E]) * future_steps

    # Rotation matrix for a_0 degrees
    theta = np.radians(a_0)
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])

    # Rotate all points
    rotated_polygon = (R @ polygon.T).T

    return rotated_polygon

def vertices_meas_range_bearing(dR: float = 0.5, dA: float = 30,
                                pos_sens: tuple = (0, 0), pos_obj: tuple = (0, 1.5)) -> np.ndarray:
    """
    Calculate vertices of a polygon representing a range-bearing sensor measurement region.
    
    This function computes the vertices of a polygon that approximates an annular sector
    representing the measurement region of a range-bearing sensor. The polygon is formed
    by 6 points (A, B, C, D, E, F) that define an area where a target might be located
    given sensor uncertainty in both range (dR) and bearing angle (dA).
    
    Parameters:
    -----------
    dR : float
        Range uncertainty (distance uncertainty) of the sensor measurement.
    dA : float
        # Bearing angle uncertainty in degrees of the sensor measurement.
    pos_sens : tuple(float, float)
        Position coordinates (x, y) of the sensor.
    pos_obj : tuple(float, float)
        Position coordinates (x, y) of the target.
    
    Returns:
    --------
    np.ndarray
        A 6x2 array of points defining the vertices of the polygon.
        The points are ordered counter-clockwise as [A, B, C, D, E, F].
    """
    # Ensure inputs are properly typed
    dR = float(dR)
    dA = float(dA)
    sens = (float(pos_sens[0]), float(pos_sens[1]))
    G = (float(pos_obj[0]), float(pos_obj[1]))

    # Calculate distance between sensor and target
    OG = np.linalg.norm(np.array(G) - np.array(sens))
    # Calculate angle offset from sensor to target
    offset_ang = np.degrees(np.arctan2(G[1]-sens[1], G[0]-sens[0]))

    # Calculate point U (center point on outer arc)
    U = (sens[0] + (OG+dR/2)*np.cos(np.radians(offset_ang)),
         sens[1] + (OG+dR/2)*np.sin(np.radians(offset_ang)))

    # Calculate vertices on outer arc
    A = (sens[0] + (OG + dR/2) * np.cos(np.radians(offset_ang - dA/2)), 
         sens[1] + (OG + dR/2) * np.sin(np.radians(offset_ang - dA/2)))

    D = (sens[0] + (OG + dR/2) * np.cos(np.radians(offset_ang + dA/2)), 
         sens[1] + (OG + dR/2) * np.sin(np.radians(offset_ang + dA/2)))

    # Calculate vertices on inner arc
    E = (sens[0] + (OG - dR/2) * np.cos(np.radians(offset_ang + dA/2)), 
         sens[1] + (OG - dR/2) * np.sin(np.radians(offset_ang + dA/2)))

    F = (sens[0] + (OG - dR/2) * np.cos(np.radians(offset_ang - dA/2)), 
         sens[1] + (OG - dR/2) * np.sin(np.radians(offset_ang - dA/2)))

    # Calculate intermediate points B and C with increased radius to better
    # approximate the curved outer arc
    lunghezza = (OG + dR/2) / np.cos(np.radians(dA/4))
    B = (sens[0] + lunghezza * np.cos(np.radians(offset_ang - dA/4)), 
         sens[1] + lunghezza * np.sin(np.radians(offset_ang - dA/4)))
    C = (sens[0] + lunghezza * np.cos(np.radians(offset_ang + dA/4)),
         sens[1] + lunghezza * np.sin(np.radians(offset_ang + dA/4)))
    
    # # Print points, dR, and dA
    # print(f"Points: A=({A[0]:.3f}, {A[1]:.3f}), B=({B[0]:.3f}, {B[1]:.3f}), C=({C[0]:.3f}, {C[1]:.3f}), D=({D[0]:.3f}, {D[1]:.3f}), E=({E[0]:.3f}, {E[1]:.3f}), F=({F[0]:.3f}, {F[1]:.3f})")
    # print(f"Radius (dR): {dR}")
    # print(f"Angle (dA): {dA} degrees")
    
    # Return vertices in counter-clockwise order as numpy array
    return np.array([A, B, C, D, E, F,A])