import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Optional

def sutherland_hodgman(subject_polygon: np.ndarray, clip_polygon: np.ndarray) -> np.ndarray:
    """
    Compute the intersection of two convex polygons using the Sutherland-Hodgman algorithm.
    
    Parameters:
        subject_polygon: np.ndarray of shape (n, 2)
        clip_polygon: np.ndarray of shape (m, 2)
    
    Returns:
        np.ndarray of the clipped polygon (the intersection) with shape (k, 2).
    """
    # Optimized inside check (point is inside if it's to the left of the directed edge)
    def inside(p, a, b):
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0
    
    # Optimized intersection calculation
    def compute_intersection(s, e, a, b):
        # Pre-compute values
        dx_se = e[0] - s[0]
        dy_se = e[1] - s[1]
        dx_ba = b[0] - a[0]
        dy_ba = b[1] - a[1]
        
        # Cross product components for intersection calculation
        cross_s_ba = (a[0] - s[0]) * dy_ba - (a[1] - s[1]) * dx_ba
        cross_se_ba = dx_se * dy_ba - dy_se * dx_ba
        
        # Calculate the parameter t
        t = cross_s_ba / cross_se_ba if cross_se_ba != 0 else 0
        
        # Calculate intersection point
        return np.array([s[0] + t * dx_se, s[1] + t * dy_se])

    # Start with subject polygon
    output_list = []
    
    # Early exit if either polygon is empty
    if len(subject_polygon) == 0 or len(clip_polygon) == 0:
        return np.array(output_list)
    
    # First iteration uses the original subject polygon
    input_list = subject_polygon.copy()
    
    # Create a closed loop by connecting last point to first point
    clip_edges = np.vstack((clip_polygon, clip_polygon[0:1]))
    
    # Process each edge of clip polygon
    for i in range(len(clip_polygon)):
        # Exit early if no vertices remain
        if len(input_list) == 0:
            return np.array(output_list)
        
        output_list = []
        cp1, cp2 = clip_edges[i], clip_edges[i+1]
        
        # Process each edge of the current input polygon
        s = input_list[-1]  # Start with the last point
        
        for e in input_list:
            e_inside = inside(e, cp1, cp2)
            s_inside = inside(s, cp1, cp2)
            
            # Case 1: end point is inside
            if e_inside:
                # If start point is outside, add intersection
                if not s_inside:
                    output_list.append(compute_intersection(s, e, cp1, cp2))
                # Always add the end point if it's inside
                output_list.append(e)
            # Case 2: end point is outside but start point is inside
            elif s_inside:
                # Add intersection point
                output_list.append(compute_intersection(s, e, cp1, cp2))
            
            # Update start point for next edge
            s = e
        
        # Update input for next clip edge
        input_list = output_list.copy() if output_list else []
    
    return np.array(output_list) if output_list else np.array([])

# Helper function to plot polygons
def plot_polygons(subject: np.ndarray, clip: np.ndarray, intersection: Optional[np.ndarray] = None, title: str = ""):
    plt.clf()
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    
    # Plot subject polygon (blue)
    patch_subject = patches.Polygon(subject, closed=True, fill=False, edgecolor='blue', linewidth=2, label='Subject')
    ax.add_patch(patch_subject)
    
    # Plot clip polygon (green)
    patch_clip = patches.Polygon(clip, closed=True, fill=False, edgecolor='green', linewidth=2, label='Clip')
    ax.add_patch(patch_clip)
    
    # Plot intersection polygon (red)
    if intersection is not None and intersection.size > 0:
        patch_inter = patches.Polygon(intersection, closed=True, fill=False, edgecolor='red', linestyle='--', linewidth=2, label='Intersection')
        ax.add_patch(patch_inter)
        # Plot the vertices of the intersection
        plt.plot(intersection[:, 0], intersection[:, 1], 'ro')
    
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()
import time
# Example usage:
if __name__ == "__main__":
    # Define two convex polygons (assumed to be in counterclockwise order)
    subject = np.array([[1, 1], [4, 1], [4, 4], [1, 4]])
    clip = np.array([[2, 0], [5, 2], [3, 5], [0, 3]])
    
    start = time.perf_counter()
    # Compute intersection using Sutherland-Hodgman
    inter_poly = sutherland_hodgman(subject, clip)
    end = time.perf_counter()
    elapsed = end - start
    print(f"Elapsed time: {elapsed*1_000_000} microseconds")
    print("Intersection polygon vertices:")
    print(inter_poly)
    
    plot_polygons(subject, clip, inter_poly, "Sutherland-Hodgman Convex Intersection")
