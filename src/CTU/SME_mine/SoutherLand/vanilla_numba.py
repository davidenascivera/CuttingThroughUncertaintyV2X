import numpy as np
import numba
from numba import njit
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Optional
import time
from scipy import optimize

@njit
def inside(p, a, b):
    """
    Return True if point p is to the left of the directed edge a->b.
    (For a clip polygon in counterclockwise order.)
    """
    return (b[0] - a[0])*(p[1] - a[1]) - (b[1] - a[1])*(p[0] - a[0]) >= 0

@njit
def compute_intersection(s, e, a, b):
    """
    Compute the intersection point between segment (s, e) and the clip edge (a, b).
    """
    dx_se = e[0] - s[0]
    dy_se = e[1] - s[1]
    dx_ba = b[0] - a[0]
    dy_ba = b[1] - a[1]
    cross_s_ba = (a[0] - s[0]) * dy_ba - (a[1] - s[1]) * dx_ba
    cross_se_ba = dx_se * dy_ba - dy_se * dx_ba
    if cross_se_ba == 0:
        t = 0.0
    else:
        t = cross_s_ba / cross_se_ba
    res = np.empty(2, dtype=s.dtype)
    res[0] = s[0] + t * dx_se
    res[1] = s[1] + t * dy_se
    return res

@njit
def sutherland_hodgman_numba(subject_polygon, clip_polygon):
    """
    Compute the intersection (clipping) of two convex polygons using the
    Sutherland–Hodgman algorithm. This version is optimized for Numba.
    
    Parameters:
        subject_polygon: (n,2) numpy array of polygon vertices (in CCW order)
        clip_polygon: (m,2) numpy array of clip polygon vertices (in CCW order)
        
    Returns:
        (k,2) numpy array of the clipped polygon vertices.
    """
    n_subject = subject_polygon.shape[0]
    n_clip = clip_polygon.shape[0]
    if n_subject == 0 or n_clip == 0:
        return np.empty((0, 2), dtype=subject_polygon.dtype)
    
    # Copy subject polygon into a working array.
    input_poly = np.copy(subject_polygon)
    input_count = n_subject
    
    # Allocate an output array with a generous size.
    max_vertices = input_count + n_clip  # maximum possible vertices after clipping
    output_poly = np.empty((max_vertices, 2), dtype=subject_polygon.dtype)
    
    # Process each clip edge.
    for i in range(n_clip):
        cp1 = clip_polygon[i]
        cp2 = clip_polygon[(i+1) % n_clip]
        output_count = 0
        
        if input_count == 0:
            return np.empty((0, 2), dtype=subject_polygon.dtype)
        
        # Start with the last vertex in the current input polygon.
        s = input_poly[input_count - 1]
        s_inside = inside(s, cp1, cp2)
        
        for j in range(input_count):
            e = input_poly[j]
            e_inside = inside(e, cp1, cp2)
            
            if e_inside:
                if not s_inside:
                    inter_pt = compute_intersection(s, e, cp1, cp2)
                    output_poly[output_count, 0] = inter_pt[0]
                    output_poly[output_count, 1] = inter_pt[1]
                    output_count += 1
                output_poly[output_count, 0] = e[0]
                output_poly[output_count, 1] = e[1]
                output_count += 1
            elif s_inside:
                inter_pt = compute_intersection(s, e, cp1, cp2)
                output_poly[output_count, 0] = inter_pt[0]
                output_poly[output_count, 1] = inter_pt[1]
                output_count += 1
            
            s = e
            s_inside = e_inside
        
        # Prepare for the next clip edge: copy output_poly to input_poly.
        input_count = output_count
        for k in range(output_count):
            input_poly[k, 0] = output_poly[k, 0]
            input_poly[k, 1] = output_poly[k, 1]
    
    # Return the clipped polygon.
    result = np.empty((input_count, 2), dtype=subject_polygon.dtype)
    for i in range(input_count):
        result[i, 0] = input_poly[i, 0]
        result[i, 1] = input_poly[i, 1]
    return result

def sutherland_hodgman(subject_polygon: np.ndarray, clip_polygon: np.ndarray) -> np.ndarray:
    return sutherland_hodgman_numba(subject_polygon, clip_polygon)

# Plotting helper remains unchanged.
def plot_polygons(subject: np.ndarray, clip: np.ndarray, intersection: Optional[np.ndarray] = None, title: str = ""):
    plt.clf()
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    
    patch_subject = patches.Polygon(subject, closed=True, fill=False, edgecolor='blue', linewidth=2, label='Subject')
    ax.add_patch(patch_subject)
    
    patch_clip = patches.Polygon(clip, closed=True, fill=False, edgecolor='green', linewidth=2, label='Clip')
    ax.add_patch(patch_clip)
    
    if intersection is not None and intersection.size > 0:
        patch_inter = patches.Polygon(intersection, closed=True, fill=False, edgecolor='red', linestyle='--', linewidth=2, label='Intersection')
        ax.add_patch(patch_inter)
        plt.plot(intersection[:, 0], intersection[:, 1], 'ro')
    
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()

def warmup_suth_numba():
    subject = np.array([[1, 1], [4, 1], [4, 4], [1, 4]], dtype=np.float64)
    clip = np.array([[2, 0], [5, 2], [3, 5], [0, 3]], dtype=np.float64)
    for _ in range(5):
        inter_poly = sutherland_hodgman(subject, clip)
    return 

def create_regular_convex_polygon(n, inradius=1.0, offset=(0, 0)):
    """
    Create a regular convex polygon with n vertices.
    The polygon is constructed as a polygon circumscribed around a circle of radius inradius.
    
    Parameters:
      - n: Number of vertices
      - inradius: Radius of the inscribed circle
      - offset: (x, y) offset for the polygon's center
      
    Returns:
      - numpy array of shape (n, 2) containing the polygon vertices
    """
    # Calculate the circumscribed radius
    R = inradius / np.cos(np.pi / n)
    vertices = np.empty((n, 2), dtype=np.float64)
    for i in range(n):
        angle = 2 * np.pi * i / n
        vertices[i, 0] = R * np.cos(angle) + offset[0]
        vertices[i, 1] = R * np.sin(angle) + offset[1]
    
    return vertices

def measure_performance(vertices_count, num_trials=100):
    """
    Measure the average execution time for intersection of two polygons with given number of vertices.
    
    Parameters:
        vertices_count: Number of vertices for each polygon
        num_trials: Number of trials to run for averaging
        
    Returns:
        Average execution time in microseconds
    """
    total_time = 0
    prova = int(vertices_count/2)  # Using half the number of vertices for the second polygon
    
    for _ in range(num_trials):
        # Create two polygons
        poly_subject = create_regular_convex_polygon(vertices_count, inradius=1.0, offset=(0, 0))
        poly_clip = create_regular_convex_polygon(prova, inradius=1.0, offset=(0.5, 0))
        
        # Measure execution time
        start = time.perf_counter()
        sutherland_hodgman(poly_subject, poly_clip)
        end = time.perf_counter()
        
        total_time += (end - start) * 1e6  # Convert to microseconds
        
    return total_time / num_trials

# Main execution with warmup and timing.
if __name__ == "__main__":
    # Original test case
    subject = np.array([[1, 1], [4, 1], [4, 4], [1, 4]], dtype=np.float64)
    clip = np.array([[2, 0], [5, 2], [3, 5], [0, 3]], dtype=np.float64)
    
    warmup_suth_numba()
    
    # Timing phase: run 100 iterations and compute the average time.
    times = np.empty(100, dtype=np.float64)
    for i in range(100):
        start = time.perf_counter()
        inter_poly = sutherland_hodgman(subject, clip)
        end = time.perf_counter()
        times[i] = end - start
    avg_time = times.mean()
    print(f"Average time over 100 runs: {avg_time*1_000_000:.2f} microseconds")
    print("Intersection polygon vertices:")
    print(inter_poly)
    
    plot_polygons(subject, clip, inter_poly, "Sutherland-Hodgman Convex Intersection")
    
    # New test case similar to python_import.py
    num_vertices = 800
    
    poly_subject = create_regular_convex_polygon(num_vertices, inradius=1.0, offset=(0, 0))
    poly_clip = create_regular_convex_polygon(int(num_vertices/2), inradius=1.0, offset=(0.5, 0))
    
    start = time.perf_counter()
    # Compute intersection using Sutherland-Hodgman
    inter_poly = sutherland_hodgman(poly_subject, poly_clip)
    end = time.perf_counter()
    elapsed = end - start
    print(f"Time elapsed for intersection calculation: {elapsed*1e6:.2f} microseconds")
    print("Number of intersection points:", len(inter_poly))
    
    plot_polygons(poly_subject, poly_clip, inter_poly, "Sutherland-Hodgman Regular Polygon Intersection (Numba)")
    
    # Scalability testing
    print("\nPerformance testing with different polygon sizes:")
    print("================================================")
    print("Vertices | Average Time (μs)")
    print("---------|------------------")
    
    # Test with different numbers of vertices
    vertex_counts = [10, 20, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    avg_times = []
    
    for count in vertex_counts:
        avg_time = measure_performance(count)
        avg_times.append(avg_time)
        print(f"{count:8d} | {avg_time:.2f}")
    
    # Create a figure with two subplots for visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # 1. Log-Log Plot
    ax1.plot(vertex_counts, avg_times, 'o-', color='#1f77b4', linewidth=2, 
            markersize=8, markerfacecolor='white', markeredgewidth=1.5)
    
    # Add trend line for log-log plot
    # Fit a power law y = ax^b (appears as a line in log-log)
    if len(vertex_counts) > 1:
        def power_law(x, a, b):
            return a * np.power(x, b)
        
        try:
            # Use non-linear least squares to fit
            params, _ = optimize.curve_fit(power_law, vertex_counts, avg_times)
            a, b = params
            
            # Generate points for the trend line
            x_trend = np.logspace(np.log10(min(vertex_counts)), np.log10(max(vertex_counts)), 100)
            y_trend = power_law(x_trend, a, b)
            
            ax1.plot(x_trend, y_trend, '--', color='#ff7f0e', linewidth=2, 
                    label=f'Trend: y = {a:.3f}x^{b:.3f}')
            
            # Add annotation for complexity
            complexity_text = f"Empirical complexity: O(n^{b:.3f})"
            ax1.text(0.05, 0.95, complexity_text, transform=ax1.transAxes, 
                    fontsize=12, verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        except:
            print("Could not fit trend line to data")
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Number of Vertices per Polygon', fontsize=12)
    ax1.set_ylabel('Average Execution Time (μs)', fontsize=12)
    ax1.set_title('Log-Log Plot of Algorithm Scalability (Numba)', fontsize=14)
    ax1.grid(True, which="both", linestyle="--", alpha=0.7)
    ax1.legend(loc='upper left')
    
    # 2. Linear Plot
    ax2.plot(vertex_counts, avg_times, 'o-', color='#2ca02c', linewidth=2, 
            markersize=8, markerfacecolor='white', markeredgewidth=1.5)
    
    # Add trend line for linear plot
    if len(vertex_counts) > 1 and 'a' in locals() and 'b' in locals():
        # Use same trend line as before but on linear scale
        x_linear = np.linspace(min(vertex_counts), max(vertex_counts), 100)
        y_linear = power_law(x_linear, a, b)
        ax2.plot(x_linear, y_linear, '--', color='#d62728', linewidth=2,
                label=f'Trend: y = {a:.3f}x^{b:.3f}')
    
    ax2.set_xlabel('Number of Vertices per Polygon', fontsize=12)
    ax2.set_ylabel('Average Execution Time (μs)', fontsize=12)
    ax2.set_title('Linear Plot of Algorithm Scalability (Numba)', fontsize=14)
    ax2.grid(True, linestyle="--", alpha=0.7)
    ax2.legend(loc='upper left')
    
    # Annotate specific data points
    for i in [0, 4, len(vertex_counts)-1]:  # First, middle-ish, and last points
        if i < len(vertex_counts):
            x, y = vertex_counts[i], avg_times[i]
            ax2.annotate(f'({x}, {y:.1f}μs)', 
                        xy=(x, y), 
                        xytext=(10, 10),
                        textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'),
                        fontsize=10)
    
    plt.tight_layout()
    plt.suptitle('Scalability Analysis of Sutherland-Hodgman Algorithm (Numba)', 
                fontsize=16, y=1.02)
    
    # Save the plot to a file
    plt.savefig('sutherland_hodgman_numba_scalability.png', dpi=300, bbox_inches='tight')
    
    # Show the plot
    plt.show()
