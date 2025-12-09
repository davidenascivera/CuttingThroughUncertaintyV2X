#!/usr/bin/env python3
"""
Python implementation of convex polygon intersection algorithm.
Based on the C code from "Computational Geometry in C" (Second Edition), Chapter 7
by Joseph O'Rourke.

This version uses ctypes to load the shared library (e.g. "./libconvexintersect.so")
and call the `convex_intersect` function.
It generates two regular convex polygons (constructed as circoscritti ad un cerchio)
that intersect, and then plots the input polygons and their intersection using matplotlib.
"""

import time
import ctypes
import matplotlib.pyplot as plt
import numpy as np
import math

# Load the shared library (adjust the filename/path if needed)
lib = ctypes.CDLL('./libconvexintersect.dll')

# Define the argument and return types for the convex_intersect function
lib.convex_intersect.argtypes = [
    ctypes.POINTER(ctypes.c_int), ctypes.c_int,  # P_flat and n
    ctypes.POINTER(ctypes.c_int), ctypes.c_int,  # Q_flat and m
    ctypes.POINTER(ctypes.c_double)              # result
]
lib.convex_intersect.restype = ctypes.c_int

def flatten_polygon(vertices, scale=100):
    """
    Convert a list of (x, y) vertices into a flat list of integers.
    The scale factor converts the float coordinates into integers,
    which is expected by the C code.
    """
    flat = []
    for (x, y) in vertices:
        flat.append(int(x * scale))
        flat.append(int(y * scale))
    return flat

def create_regular_convex_polygon(n, inradius=1.0, scale=100, offset=(0, 0)):
    """
    Crea un poligono regolare convesso costruito come poligono circoscritto al cerchio.
    I lati del poligono saranno tangenti a un cerchio di raggio "inradius".
    
    Parameters:
      - n: Numero di vertici del poligono.
      - inradius: Raggio del cerchio inscritto (il cerchio tangente ai lati).
      - scale: Fattore di scala per convertire le coordinate float in interi.
      - offset: Coppia (x, y) per traslare il poligono.
      
    Returns:
      - vertices: Lista di tuple (x, y) in ordine antiorario.
      - flat: Lista piatta di interi (scalati) per l'uso con la libreria C.
    """
    # Calcola il raggio circoscritto R, in modo che il cerchio inscrito abbia raggio inradius.
    R = inradius / math.cos(math.pi / n)
    vertices = []
    for i in range(n):
        angle = 2 * math.pi * i / n
        x = R * math.cos(angle) + offset[0]
        y = R * math.sin(angle) + offset[1]
        vertices.append((x, y))
    
    flat = flatten_polygon(vertices, scale=scale)
    return vertices, flat

def plot_polygons_and_intersection(P, Q, intersection):
    """Plot the input polygons and their intersection."""
    # Convert flat lists to Nx2 numpy arrays
    P = np.array(P).reshape(-1, 2)
    Q = np.array(Q).reshape(-1, 2)
    intersection = np.array(intersection).reshape(-1, 2)
    
    # Chiudere i poligoni (appendendo il primo punto alla fine)
    P = np.vstack([P, P[0]])
    Q = np.vstack([Q, Q[0]])
    if len(intersection) > 0:
        intersection = np.vstack([intersection, intersection[0]])
    
    plt.figure(figsize=(8, 8))
    
    plt.plot(P[:,0], P[:,1], 'b-', label='Polygon P')
    plt.fill(P[:,0], P[:,1], 'b', alpha=0.3)
    
    plt.plot(Q[:,0], Q[:,1], 'r-', label='Polygon Q')
    plt.fill(Q[:,0], Q[:,1], 'r', alpha=0.3)
    
    if len(intersection) > 0:
        plt.plot(intersection[:,0], intersection[:,1], 'g-', label='Intersection')
        plt.fill(intersection[:,0], intersection[:,1], 'g', alpha=0.5)
    
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Regular Convex Polygon Intersection')
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def measure_performance(vertices_count, num_trials=1000):
    """
    Measure the average execution time for intersection of two polygons with given number of vertices..
    
    Parameters:
        vertices_count: Number of vertices for each polygon
        num_trials: Number of trials to run for averaging
        
    Returns:
        Average execution time in microseconds
    """
    total_time = 0
    prova = int(vertices_count/2)
    
    for _ in range(num_trials):
        # Create two polygons
        poly_P, flat_P = create_regular_convex_polygon(vertices_count, inradius=1.0, scale=1e6, offset=(0, 0))
        poly_Q, flat_Q = create_regular_convex_polygon(prova, inradius=1.0, scale=1e6, offset=(0.5, 0))
        
        n = len(flat_P) // 2
        m = len(flat_Q) // 2
        
        # Prepare result array
        result = (ctypes.c_double * (2 * (n + m)))()
        
        # Measure execution time
        start = time.perf_counter()
        lib.convex_intersect(
            (ctypes.c_int * len(flat_P))(*flat_P), n,
            (ctypes.c_int * len(flat_Q))(*flat_Q), m,
            result
        )
        end = time.perf_counter()
        
        total_time += (end - start) * 1e6  # Convert to microseconds
        
    return total_time / num_trials

def main():
    # Test specific polygon intersection (original functionality)
    num_vertices = 800
    
    poly_P, flat_P = create_regular_convex_polygon(num_vertices, inradius=1.0, scale=1e6, offset=(0, 0))
    poly_Q, flat_Q = create_regular_convex_polygon(num_vertices, inradius=1.0, scale=1e6, offset=(0.5, 0))
    
    n = len(flat_P) // 2
    m = len(flat_Q) // 2
    
    result = (ctypes.c_double * (2 * (n + m)))()
    start = time.perf_counter()
    num_points = lib.convex_intersect(
        (ctypes.c_int * len(flat_P))(*flat_P), n,
        (ctypes.c_int * len(flat_Q))(*flat_Q), m,
        result
    )
    end = time.perf_counter()
    elapsed_time = end - start
    print(f"Time elapsed for intersection calculation: {elapsed_time*1e6:.2f} microseconds")
    
    intersection = [result[i] for i in range(2 * num_points)]
    print("Number of intersection points:", num_points)
    
    # Plot the polygons and their intersection
    plot_polygons_and_intersection(flat_P, flat_Q, intersection)
    
    # Test scalability with different numbers of vertices
    vertex_counts = [10, 20, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    avg_times = []
    
    print("\nPerformance testing with different polygon sizes:")
    print("================================================")
    print("Vertices | Average Time (μs)")
    print("---------|------------------")
    
    for count in vertex_counts:
        avg_time = measure_performance(count)
        avg_times.append(avg_time)
        print(f"{count:8d} | {avg_time:.2f}")
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # 1. Log-Log Plot
    ax1.plot(vertex_counts, avg_times, 'o-', color='#1f77b4', linewidth=2, 
            markersize=8, markerfacecolor='white', markeredgewidth=1.5)
    
    # Add trend line for log-log plot
    # Fit a power law y = ax^b (appears as a line in log-log)
    if len(vertex_counts) > 1:
        from scipy import optimize
        
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
    ax1.set_title('Log-Log Plot of Algorithm Scalability', fontsize=14)
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
    ax2.set_title('Linear Plot of Algorithm Scalability', fontsize=14)
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
    plt.suptitle('Scalability Analysis of Convex Polygon Intersection Algorithm', 
                fontsize=16, y=1.02)
    
    # Save the plot to a file
    plt.savefig('scalability_analysis.png', dpi=300, bbox_inches='tight')
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()
