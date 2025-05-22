import numpy as np
import os
os.environ["USE_NUMBA"] = "0"  # o "1"
import matplotlib.pyplot as plt
import time
import convex_int_numba
from convex_int_numba import convex_intersect



def polygon_closed(poly):
    # poly is a 2D NumPy array of shape (n,2)
    return np.vstack([poly, poly[0]])



if __name__ == "__main__":
    # Define example polygon shapes (scaled to integers)
    scale = 1000
    # Test intersection for second example (repeat the same logic as above)
    
    P2_points = np.array([
        [100, 100],
        [300, 100],
        [300, 200],
        [100, 200]
    ])
    
    Q2_points = np.array([
        [200, 150],
        [350, 150],
        [350, 250],
        [200, 250]
    ])
    P2 = P2_points.flatten().astype(np.int32)
    Q2 = Q2_points.flatten().astype(np.int32)
    nP2 = len(P2_points)
    nQ2 = len(Q2_points)
    # Prepare an output buffer for up to maxOut intersection points
    maxOut = 40  # maximum expected points
    outPoints_numba = np.zeros(2 * maxOut, dtype=np.float64)
    
    # Warmup cycles for Numba
    print("Performing warmup cycles...")
    for _ in range(5):
        _, _ = convex_intersect(P2, nP2, Q2, nQ2, maxOut)

    # Example 1: The original example from import.py
    P_points = np.array([
        [ 0.67811529, -1.65960514],
        [ 1.57945204, -1.65960514],
        [ 2.02945204, -1.332661  ],
        [ 2.20133675, -0.80365427],
        [ 2.3,        -0.5       ],
        [ 2.3,         0.5       ],
        [ 2.20133675,  0.80365427],
        [ 2.02945204,  1.332661  ],
        [ 1.57945204,  1.65960514],
        [ 1.02322145,  1.65960514],
        [ 0.12188471,  1.65960514],
        [-0.1985598,   1.57374229],
        [-0.6485598,   1.24679815],
        [-0.7110598,   1.18429815],
        [-0.8829445,   0.65529143],
        [-0.8829445,  -0.65529143],
        [-0.7110598,  -1.18429815],
        [-0.6485598,  -1.24679815],
        [-0.1985598,  -1.57374229],
        [ 0.12188471, -1.65960514]
    ]) * scale
    
    Q_points = np.array([
        [1.3,        -1.10730855],
        [2.20133675, -1.10730855],
        [2.26383675, -1.04480855],
        [2.3625,     -0.74115427],
        [2.3625,      0.74115427],
        [2.26383675,  1.04480855],
        [2.20133675,  1.10730855],
        [1.3,         1.10730855],
        [0.9795555,   1.0214457 ],
        [0.65911099,  0.93558285],
        [0.59661099,  0.87308285],
        [0.53411099,  0.81058285],
        [0.53411099, -0.81058285],
        [0.59661099, -0.87308285],
        [0.65911099, -0.93558285],
        [0.9795555,  -1.0214457 ]
    ]) * scale

    # Test intersection logic
    P = P_points.flatten().astype(np.int32)
    Q = Q_points.flatten().astype(np.int32)
    nP = len(P_points)
    nQ = len(Q_points)

    
    
    # Call the Numba implementation
    start_numba = time.perf_counter()
    outPoints_numba, numPoints_numba = convex_intersect(P, nP, Q, nQ, maxOut)
    end_numba = time.perf_counter()
    time_numba = (end_numba - start_numba) * 1e6  # Convert to microseconds
    pts_numba = outPoints_numba[:2*numPoints_numba].reshape((numPoints_numba, 2))

    # Print results and timing
    print("\nOriginal Example")
    print("\nNumba implementation (after 5 warmup cycles):")
    print(f"  Number of intersection points: {numPoints_numba}")
    print(f"  Time elapsed: {time_numba:.2f} µs")

    # Plot results
    polyP = P.reshape((nP, 2))
    polyQ = Q.reshape((nQ, 2))
    polyP_closed = polygon_closed(polyP)
    polyQ_closed = polygon_closed(polyQ)

    plt.figure(figsize=(8, 6))
    plt.plot(polyP_closed[:, 0], polyP_closed[:, 1], 'b-', label='Polygon P')
    plt.plot(polyQ_closed[:, 0], polyQ_closed[:, 1], 'g-', label='Polygon Q')
    
    if numPoints_numba > 0:
        polyInt = pts_numba
        polyInt_closed = polygon_closed(polyInt)
        plt.plot(polyInt_closed[:, 0], polyInt_closed[:, 1], 'r-', linewidth=2, label='Intersection')
        plt.plot(polyInt[:, 0], polyInt[:, 1], 'ro')
    else:
        plt.text(0.5, 0.5, "No Intersection", horizontalalignment='center', verticalalignment='center')
    
    plt.title(f'Polygon Intersection ({time_numba:.2f} µs)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

    # Example 2: Simple rectangles that intersect
    P2_points = np.array([
        [100, 100],
        [300, 100],
        [300, 200],
        [100, 200]
    ])
    
    Q2_points = np.array([
        [200, 150],
        [350, 150],
        [350, 250],
        [200, 250]
    ])
    
    # Test intersection for second example (repeat the same logic as above)
    P = P2_points.flatten().astype(np.int32)
    Q = Q2_points.flatten().astype(np.int32)
    nP = len(P2_points)
    nQ = len(Q2_points)

    outPoints_numba = np.zeros(2 * maxOut, dtype=np.float64)
    start_numba = time.perf_counter()
    outPoints_numba, numPoints_numba = convex_intersect(P, nP, Q, nQ, maxOut)
    print(f"outPoints_numba: {outPoints_numba}")
    end_numba = time.perf_counter()
    time_numba = (end_numba - start_numba) * 1e6  # Convert to microseconds
    pts_numba = outPoints_numba[:2*numPoints_numba].reshape((numPoints_numba, 2))

    # Plot results for second example (same plotting code as above)
    polyP = P.reshape((nP, 2))
    polyQ = Q.reshape((nQ, 2))
    polyP_closed = polygon_closed(polyP)
    polyQ_closed = polygon_closed(polyQ)

    plt.figure(figsize=(8, 6))
    plt.plot(polyP_closed[:, 0], polyP_closed[:, 1], 'b-', label='Polygon P')
    plt.plot(polyQ_closed[:, 0], polyQ_closed[:, 1], 'g-', label='Polygon Q')
    
    if numPoints_numba > 0:
        polyInt = pts_numba
        polyInt_closed = polygon_closed(polyInt)
        plt.plot(polyInt_closed[:, 0], polyInt_closed[:, 1], 'r-', linewidth=2, label='Intersection')
        plt.plot(polyInt[:, 0], polyInt[:, 1], 'ro')
    else:
        plt.text(0.5, 0.5, "No Intersection", horizontalalignment='center', verticalalignment='center')
    
    plt.title(f'Polygon Intersection ({time_numba:.2f} µs)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
