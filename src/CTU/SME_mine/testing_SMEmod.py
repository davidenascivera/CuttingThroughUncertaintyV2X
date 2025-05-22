import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Polygon
from typing import List, Tuple
from SME_module import SMEModule
# -----------------------------

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


def main():
    # Parametri globali
    deltaR = 2
    deltaA = 10
    sens = (-1, 0)
    pos_obj = (20, 0)


    # Otteniamo la lista dei vertici
    point_list = vertices_meas_range_bearing(dR=deltaR, dA=deltaA, pos_sens=sens, pos_obj=pos_obj)
    
    SME = SMEModule(meas_set=point_list)
    
    
    # Creazione della figura
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(point_list[:, 0], point_list[:, 1], color='blue', label='Target Path', zorder=2000, linewidth=1)
    
    
    ax.set_aspect('equal', 'box')
    buffer = 5
    ax.set_xlim(pos_obj[0] - buffer, pos_obj[0] + buffer)
    ax.set_ylim(pos_obj[1] - buffer, pos_obj[1] + buffer)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.legend(loc='upper right', fontsize=10)
    ax.set_title("Area Analysis of Annular Sector and Polygon")


    plt.show()

if __name__ == """__main__""":
    main()