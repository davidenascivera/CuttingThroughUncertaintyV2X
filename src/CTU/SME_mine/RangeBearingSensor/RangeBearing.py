import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Polygon
from typing import List, Tuple

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

    # Calcoliamo OG e offset_ang in modo globale per il plotting degli archi
    OG = np.linalg.norm(np.array(pos_obj) - np.array(sens))
    offset_ang = np.degrees(np.arctan2(pos_obj[1]-sens[1], pos_obj[0]-sens[0]))

    # Otteniamo la lista dei vertici
    point_list = vertices_meas_range_bearing(dR=deltaR, dA=deltaA, pos_sens=sens, pos_obj=pos_obj)
    labels = ['A', 'D', 'E', 'F', 'B', 'C']

    # Creazione della figura
    fig, ax = plt.subplots(figsize=(7, 7))

    # Disegna i punti e li etichetta
    for label, point in zip(labels, point_list):
        ax.scatter(point[0], point[1], s=40)
        ax.text(point[0], point[1], f' {label}', fontsize=12,
                verticalalignment='bottom', horizontalalignment='right')

    # Crea il poligono dall'area racchiusa dai punti
    polygon_area = Polygon(point_list, closed=True, 
                          fill=True,
                          facecolor='lightblue',
                          alpha=0.5, zorder=1000,
                          edgecolor='black', 
                          linewidth=2.0,  # Increased line width for visibility
                          label='Area racchiusa')
    ax.add_patch(polygon_area)


    ax.scatter(pos_obj[0], pos_obj[1], s=40, color='red', label='Target Position', zorder=2000)
    ax.scatter(sens[0], sens[1], s=40, color='red', label='Sensor Position', zorder=2000)



    '''
    draw_annulus_ring
    '''

    # Disegna gli archi dell'anello
    outer_radius = OG + deltaR/2
    inner_radius = OG - deltaR/2
    # Per gli archi usiamo angoli assoluti (misurati dal centro, a partire dall'asse x)
    start_angle = offset_ang - deltaA/2
    end_angle = offset_ang + deltaA/2

    # Arco esterno: qui impostiamo angle=0 poiché start_angle ed end_angle sono assoluti
    outer_arc = Arc(sens, 2*outer_radius, 2*outer_radius,
                    theta1=start_angle, theta2=end_angle, angle=0,
                    color='blue', linewidth=2, label="Arco esterno")
    ax.add_patch(outer_arc)

    # Arco interno
    inner_arc = Arc(sens, 2*inner_radius, 2*inner_radius,
                    theta1=start_angle, theta2=end_angle, angle=0,
                    color='blue', linewidth=2, label="Arco interno")
    ax.add_patch(inner_arc)

    # Riempiamo la porzione di anello (slice)
    theta = np.linspace(np.radians(start_angle), np.radians(end_angle), 50)
    outer_x = sens[0] + outer_radius * np.cos(theta)
    outer_y = sens[1] + outer_radius * np.sin(theta)
    inner_x = sens[0] + inner_radius * np.cos(theta)
    inner_y = sens[1] + inner_radius * np.sin(theta)

    polygon_x = np.concatenate([outer_x, inner_x[::-1]])
    polygon_y = np.concatenate([outer_y, inner_y[::-1]])
    annulus_polygon = Polygon(np.column_stack([polygon_x, polygon_y]),
                              alpha=0.3, color='skyblue', 
                              edgecolor='blue', linewidth=1.5,
                              zorder=2, label='Annulus Ring Slice')
    ax.add_patch(annulus_polygon)


    '''
    Area Calculations
    '''
    # Calculate annular sector area
    annular_area = (np.pi * (outer_radius**2 - inner_radius**2) * deltaA/360)

    # Calculate polygon area using vertices
    def polygon_area_calc(vertices):
        n = len(vertices)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += vertices[i][0] * vertices[j][1]
            area -= vertices[j][0] * vertices[i][1]
        area = abs(area) / 2.0
        return area

    poly_area = polygon_area_calc(point_list)
    area_ratio = annular_area / poly_area

    '''
    Plotting
    '''
    # Impostazioni del grafico
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

    # Add text box with area information
    text_info = f'Annular Area: {annular_area:.2f}\nPolygon Area: {poly_area:.2f}\nRatio (Annular/Polygon): {area_ratio:.2f}'
    plt.text(0.02, 0.98, text_info, transform=ax.transAxes, 
             bbox=dict(facecolor='white', alpha=0.8),
             verticalalignment='top')

    plt.show()

if __name__ == """__main__""":
    main()