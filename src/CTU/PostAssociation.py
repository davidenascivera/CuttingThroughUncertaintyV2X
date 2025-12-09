import numpy as np
import logging

from shapely.geometry import Polygon

from SBE_module.utils import (
    get_annular_sector_points,
    minkowski_safe,
    regular_polygon
)

# Add logger setup
logger = logging.getLogger(__name__)

class PostAssociation:
    """
    Visualization of:
      1) pure intersections of two SME prediction polygons
      2) reachable‐set expansions (velocity+acceleration) for mature tracks
    """

    def __init__(self,
                 SME_dict: dict[int, object],
                 renderer,
                 dA: float,
                 dR: float,
                 max_vel: float = 1.8,
                 max_acc: float = 1.0):
        self.SME_dict = SME_dict    
        self.renderer = renderer
        self.dA1      = dA
        self.dR1      = dR
        self.max_vel  = max_vel
        self.max_acc  = max_acc

    def visualize_intersections(self,
                                ass_list: list[tuple[int,int]]):
        """
        Draw only the intersection boundary of each pair’s predicted sets.
        """
        ax = self.renderer.ax
        for t1, t2 in ass_list:
            poly1 = Polygon(np.array(self.SME_dict[t1].Xc))
            poly2 = Polygon(np.array(self.SME_dict[t2].Xc))
            inter = poly1.intersection(poly2)
            if inter.is_empty or inter.area < 1e-2:
                continue
            pts = np.array(inter.exterior.coords)
            ax.plot(pts[:,0], pts[:,1],
                    linestyle='-',
                    linewidth=2,
                    color='red',
                    zorder=100000)
            # logger.critical(f"AREA INT: {inter.area:.4f}")

    def visualize_predictions(self,
                              ass_list: list[tuple[int,int]], t: float):
        """
        For each pair where neither track is "missed",
        expand their intersection by velocity & acceleration.
        """
        ax = self.renderer.ax
        for t1, t2 in ass_list:
            tr1 = self.SME_dict[t1]
            tr2 = self.SME_dict[t2]
            if tr1.miss_in_a_row >= 1 or tr2.miss_in_a_row >= 1:
                continue

            base1 = Polygon(np.array(tr1.Xc))
            base2 = Polygon(np.array(tr2.Xc))
            inter = base1.intersection(base2)
            if inter.is_empty:
                continue
            base_pts = np.array(inter.exterior.coords)

            for step in [0.0, 0.2, 0.4, 0.6, 0.8]:

                sec1 = get_annular_sector_points(
                    a_0=tr1.last_vel_meas[0],
                    r_0=tr1.last_vel_meas[1],
                    delta_a=self.dA1,
                    delta_r=self.dR1,
                    future_steps=step
                )
                sec2 = get_annular_sector_points(
                    a_0=tr2.last_vel_meas[0],
                    r_0=tr2.last_vel_meas[1],
                    delta_a=self.dA1,
                    delta_r=self.dR1,
                    future_steps=step
                )
                iv = Polygon(sec1).intersection(Polygon(sec2))
                if iv.is_empty:
                    continue

                acc_disp   = 0.5 * step**2 * self.max_acc
                acc_poly   = regular_polygon(4, radius=acc_disp)
                vel_circle = regular_polygon(10, radius=self.max_vel * step)

                reach_vel = minkowski_safe(
                    np.array(iv.exterior.coords),
                    base_pts
                )
                reach_acc = minkowski_safe(reach_vel, acc_poly)
                reach_max = minkowski_safe(vel_circle, base_pts)

                final = Polygon(reach_acc).intersection(Polygon(reach_max))
                if final.is_empty:
                    continue
                pts = np.array(final.exterior.coords)
                ax.plot(pts[:,0], pts[:,1],
                        linestyle='--',
                        linewidth=0.5,
                        color='red',
                        zorder=100)
                
                pts = reach_max
                pts_closed = np.vstack([pts, pts[0]])  # chiude il contorno
                
                # Debug: plot reach_max in light gray
                # ax.plot(pts_closed[:,0], pts_closed[:,1],
                #         linestyle='--',
                #         linewidth=0.5,
                #         color='lightgray',
                #         zorder=100)
                
                area_improvement = 1- final.area / Polygon(reach_max).area
                # logger.critical(f"Improvement at step {step:.2f}: {area_improvement:.4f} ")
        
        if t > 7.3:
            pts = self.SME_dict[1].Xc
            ax.plot(pts[:,0], pts[:,1],
                    linestyle='-',
                    linewidth=2,
                    color='red',
                    zorder=100000)
            
            for step in [0.0, 0.2, 0.4, 0.6, 0.8]:
                print(step, "second part")
                sec1 = get_annular_sector_points(
                    a_0=self.SME_dict[1].last_vel_meas[0],
                    r_0=self.SME_dict[1].last_vel_meas[1],
                    delta_a=self.dA1,
                    delta_r=self.dR1,
                    future_steps=step
                )
                
                base_pts = pts

                acc_disp   = 0.5 * step**2 * self.max_acc
                acc_poly   = regular_polygon(4, radius=acc_disp)
                vel_circle = regular_polygon(10, radius=self.max_vel * step)

                reach_vel = minkowski_safe(
                    sec1,
                    base_pts
                )
                reach_acc = minkowski_safe(reach_vel, acc_poly)
                reach_max = minkowski_safe(vel_circle, base_pts)

                final = Polygon(reach_acc).intersection(Polygon(reach_max))
                if final.is_empty:
                    continue
                pts = np.array(final.exterior.coords)
                ax.plot(pts[:,0], pts[:,1],
                        linestyle='--',
                        linewidth=0.5,
                        color='red',
                        zorder=100)

