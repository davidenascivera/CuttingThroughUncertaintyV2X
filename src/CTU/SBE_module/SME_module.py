import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import box, Polygon
import logging
logger = logging.getLogger(__name__)

USE_LINEAR_ALG = True
if USE_LINEAR_ALG:
    from SBE_module.utils_efficient import minkowski_efficient as minkowski
else:
    from SBE_module.utils import minkowski_safe as minkowski 
from SBE_module.utils import  regular_polygon, reorder_polygon, get_annular_sector_points


class SMEModule:
    def __init__(self, Xc:np.ndarray, delta_a: int, delta_r: float, max_vel = 2, id:int = 10, acc:float = 0.6, dt:float = 0.1, 
                 ax = None, plot_set:bool = False,plot_reach:bool = False, plot_id:bool = False, sens_id:int = 0, verbose:bool = False, 
                 colors:dict[int,str] = {}, track_id:int = -1, err_vel:float = 0, err_heading:float = 0 ) -> None: 
        
        # --- State and prediction ---
        self.Xc: np.ndarray = Xc            # Current set (polygon)
        self.Xp: np.ndarray = None          # Predicted set (to be computed)
        self.last_state_intersection = None # Last intersection state
        self.last_timestamp_intersection: float = -1  # Last valid timestamp for intersection
        self.last_vel_meas = None       # Last velocity measurement (a, v)
        self.centroid:tuple[float,float] = None
        self.track_id = track_id

        # --- Physical model ---
        self.max_vel = max_vel
        self.acc = acc
        self.dt = dt

        # --- Identification and tracking ---
        self.id = id
        self.sensor_id = sens_id
        self.mature_track = False           # Has enough hits to be considered mature
        self.ready_to_delete = False        # Marked for deletion
        self.hits = 0
        self.misses = 0
        self.miss_in_a_row = 0

        # --- Uncertainty modeling ---
        self.delta_a = delta_a              # Angular uncertainty (degrees)
        self.delta_r = delta_r              # Radial uncertainty (meters)

        # --- Plotting and visualization ---
        self.ax = ax                        # Axis for rendering (matplotlib)
        self.plot_set = plot_set            # Show current set
        self.plot_reach = plot_reach        # Show reachable set
        self.plot_id = plot_id              # Show object ID
        self.color_spec = colors[self.sensor_id]
        self.ls_pred = ':'                 # Prediction line style
        self.verbose = verbose
        
        # Note that here we are not considering the bounded heading and vel but we are just considering a square.
        
        self.err_heading = err_heading
        self.err_vel = err_vel
        
    
    @property
    def area(self) -> float:
        """Calculate the area of the current state estimate polygon."""
        if self.Xc is None:
            return 0.0
        return Polygon(self.Xc).area
    
    def predict(self, a_meas:float, v_meas:float, lw_pred = 1, dt_update:float = 0) -> np.ndarray:
        '''
        note that a_meas is in deg and not rad
        Computing the reachable set of the position for the pedestrians. Performed as sequence of minkowski sum 
        
        time needed for the naive is 0.0032 seconds
        '''
        if self.Xp is not None:
            # self.ax.plot(self.Xp[:, 0], self.Xp[:, 1], color=self.color_spec, label='Target Path', zorder=2000, linewidth=0.7, ls ='-')
            centroid_x = np.mean(self.Xp[:, 0])
            centroid_y = np.mean(self.Xp[:, 1])
            # self.ax.scatter(centroid_x, centroid_y, color=self.color_spec, marker='s', label='Target Path', zorder=2000, s=10)
            self._compute_center_of_polygon()
        
        self.last_vel_meas = (a_meas, v_meas)
        # Select the last known intersection state if available; otherwise use the current center.
        last_state = self.last_state_intersection if self.last_state_intersection is not None else self.Xc
        
        # Check if we ever got an intersection. If not, we set the last state to the current state. This happen for clutters 
        if self.last_timestamp_intersection == -1:
            self.last_timestamp_intersection = dt_update
            self.last_state_intersection = last_state
            
            
        # Compute the time horizon for prediction based on last intersection update.
        prediction_time = dt_update -self.last_timestamp_intersection + self.dt
        
        # Estimate the positional uncertainty due to bounded acceleration.
        acc_displacement = 0.5 * prediction_time ** 2 * self.acc
        acc_poly = regular_polygon(4, radius = acc_displacement)
        acc_poly = reorder_polygon(acc_poly)
        
        # Compute a circular velocity limit region based on maximum velocity.
        max_vel_radius = regular_polygon(10, self.max_vel * prediction_time , np.array([0, 0]))

        
        # Step 0 - Compute the annular sector of the sensor. NB that we are considering the same error for the one used for estimating the positon.
        #          So at the end we consider an error in heading of +-5 degrees and an error in distance of +-0.5 m.
        max_vel_poly = get_annular_sector_points(a_0=a_meas,r_0= v_meas, delta_a=self.err_heading, delta_r=self.err_vel, future_steps=prediction_time)
        # Step 1 - Compute the reachable set of the 
        # start_time = time.perf_counter_ns()
        reach_vel = minkowski(max_vel_poly, last_state, plot_intermediate = False)
        # end_time = time.perf_counter_ns()
        # print(f"AAA: Time for the efficient minkowski: {(end_time - start_time)*1e-6} ms")

        # Step 2 - Compute the reachable set of the acceleration
        reac_acc = minkowski(reach_vel, acc_poly, plot_intermediate = False)

        # Step 3 - Compute the reachable set of the max position
        max_vel =  minkowski(max_vel_radius, self.Xc, plot_intermediate = False)
        
        
        
        # Step 4 - Prune the reachable set of the position with the velocity circles
        poly_max_vel = Polygon(max_vel) 
        poly_reach_acc = Polygon(reac_acc)
        poly_intersection = poly_max_vel.intersection(poly_reach_acc)
        
        # # DOUBLE INTEGRATION ENABLER
        # poly_intersection  = poly_max_vel   
        
        if poly_intersection.is_empty:  
            raise ValueError("The reachable set is empty")
        else:
            # Update the current set with the intersection
            self.Xp = np.array(poly_intersection.exterior.coords)
            
            if self.plot_reach: 
                self.ax.plot(self.Xp[:, 0], self.Xp[:, 1], color=self.color_spec, label='Target Path', zorder=2000, linewidth=1, ls =self.ls_pred)
            if self.plot_id:   
                centroid_x = np.mean(self.Xp[:, 0])
                centroid_y = np.mean(self.Xp[:, 1])

                self.ax.text(centroid_x, centroid_y, str(self.id),
                          fontsize=8, color='black', zorder=1,
                          ha='center', va='center',
                          bbox=dict(facecolor='white', edgecolor='none', alpha=0.2))   
            
            self.centroid = self._compute_center_of_polygon()
        self.improve_or_delete_track()
        
        

    
    def intersect(self, meas_set:np.ndarray, lw_int = 2,lw_prev_reach= 0, dt_update:float = 0) -> bool:
        """Computes and visualizes the intersection between the predicted set and new measurement set.

        This method calculates the intersection between the previously predicted reachable set
        and a new measurement set. The intersection is visualized on the plot and becomes the
        new current state estimate.

        Args:
            meas_set: np.ndarray of shape (n,2) containing the vertices of the measurement set polygon
            lw_int: Line width for plotting the intersection polygon (default: 2)
            lw_prev_reach: Line width for plotting the previous reachable set (default: 0)

        Returns:
            bool: True if intersection exists, False if intersection is empty

        """
        
         # This is for the previous reachable set
        self.ax.plot(self.Xp[:, 0], self.Xp[:, 1], color='pink', label='Target Path', zorder=2000, linewidth=lw_prev_reach)

        current_set = self.Xp # Note that we are considering the previus prediction.
        if meas_set is not None:
            poly_Xc = Polygon(current_set)
            poly_meas = Polygon(meas_set)
        
            poly_Xc = poly_Xc.buffer(0)
            poly_meas = poly_meas.buffer(0)
            poly_intersection = poly_Xc.intersection(poly_meas)
            if poly_intersection.is_empty:
                print("No intersection")
                return False
            else:
                # Update the current set with the intersection
                self.Xc = np.array(poly_intersection.exterior.coords)
                self.last_state_intersection = self.Xc
                self.last_timestamp_intersection = dt_update
                self.hits += 1
                self.miss_in_a_row = 0
                if self.plot_set:
                    self.ax.plot(self.Xc[:, 0], self.Xc[:, 1], color=self.color_spec, label='Target Path', zorder=2000, linewidth=lw_int)
                    # if self.mature_track:
                        # logger.critical(f"AREA {self.sensor_id}: {Polygon(self.Xc).area}")
                return True
        else:
            self.misses += 1
            self.miss_in_a_row += 1
            self.Xc = self.Xp
        
        
    def improve_or_delete_track(self):
        if self.hits >= 1 and self.misses == 0 and not self.mature_track:
            logger.info(f"Track that is tracking ped {self.id} from sensor {self.sensor_id} upgraded to mature status")
            self.mature_track = True
            
        if self.misses > 0 and self.hits < 3:
            if self.verbose:
                logger.error(f"track {self.id} non confermato del pedone {self.id}")
            
            self.ready_to_delete = True
            
        if self.miss_in_a_row > 3:
            if self.verbose:
                logger.debug(f"track {self.id} maturo eliminato dal sensore {self.sensor_id}")
            self.ready_to_delete = True
    
    
    def _compute_center_of_polygon(self) -> tuple[float, float]:
        """Compute the centroid of a polygon defined by its vertices.

        Args:
            polygon: np.ndarray of shape (n, 2) containing the vertices of the polygon

        Returns:
            Tuple[float, float]: Centroid coordinates (x, y)
        """
        poly = Polygon(self.Xp)
        return poly.centroid.x, poly.centroid.y

if __name__ == "__main__":
    
    print("test")