import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import csv
import matplotlib.patches as patches

from utils import ray_intersects_rectangle

class MapRenderer:
    def __init__(self, fig_size=None):
        """
        Initialize the Matplotlib figure and axis in interactive mode.
        """
        if fig_size:
            self.fig, self.ax = plt.subplots(figsize=fig_size)
        else:
            self.fig, self.ax = plt.subplots()
        plt.ion()
        self.ax.set_facecolor('white')
        self.ax.grid(False)
        self.log_data = False
        self.track = []

    def init_csv(self, filename='simulation_results.csv', header=None, log_data=False):
        # Create or initialize CSV file
        if log_data:
            self.log_data = True    # Enable logging
            with open(filename, 'w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                if header is not None:  # Only write header if it's provided
                    csv_writer.writerow(header)
        return filename

    def log_to_csv(self, filename, data):
        # Append data to CSV file
        if self.log_data:
            with open(filename, 'a', newline='') as csv_file:
            
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(data)

    def clear_scene(self):
        """
        Clears the current axes so we can draw a fresh frame.
        """
        self.ax.clear()
        self.ax.set_facecolor('white')
        self.ax.grid(False)

    def draw_road(self, road_start_x, road_start_y, road_length, lane_width, num_lanes, shoulder_width):
        """
        Draws the road and returns (y_top_edge, y_bottom_edge).
        """
        road_total_width = num_lanes * lane_width + 2 * shoulder_width
        road_pos_y       = road_start_y - road_total_width / 2

        # Road rectangle
        road = patches.Rectangle(
            (road_start_x, road_pos_y),
            road_length,
            road_total_width,
            facecolor='gray',
            edgecolor='none'
        )
        self.ax.add_patch(road)

        # Lane dividing lines
        for lane_idx in range(1, num_lanes):
            y_line = road_start_y + lane_width/2 * (2*lane_idx - num_lanes)
            self.ax.plot(
                [road_start_x, road_start_x + road_length],
                [y_line, y_line],
                '--', color='white', linewidth=2
            )

        # Shoulder lines
        y_top_edge    = road_start_y + (road_total_width / 2)
        y_bottom_edge = road_start_y - (road_total_width / 2)

        self.ax.plot(
            [road_start_x, road_start_x + road_length],
            [y_top_edge, y_top_edge],
            color='white', linewidth=2
        )
        self.ax.plot(
            [road_start_x, road_start_x + road_length],
            [y_bottom_edge, y_bottom_edge],
            color='white', linewidth=2
        )

        return y_top_edge, y_bottom_edge

    def draw_vehicle(self, position, length, width, color, label, zorder=2, label_color='white'):
        """
        Draws a rectangle for a vehicle plus a label.
        """
        rect = patches.Rectangle(
            (position[0] - length/2, position[1] - width/2),
            length,
            width,
            facecolor=color,
            edgecolor='black',
            zorder=zorder
        )
        self.ax.add_patch(rect)
        self.ax.text(
            position[0],
            position[1],
            label,
            color=label_color,
            ha='center',
            va='center',
            fontsize=10,
            fontweight='bold',
            zorder=zorder + 1
        )

    def draw_pedestrian(self, position: tuple[float, float], color: str, label: str) -> None:
        """
        Draws a pedestrian as a scatter point with a label.
        """
        # Ensure position values are cast to float for robustness
        x, y = float(position[0]), float(position[1])

        # Ensure color and label are explicitly strings
        color = str(color)
        label = str(label)

        # Plot pedestrian as a scatter point
        self.ax.scatter(x, y, s=10, color=color, zorder=1000)

        # Add label next to the pedestrian
        self.ax.text(
            x + 0.5,  # Slightly shift text in the x direction
            y,
            label,
            color=color,
            fontweight='bold',
            zorder=4
        )


    def draw_pedestrian_2(self, position, color, label):
        """
        Draws a pedestrian with a surrounding circle of radius 0.3 at each past position in track.
        """
        # Store the position in the track list
        self.track.append(position)  
    
        # Draw scatter point for the current position
        self.ax.scatter(*position, s=30, color=color, zorder=4)
    
        # Plot circles for all past track positions
        for past_pos in self.track:
            circle = patches.Circle(past_pos, radius=0.25, edgecolor=color, facecolor='none', linewidth=1, zorder=2)
            self.ax.add_patch(circle)
    
        # Add label near the pedestrian
        self.ax.text(
            position[0] + 0.5,
            position[1],
            label,
            color=color,
            fontweight='bold',
            zorder=4
        )

    def draw_rsu(self, position, width, height, color, label):
        """
        Draws an RSU (rectangle + label).
        """
        rsu_rect = patches.Rectangle(
            position,
            width,
            height,
            facecolor=color,
            edgecolor='black',
            zorder=2
        )
        self.ax.add_patch(rsu_rect)
        self.ax.text(
            position[0] + width/2,
            position[1] + height/2,
            label,
            color='white',
            ha='center',
            va='center',
            fontweight='bold',
            zorder=4
        )

    def draw_rays(self, vehicles: list[dict[str, any]], vru_objects: list[dict[str, any]], alpha: float = 0.2, 
                  vru_occlusion_shadow: bool = False, ray_cfg:any = None,ray_origin:tuple = (0,0)) -> list[dict[str, any]]:
        """
        Draws multiple rays from a given origin. Each ray stops at the first intersection
        with any object (vehicles or VRUs). For VRUs, the test is performed by projecting the
        VRU’s center onto the ray. Additionally, the VRU can be optionally counted as an occlusion.

        Returns:
            list[dict]: A list of VRU dictionaries that were hit by one or more rays.
                        Each dictionary has the same structure as the elements in `vru_objects`, i.e.:
                        {
                            'pos': Tuple[float, float],   # The (x, y) position of the VRU
                            'id': int,                    # Unique identifier of the VRU
                            'head': float,                # Heading angle (in radians or degrees)
                            'vel': float                  # Linear velocity (in m/s)
                        }
        """
        if ray_cfg is not None and ray_cfg.get('origin') is not None:
            ray_origin = ray_cfg['origin']
            
        fan_deg = ray_cfg['fan_deg']
        n_rays = ray_cfg['n_rays']
        ray_length = ray_cfg['ray_length']
        heading = ray_cfg['heading']
        color = ray_cfg['color']
        
        # Calculate the angles for the rays
        angles = np.linspace(-np.radians(fan_deg)/2, np.radians(fan_deg)/2, n_rays) + np.radians(heading)

        list_of_vru_intersections = []
        # Set a detection "radius" for VRUs (half-width)
        vru_radius = 0.25  # adjust as needed

        for angle in angles:
            ray_dir = np.array([np.cos(angle), np.sin(angle)])
            # By default, assume the ray goes the full length.
            ray_end = np.array(ray_origin) + ray_dir * ray_length
            min_dist = ray_length
            hit_obj = None

            # --- First check intersections with vehicles ---
            for obj in vehicles:
                intersects, intersection_pt = ray_intersects_rectangle(
                    np.array(ray_origin),
                    ray_dir,
                    obj['pos'],
                    obj['width'],
                    obj['height']
                )
                if intersects:
                    dist = np.linalg.norm(intersection_pt - np.array(ray_origin))
                    if dist < min_dist:
                        min_dist = dist
                        ray_end = intersection_pt
                        hit_obj = obj

            # --- Now check VRU objects (if provided) ---
            if vru_objects is not None:
                for vru in vru_objects:
                    vru_pos = np.array(vru['pos'])
                    vec_to_vru = vru_pos - np.array(ray_origin)
                    proj_length = np.dot(vec_to_vru, ray_dir)
                    if proj_length < 0 or proj_length > ray_length:
                        continue
                    perp_dist = np.linalg.norm(vec_to_vru - proj_length * ray_dir)
                    # Add check to ensure VRU is before the vehicle hit.
                    if perp_dist <= vru_radius and proj_length < min_dist:
                        if vru not in list_of_vru_intersections:
                            list_of_vru_intersections.append(vru)
                        if vru_occlusion_shadow:
                            min_dist = proj_length
                            ray_end = np.array(ray_origin) + ray_dir * proj_length
                            hit_obj = vru


            # (Optional) If the hit object is a VRU and it wasn't already added, add it.
            # This branch will only matter if vru_occlusion is True.
            if hit_obj is not None and hit_obj.get('type', None) == 'VRU':
                if hit_obj not in list_of_vru_intersections:
                    list_of_vru_intersections.append(hit_obj)

            # Draw the ray from origin to the determined end point.
            self.ax.plot(
                [ray_origin[0], ray_end[0]],
                [ray_origin[1], ray_end[1]],
                color=color,
                alpha=alpha
            )

        return list_of_vru_intersections

    

    def render_frame(self, x_limits, y_limits, title, xlabel, ylabel, scale=1.0, recording=False, dt=0, screen_dt=0.1):
        """
        Sets axis limits, labels, and updates the figure in interactive mode.
        Extra margins are removed when recording=True, so that the exported image 
        corresponds exactly to the plot area. When recording=False, margins are left 
        so that the axes (with units in meters) are visible.
        """
        # Calculate the aspect ratio based on the limits.
        x_range = x_limits[1] - x_limits[0]
        y_range = y_limits[1] - y_limits[0]
        
        width = x_range * scale
        height = y_range * scale
        self.fig.set_size_inches(width, height)
        
        self.ax.set_aspect('equal')
        self.ax.set_xlim(x_limits)
        self.ax.set_ylim(y_limits)
        self.ax.autoscale(enable=False)
        
        # If recording, remove margins so the axes fill the figure.
        # Otherwise, keep margins so the axes and their labels (with units) are visible.
        if recording:
            self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        else:
            self.fig.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
        
        title_annotate = title + (" (Recording)" if recording else "")
        self.ax.set_title(title_annotate)
        
        # Update axis labels: add unit ("m") if not recording.
        if not recording:
            unit = "m"  # Unit of measurement
            self.ax.set_xlabel(f"{xlabel} [{unit}]")
            self.ax.set_ylabel(f"{ylabel} [{unit}]")
        else:
            self.ax.set_xlabel(xlabel)
            self.ax.set_ylabel(ylabel)
        
        plt.draw()
        plt.pause(0.01)
        
        if recording:
            self.fig.savefig(f"frames/timestep_{dt:.2f}.png", dpi=300)
            if dt == screen_dt:
                self.fig.savefig("exported_canvas.png", dpi=300)
    


        

    def finalize(self):
        """
        Turn off interactive mode and show the final result in a blocking window.
        """
        plt.ioff()
        plt.show()




