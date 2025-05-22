import tkinter as tk
from PIL import Image, ImageTk
import json
import math
import os

def sample_polyline(polyline, dt=0.1, velocity=0.0):
    """
    Given a list of (x, y) points defining a polyline, sample the trajectory at constant
    time intervals dt assuming constant velocity.
    
    Returns a list of dictionaries with:
      - "dt": sample index,
      - "timestamp": time (seconds),
      - "x": interpolated x coordinate,
      - "y": interpolated y coordinate.
    """
    if len(polyline) < 2:
        return [{"dt": 0, "timestamp": 0.0, "x": polyline[0][0], "y": polyline[0][1]}]
    
    cum_dist = [0.0]
    for i in range(1, len(polyline)):
        x0, y0 = polyline[i-1]
        x1, y1 = polyline[i]
        d = math.hypot(x1 - x0, y1 - y0)
        cum_dist.append(cum_dist[-1] + d)
    
    total_length = cum_dist[-1]
    total_time = total_length / velocity if velocity != 0 else 0
    timestamps = [round(t, 2) for t in frange(0, total_time, dt)]
    
    samples = []
    seg_idx = 0
    for idx, t in enumerate(timestamps):
        d_target = velocity * t
        while seg_idx < len(cum_dist) - 1 and cum_dist[seg_idx+1] < d_target:
            seg_idx += 1
        if seg_idx >= len(polyline) - 1:
            samples.append({
                "dt": idx,
                "timestamp": t,
                "x": polyline[-1][0],
                "y": polyline[-1][1]
            })
            continue

        d0 = cum_dist[seg_idx]
        d1 = cum_dist[seg_idx+1]
        f = 0 if d1 - d0 == 0 else (d_target - d0) / (d1 - d0)
        x0, y0 = polyline[seg_idx]
        x1, y1 = polyline[seg_idx+1]
        x_interp = x0 + f * (x1 - x0)
        y_interp = y0 + f * (y1 - y0)
        
        samples.append({
            "dt": idx,
            "timestamp": t,
            "x": x_interp,
            "y": y_interp
        })
        
    return samples

def frange(start, stop, step):
    t = start
    while t <= stop:
        yield t
        t += step

class DrawOnImageApp:
    def __init__(self, root, velocity=0, window_scale=1):
        self.root = root
        self.root.title("Draw on Exported Plot")

        # These limits must match those used in the MapRenderer.
        self.x_limits = (-2, 44)
        self.y_limits = (-10, 10)

        # Load the exported image (should have no extra margins)
        self.img = Image.open("exported_canvas.png")
        
        # Use the image’s native resolution as the base dimensions.
        self.base_width, self.base_height = self.img.size

        # Scale the image for display.
        display_width = int(self.base_width * window_scale)
        display_height = int(self.base_height * window_scale)
        self.window_scale = window_scale

        self.display_img = self.img.resize((display_width, display_height), resample=Image.Resampling.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(self.display_img)

        self.canvas = tk.Canvas(root, width=display_width, height=display_height)
        self.canvas.pack()
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)

        self.trajectory_pixels = []
        self.drawing = False

        self.canvas.bind("<ButtonPress-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.draw_trajectory)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)

        self.velocity = velocity

    def start_drawing(self, event):
        self.drawing = True
        self.trajectory_pixels = [(event.x, event.y)]
        print(f"Start (display pixel): ({event.x}, {event.y})")
        print(f"Start (plot): {self.pixel_to_plot(event.x, event.y)}")

    def draw_trajectory(self, event):
        if self.drawing:
            self.canvas.create_line(
                self.trajectory_pixels[-1][0],
                self.trajectory_pixels[-1][1],
                event.x,
                event.y,
                fill="red",
                width=2
            )
            self.trajectory_pixels.append((event.x, event.y))
            print(f"Moving (display pixel): ({event.x}, {event.y})")
            print(f"Moving (plot): {self.pixel_to_plot(event.x, event.y)}")

    def stop_drawing(self, event):
        self.drawing = False
        print("Final Trajectory Points (display pixel):")
        for point in self.trajectory_pixels:
            print(point)
        
        trajectory_plot = [self.pixel_to_plot(x, y) for (x, y) in self.trajectory_pixels]
        print("\nFinal Trajectory Points (plot coordinates):")
        for point in trajectory_plot:
            print(point)
        
        trajectory_samples = sample_polyline(trajectory_plot, dt=0.1, velocity=self.velocity)
        
        # Save in the same directory as the script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(script_dir, "trajectory.json")
        
        with open(json_path, "w") as f:
            json.dump(trajectory_samples, f, indent=4)
        
        print(f"\nTrajectory saved to {json_path}")

    def pixel_to_plot(self, x_pixel, y_pixel):
        """
        Converts display (Tkinter) pixel coordinates to plot coordinates.
        Assumes that the exported image exactly corresponds to the logical plot (i.e. no extra margins).
        """
        # Convert the pixel coordinates from the displayed (scaled) image back to the base image.
        x_base = x_pixel / self.window_scale
        y_base = y_pixel / self.window_scale

        x_min, x_max = self.x_limits
        y_min, y_max = self.y_limits

        # The x-coordinate scales directly.
        x_plot = x_min + (x_base / self.base_width) * (x_max - x_min)
        # For y, note that in the exported image the top corresponds to y_max.
        y_plot = y_max - (y_base / self.base_height) * (y_max - y_min)
        return (x_plot, y_plot)

if __name__ == "__main__":
    root = tk.Tk()
    # Adjust window_scale as desired (this scales the display only; the coordinate conversion still uses the base image).
    app = DrawOnImageApp(root, velocity=1.2, window_scale=0.5)
    root.mainloop()
