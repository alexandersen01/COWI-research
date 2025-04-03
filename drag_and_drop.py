import pulp
import numpy as np
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt
from shapely.ops import unary_union
from math import log
from matplotlib.colors import LinearSegmentedColormap

import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class GradientCircleCoverageSolver:
    def __init__(self, room_vertices, grid_size=1.0, circle_radius=1.0, area_cell_size=0.1):
        self.room = Polygon(room_vertices)
        self.grid_size = grid_size
        self.circle_radius = circle_radius
        self.area_cell_size = area_cell_size
        self.room_area = self.room.area

        # Create custom colormap from blue to green
        colors = [
            (0, 0, 1, 0.1),  # Blue (lowest)
            (0, 0.5, 0.5, 0.4),  # Blue-green (low)
            (0, 0.8, 0.2, 0.7),  # Mostly green (medium)
            (0, 1, 0, 1),  # Pure green (high)
        ]
        self.gradient_cmap = LinearSegmentedColormap.from_list("custom", colors)

    def generate_grid_points(self):
        bounds = self.room.bounds
        x_min, y_min, x_max, y_max = bounds
        x_coords = np.arange(x_min, x_max + self.grid_size, self.grid_size)
        y_coords = np.arange(y_min, y_max + self.grid_size, self.grid_size)

        return [(x, y) for x in x_coords for y in y_coords if self.room.intersects(Point(x, y))]

    def generate_area_cells(self):
        bounds = self.room.bounds
        x_min, y_min, x_max, y_max = bounds
        x_coords = np.arange(x_min, x_max + self.area_cell_size, self.area_cell_size)
        y_coords = np.arange(y_min, y_max + self.area_cell_size, self.area_cell_size)

        return [(x, y) for x in x_coords for y in y_coords if self.room.contains(Point(x, y))]

    def get_coverage_value(self, circle_center, cell_center):
        dx = circle_center[0] - cell_center[0]
        dy = circle_center[1] - cell_center[1]
        distance = np.sqrt(dx**2 + dy**2)
        return 1 / (1 + 3 * (distance / self.circle_radius)) if distance <= self.circle_radius else 0

    def solve(self, min_light_level=0.2):
        grid_points = self.generate_grid_points()
        area_cells = self.generate_area_cells()

        prob = pulp.LpProblem("MinimumCircleCoverage", pulp.LpMinimize)
        circle_vars = pulp.LpVariable.dicts("circle", grid_points, cat="Binary")
        cell_vars = pulp.LpVariable.dicts("cell", area_cells, lowBound=0)

        # Objective: Minimize the number of circles
        prob += pulp.lpSum(circle_vars[x, y] for x, y in grid_points)

        for cell_x, cell_y in area_cells:
            covering_circles = [self.get_coverage_value((gx, gy), (cell_x, cell_y)) * circle_vars[gx, gy]
                                for gx, gy in grid_points if self.get_coverage_value((gx, gy), (cell_x, cell_y)) > 0]
            prob += cell_vars[cell_x, cell_y] == pulp.lpSum(covering_circles)

        for gx, gy in grid_points:
            grid_cells = [(x, y) for x, y in area_cells if gx - self.grid_size/2 <= x < gx + self.grid_size/2
                          and gy - self.grid_size/2 <= y < gy + self.grid_size/2]
            if grid_cells:
                prob += pulp.lpSum(cell_vars[x, y] for x, y in grid_cells) >= min_light_level * len(grid_cells)

        prob.solve()
        selected_circles = [(x, y) for x, y in grid_points if circle_vars[x, y].value() > 0.5]
        return selected_circles, {cell: sum(self.get_coverage_value(circle, cell) for circle in selected_circles)
                                  for cell in area_cells}

    def visualize(self, circles, cell_coverage, ax):
        ax.clear()

        # Room boundary
        room_x, room_y = self.room.exterior.xy
        ax.plot(room_x, room_y, "k-", label="Room boundary")

        # Coverage visualization
        max_coverage = max(cell_coverage.values(), default=1.0)
        for (x, y), coverage in cell_coverage.items():
            rect = plt.Rectangle((x - self.area_cell_size / 2, y - self.area_cell_size / 2),
                                 self.area_cell_size, self.area_cell_size,
                                 color=self.gradient_cmap(coverage / max_coverage), alpha=0.8)
            ax.add_patch(rect)

        # Circles
        for x, y in circles:
            ax.add_patch(plt.Circle((x, y), self.circle_radius, fill=False, color="red", alpha=0.5))
            ax.plot(x, y, 'ro', markersize=6, picker=True)

        ax.set_aspect("equal")
        ax.set_title("Gradient Coverage Optimization Results")
        ax.legend()

class CoverageGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Circle Coverage Optimization")

        self.room_vertices = [(0, 0), (10, 0), (10, 5), (5, 10), (0, 10)]
        self.solver = GradientCircleCoverageSolver(self.room_vertices, grid_size=1, circle_radius=4, area_cell_size=0.2)

        self.selected_circles = []

        self.btn_optimal = ttk.Button(root, text="Show Optimal Placement", command=self.show_optimal)
        self.btn_optimal.pack(pady=5)

        self.btn_pattern = ttk.Button(root, text="Show Pattern Placement", command=self.show_pattern)
        self.btn_pattern.pack(pady=5)

        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack()
        self.canvas.mpl_connect("pick_event", self.on_pick)

    def show_optimal(self):
        MIN_LUX = 300
        LAMP_LUX = 318
        MIN_AVG_LUX = MIN_LUX / LAMP_LUX

        self.selected_circles, cell_coverage = self.solver.solve(min_light_level=MIN_AVG_LUX)
        self.solver.visualize(self.selected_circles, cell_coverage, self.ax)
        self.canvas.draw()

    def show_pattern(self):
        self.selected_circles = [(x, y) for x in range(2, 10, 3) for y in range(2, 10, 3) if self.solver.room.contains(Point(x, y))]
        self.solver.visualize(self.selected_circles, {}, self.ax)
        self.canvas.draw()

    def on_pick(self, event):
        if event.artist not in self.ax.patches:
            return
        circle = event.artist.center
        self.selected_circles.remove(circle)
        new_x, new_y = event.mouseevent.xdata, event.mouseevent.ydata
        self.selected_circles.append((new_x, new_y))
        self.solver.visualize(self.selected_circles, {}, self.ax)
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = CoverageGUI(root)
    root.mainloop()
