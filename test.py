import pulp
import numpy as np
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt
from shapely.ops import unary_union
from math import log
from matplotlib.colors import LinearSegmentedColormap


class GradientCircleCoverageSolver:
    def __init__(
        self, room_vertices, grid_size=1.0, circle_radius=1.0, area_cell_size=0.1
    ):
        self.room = Polygon(room_vertices)
        self.grid_size = grid_size
        self.circle_radius = circle_radius
        self.area_cell_size = area_cell_size
        self.room_area = self.room.area

        # Create custom colormap from green to blue with more intermediate colors
        colors = [
            (0, 0, 1, 0.1),  # blue (lowest)
            (0, 0.5, 0.5, 0.4),  # blue-green (low)
            (0, 0.8, 0.2, 0.7),  # mostly green (medium)
            (0, 1, 0, 1),
        ]  # pure green (high)
        self.gradient_cmap = LinearSegmentedColormap.from_list("custom", colors)

    def generate_grid_points(self):
        """Generate valid grid points within the room."""
        bounds = self.room.bounds
        x_min, y_min, x_max, y_max = bounds

        x_coords = np.arange(x_min, x_max + self.grid_size, self.grid_size)
        y_coords = np.arange(y_min, y_max + self.grid_size, self.grid_size)

        grid_points = []
        for x in x_coords:
            for y in y_coords:
                point = Point(x, y)
                if self.room.intersects(point):
                    grid_points.append((x, y))

        return grid_points

    def generate_area_cells(self):
        """Generate cells for tracking area coverage."""
        bounds = self.room.bounds
        x_min, y_min, x_max, y_max = bounds

        x_coords = np.arange(x_min, x_max + self.area_cell_size, self.area_cell_size)
        y_coords = np.arange(y_min, y_max + self.area_cell_size, self.area_cell_size)

        cells = []
        for x in x_coords:
            for y in y_coords:
                point = Point(x, y)
                if self.room.contains(point):
                    cells.append((x, y))

        return cells

    def get_coverage_value(self, circle_center, cell_center):
        """Calculate gradient coverage value based on distance from circle center."""
        dx = circle_center[0] - cell_center[0]
        dy = circle_center[1] - cell_center[1]
        distance = np.sqrt(dx * dx + dy * dy)

        if distance <= self.circle_radius:
            normalized_distance = distance / self.circle_radius
            return 1 / (1 + 0.8 * normalized_distance)

        return 0

    def get_cells_in_grid_cell(self, grid_x, grid_y, area_cells):
        """Get all small area cells that fall within a given grid cell."""
        min_x = grid_x - self.grid_size / 2
        max_x = grid_x + self.grid_size / 2
        min_y = grid_y - self.grid_size / 2
        max_y = grid_y + self.grid_size / 2

        return [
            (x, y) for x, y in area_cells if min_x <= x < max_x and min_y <= y < max_y
        ]

    def solve(self, min_light_level=0.2):
        """
        Solve the optimization problem to minimize number of circles while maintaining
        minimum average coverage in each grid cell.

        Args:
            min_light_level (float): Minimum average light level required in each grid cell (default: 0.2)
        """
        grid_points = self.generate_grid_points()
        area_cells = self.generate_area_cells()

        prob = pulp.LpProblem("MinimumCircleCoverage", pulp.LpMinimize)

        # Binary variables for circle placement
        circle_vars = pulp.LpVariable.dicts(
            "circle", ((x, y) for x, y in grid_points), cat="Binary"
        )

        # Continuous variables for cell light levels
        cell_vars = pulp.LpVariable.dicts(
            "cell", ((x, y) for x, y in area_cells), lowBound=0
        )

        # Objective: Minimize number of circles
        prob += pulp.lpSum(circle_vars[x, y] for x, y in grid_points)

        # Set up constraints for small cell values based on circle coverage
        for cell_x, cell_y in area_cells:
            covering_circles = []
            for grid_x, grid_y in grid_points:
                coverage_value = self.get_coverage_value(
                    (grid_x, grid_y), (cell_x, cell_y)
                )
                if coverage_value > 0:
                    covering_circles.append(
                        coverage_value * circle_vars[grid_x, grid_y]
                    )

            prob += cell_vars[cell_x, cell_y] == pulp.lpSum(covering_circles)

        # Add minimum average constraint for each grid cell using the parameter
        for grid_x, grid_y in grid_points:
            grid_cells = self.get_cells_in_grid_cell(grid_x, grid_y, area_cells)
            if grid_cells:  # Only add constraint if grid cell has area cells
                prob += pulp.lpSum(
                    cell_vars[x, y] for x, y in grid_cells
                ) >= min_light_level * len(grid_cells)
                # prob += pulp.lpSum(
                #     cell_vars[x, y] for x, y in grid_cells
                # ) <= max_light_level * len(grid_cells)

        status = prob.solve()
        print(f"Solver status: {pulp.LpStatus[status]}")

        if status != 1:  # If not optimal
            print("Warning: Solver did not find an optimal solution.")
            print(
                f"Try adjusting the minimum light requirement (currently {min_light_level}) or grid/circle parameters."
            )

        # Extract results
        selected_circles = []
        for x, y in grid_points:
            if circle_vars[x, y].value() > 0.5:
                selected_circles.append((x, y))

        # Calculate light levels for visualization and verification
        cell_coverage = {}
        grid_averages = {}
        for grid_x, grid_y in grid_points:
            grid_cells = self.get_cells_in_grid_cell(grid_x, grid_y, area_cells)
            grid_total = 0

            for cell_x, cell_y in grid_cells:
                cell_total = 0
                for circle_x, circle_y in selected_circles:
                    coverage = self.get_coverage_value(
                        (circle_x, circle_y), (cell_x, cell_y)
                    )
                    cell_total += coverage
                cell_coverage[(cell_x, cell_y)] = cell_total
                grid_total += cell_total

            if grid_cells:
                grid_avg = grid_total / len(grid_cells)
                grid_averages[(grid_x, grid_y)] = grid_avg
                print(f"Grid cell ({grid_x:.1f}, {grid_y:.1f}) average: {grid_avg:.3f}")

        return selected_circles, cell_coverage

    def visualize(self, circles, cell_coverage):
        """Visualize the solution with gradient coverage."""
        fig, ax = plt.subplots(figsize=(12, 12))

        # Plot room boundary
        room_x, room_y = self.room.exterior.xy
        ax.plot(room_x, room_y, "k-", label="Room boundary")

        # Plot gradient coverage
        max_coverage = max(cell_coverage.values()) if cell_coverage else 1.0
        if max_coverage == 0:
            max_coverage = 1.0

        for (x, y), coverage in cell_coverage.items():
            normalized_coverage = coverage / max_coverage
            rect = plt.Rectangle(
                (x - self.area_cell_size / 2, y - self.area_cell_size / 2),
                self.area_cell_size,
                self.area_cell_size,
                color=self.gradient_cmap(normalized_coverage),
                alpha=0.8,
            )
            ax.add_patch(rect)

        # Plot grid cells for reference
        grid_points = self.generate_grid_points()
        for x, y in grid_points:
            rect = plt.Rectangle(
                (x - self.grid_size / 2, y - self.grid_size / 2),
                self.grid_size,
                self.grid_size,
                fill=False,
                color="gray",
                linestyle="--",
                alpha=0.5,
            )
            ax.add_patch(rect)

        # Plot circles
        for x, y in circles:
            circle = plt.Circle(
                (x, y), self.circle_radius, fill=False, color="red", alpha=0.5
            )
            ax.add_patch(circle)

        # Add metrics text
        metrics_text = (
            f"Room Area: {self.room_area:.2f} sq units\n"
            f"Number of Circles: {len(circles)}\n"
            f"Circle Radius: {self.circle_radius}"
        )

        plt.text(
            0.02,
            0.98,
            metrics_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        ax.set_aspect("equal")
        plt.grid(True)
        plt.title("Gradient Coverage Optimization Results")

        # Set limits with padding
        bounds = self.room.bounds
        plt.xlim(bounds[0] - 1, bounds[2] + 1)
        plt.ylim(bounds[1] - 1, bounds[3] + 1)

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=self.gradient_cmap)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="Coverage Intensity")

        plt.show()


if __name__ == "__main__":
    # Simple L-shaped room
    # room_vertices = [(0, 0), (10, 0), (10, 5), (5, 5), (5, 10), (0, 10)]
    room_vertices = [(0, 0), (0, 10), (10, 10), (10, 0)]

    solver = GradientCircleCoverageSolver(
        room_vertices, grid_size=1, circle_radius=2, area_cell_size=0.2
    )
    
    MIN_LUX = 300
    LAMP_LUX = 318
    MIN_AVG_LUX = MIN_LUX / LAMP_LUX

    # Now you can specify different minimum light levels
    circles, cell_coverage = solver.solve(min_light_level=MIN_AVG_LUX)

    print(f"\nOptimization Results:")
    print(f"Number of circles: {len(circles)}")
    print(f"Room area: {solver.room_area:.2f} square units")

    solver.visualize(circles, cell_coverage)
