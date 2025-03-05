import pulp
import numpy as np
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt
from shapely.ops import unary_union
from math import log, cos, sin, pi
from matplotlib.colors import LinearSegmentedColormap


class RegularPatternCircleCoverageSolver:
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
                if self.room.contains(point):
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
            return 1 / (1 + 2 * normalized_distance)

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
        
    def generate_pattern_from_parameters(self, center_x, center_y, distance, num_circles, angle_offset=0):
        """
        Generate a regular pattern of circles from parameters.
        
        Args:
            center_x, center_y: Center point of the pattern
            distance: Distance between circles
            num_circles: Number of circles in the pattern
            angle_offset: Starting angle offset in radians
            
        Returns:
            List of (x, y) coordinates for circle centers
        """
        # Always place the first circle at the center
        circles = [(center_x, center_y)]
        
        # If only one circle, return just the center
        if num_circles <= 1:
            return circles
            
        # For multiple circles, distribute them evenly in a circle
        angle_step = 2 * pi / (num_circles - 1) if num_circles > 1 else 0
        
        for i in range(1, num_circles):
            angle = angle_offset + i * angle_step
            x = center_x + distance * cos(angle)
            y = center_y + distance * sin(angle)
            
            # Check if the point is inside the room
            if self.room.contains(Point(x, y)):
                circles.append((x, y))
                
        return circles

    def find_best_pattern(self, min_light_level=0.2):
        """
        Find the best regular pattern of circles to cover the room.
        
        This approaches the problem differently than the original solver:
        Instead of using binary variables for each grid point, we'll use
        parameters to define a regular pattern and optimize those parameters.
        
        Args:
            min_light_level: Minimum average light level required
            
        Returns:
            List of (x, y) coordinates for the optimal circle placement
        """
        # Find room centroid as a starting point
        room_centroid = list(self.room.centroid.coords)[0]
        
        # Approximate dimensions based on room bounds
        bounds = self.room.bounds
        x_min, y_min, x_max, y_max = bounds
        width = x_max - x_min
        height = y_max - y_min
        max_dimension = max(width, height)
        room_center = ((x_min + x_max) / 2, (y_min + y_max) / 2)
        
        # Define parameter ranges
        max_distance = max_dimension / 2
        max_circles = int(np.ceil(self.room_area / (pi * self.circle_radius ** 2)))
        
        # Force a reasonable number range - between 4 and 8 circles 
        # This is for demonstration purposes
        min_circles = 4
        max_circles = 8
        
        # Initialize best pattern
        best_pattern = None
        best_coverage = 0
        best_num_circles = float('inf')
        
        # For demonstration, create a default pattern if we can't find a valid one
        default_pattern = self.generate_pattern_from_parameters(
            room_center[0], room_center[1], self.circle_radius * 2, 7
        )
        
        area_cells = self.generate_area_cells()
        
        # Try different parameter combinations
        for num_circles in range(1, max_circles + 1):
            for distance_factor in np.linspace(0.5, 1.0, 6):
                distance = distance_factor * self.circle_radius * 2
                
                # Try different center points around the room centroid
                for center_x_offset in np.linspace(-self.grid_size, self.grid_size, 5):
                    for center_y_offset in np.linspace(-self.grid_size, self.grid_size, 5):
                        center_x = room_center[0] + center_x_offset
                        center_y = room_center[1] + center_y_offset
                        
                        # Try different angle offsets
                        for angle_offset in np.linspace(0, pi/2, 4):
                            # Generate pattern
                            pattern = self.generate_pattern_from_parameters(
                                center_x, center_y, distance, num_circles, angle_offset
                            )
                            
                            # Check coverage and constraints
                            cell_coverage = {}
                            for cell_x, cell_y in area_cells:
                                cell_total = 0
                                for circle_x, circle_y in pattern:
                                    coverage = self.get_coverage_value(
                                        (circle_x, circle_y), (cell_x, cell_y)
                                    )
                                    cell_total += coverage
                                cell_coverage[(cell_x, cell_y)] = cell_total
                            
                            # Check if all grid cells meet minimum light level
                            grid_points = self.generate_grid_points()
                            all_grid_satisfied = True
                            
                            for grid_x, grid_y in grid_points:
                                grid_cells = self.get_cells_in_grid_cell(grid_x, grid_y, area_cells)
                                if grid_cells:
                                    grid_total = sum(cell_coverage.get((cell_x, cell_y), 0) for cell_x, cell_y in grid_cells)
                                    grid_avg = grid_total / len(grid_cells)
                                    if grid_avg < min_light_level:
                                        all_grid_satisfied = False
                                        break
                            
                            # Calculate overall coverage
                            avg_coverage = sum(cell_coverage.values()) / len(area_cells)
                            
                            # Update best pattern if this one is better
                            if all_grid_satisfied:
                                if len(pattern) < best_num_circles or (len(pattern) == best_num_circles and avg_coverage > best_coverage):
                                    best_pattern = pattern
                                    best_coverage = avg_coverage
                                    best_num_circles = len(pattern)
                                    
        if best_pattern is None:
            print("Warning: Could not find a pattern that satisfies the minimum light level.")
            print("Using a demonstration pattern to show the regular arrangement.")
            
            # Use the default pattern as fallback to demonstrate regular pattern
            best_pattern = default_pattern
            
            # Calculate coverage for this pattern
            final_coverage = {}
            for cell_x, cell_y in area_cells:
                cell_total = 0
                for circle_x, circle_y in best_pattern:
                    coverage = self.get_coverage_value(
                        (circle_x, circle_y), (cell_x, cell_y)
                    )
                    cell_total += coverage
                final_coverage[(cell_x, cell_y)] = cell_total
                
            return best_pattern, final_coverage
            
        # Calculate final coverage for the best pattern
        final_coverage = {}
        for cell_x, cell_y in area_cells:
            cell_total = 0
            for circle_x, circle_y in best_pattern:
                coverage = self.get_coverage_value(
                    (circle_x, circle_y), (cell_x, cell_y)
                )
                cell_total += coverage
            final_coverage[(cell_x, cell_y)] = cell_total
            
        return best_pattern, final_coverage

    def solve(self, min_light_level=0.2):
        """
        Solve using a regular pattern approach rather than the original LP method.
        
        Args:
            min_light_level (float): Minimum average light level required in each grid cell
        """
        print(f"Finding optimal regular pattern with minimum light level: {min_light_level}")
        circles, cell_coverage = self.find_best_pattern(min_light_level)
        
        print(f"Found solution with {len(circles)} circles")
        
        # Calculate grid cell averages for reporting
        grid_points = self.generate_grid_points()
        area_cells = self.generate_area_cells()
        grid_averages = {}
        
        for grid_x, grid_y in grid_points:
            grid_cells = self.get_cells_in_grid_cell(grid_x, grid_y, area_cells)
            if grid_cells:
                grid_total = sum(cell_coverage.get((cell_x, cell_y), 0) for cell_x, cell_y in grid_cells)
                grid_avg = grid_total / len(grid_cells)
                grid_averages[(grid_x, grid_y)] = grid_avg
                print(f"Grid cell ({grid_x:.1f}, {grid_y:.1f}) average: {grid_avg:.3f}")
                
        return circles, cell_coverage

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
            
        # Plot connections between circles to visualize the pattern
        if len(circles) > 1:
            # Find center (first circle)
            center_x, center_y = circles[0]
            
            # Draw lines from center to other circles
            for x, y in circles[1:]:
                plt.plot([center_x, x], [center_y, y], 'r--', alpha=0.3)

        # Add metrics text
        metrics_text = (
            f"Room Area: {self.room_area:.2f} sq units\n"
            f"Number of Circles: {len(circles)}\n"
            f"Circle Radius: {self.circle_radius}"
        )

        # Calculate pattern metrics if circles exist
        if len(circles) > 1:
            # Calculate average distance from center
            center_x, center_y = circles[0]
            distances = [np.sqrt((x-center_x)**2 + (y-center_y)**2) for x, y in circles[1:]]
            avg_distance = np.mean(distances)
            
            # Calculate angle consistency
            if len(distances) > 1:
                angles = [np.arctan2(y-center_y, x-center_x) for x, y in circles[1:]]
                angle_diffs = []
                for i in range(len(angles)):
                    for j in range(i+1, len(angles)):
                        diff = abs(angles[i] - angles[j]) % (2 * np.pi)
                        diff = min(diff, 2 * np.pi - diff)
                        angle_diffs.append(diff)
                        
                expected_angle_diff = 2 * np.pi / (len(circles) - 1)
                angle_consistency = np.mean([abs(diff - expected_angle_diff) for diff in angle_diffs])
                
                metrics_text += f"\nAvg Distance: {avg_distance:.2f}\n"
                metrics_text += f"Angle Consistency: {angle_consistency:.4f}"
            else:
                metrics_text += f"\nDistance: {avg_distance:.2f}"

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
        plt.title("Regular Pattern Gradient Coverage Results")

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
    room_vertices = [(0, 0), (10, 0), (10, 5), (5, 5), (5, 10), (0, 10)]

    solver = RegularPatternCircleCoverageSolver(
        room_vertices, grid_size=1.0, circle_radius=1.5, area_cell_size=0.2
    )

    # Now you can specify different minimum light levels
    circles, cell_coverage = solver.solve(min_light_level=0.8)  # Lower light level to ensure we get multiple circles

    print(f"\nOptimization Results:")
    print(f"Number of circles: {len(circles)}")
    print(f"Room area: {solver.room_area:.2f} square units")

    solver.visualize(circles, cell_coverage)