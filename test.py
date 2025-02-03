import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import math

class CirclePacker:
    def __init__(self, room_vertices, grid_size=1.0):
        """
        Initialize the circle packer with room shape and grid size.
        
        Args:
            room_vertices: List of (x,y) tuples defining the room corners
            grid_size: Size of the grid spacing
        """
        self.room = Polygon(room_vertices)
        self.grid_size = grid_size
        self.circles = []
        
    def generate_grid_points(self):
        """Generate grid points within the room bounds."""
        bounds = self.room.bounds
        x_min, y_min, x_max, y_max = bounds
        
        # Create grid points
        x_coords = np.arange(x_min, x_max + self.grid_size, self.grid_size)
        y_coords = np.arange(y_min, y_max + self.grid_size, self.grid_size)
        grid_points = []
        
        for x in x_coords:
            for y in y_coords:
                point = Point(x, y)
                if self.room.contains(point):
                    grid_points.append((x, y))
                    
        return grid_points
    
    def optimize_circle_radius(self, grid_points, min_radius=0.1, max_radius=None):
        """
        Find the optimal radius for circles placed on grid points.
        Uses binary search to find the largest radius that allows no overlaps.
        """
        if max_radius is None:
            max_radius = self.grid_size / 2
            
        left = min_radius
        right = max_radius
        optimal_radius = min_radius
        
        while right - left > 0.001:  # Precision threshold
            mid = (left + right) / 2
            circles = [Point(p).buffer(mid) for p in grid_points]
            
            # Check for overlaps
            union = unary_union(circles)
            total_area = sum(circle.area for circle in circles)
            
            if abs(union.area - total_area) < 0.001:  # No significant overlap
                optimal_radius = mid
                left = mid
            else:
                right = mid
                
        return optimal_radius
    
    def pack_circles(self):
        """Execute the circle packing algorithm."""
        # Generate grid points within the room
        grid_points = self.generate_grid_points()
        
        # Find optimal radius
        optimal_radius = self.optimize_circle_radius(grid_points)
        
        # Create circles with optimal radius
        circles = [(point, optimal_radius) for point in grid_points]
        self.circles = circles
        
        # Calculate coverage
        total_area = sum([math.pi * r * r for _, r in circles])
        coverage_ratio = total_area / self.room.area
        
        return circles, coverage_ratio
    
    def visualize(self):
        """Visualize the room and packed circles."""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot room
        room_x, room_y = self.room.exterior.xy
        ax.plot(room_x, room_y, 'k-')
        
        # Plot circles
        for (x, y), r in self.circles:
            circle = plt.Circle((x, y), r, fill=False, color='blue')
            ax.add_patch(circle)
            
        # Set equal aspect ratio and display
        ax.set_aspect('equal')
        plt.grid(True)
        plt.title(f'Circle Packing (Coverage: {self.coverage_ratio:.2%})')
        plt.show()

# Example usage
if __name__ == "__main__":
    # Define a simple rectangular room
    room_vertices = [(0, 0), (10, 0), (10, 8), (0, 8)]
    
    # Create packer instance with 1.0 unit grid size
    packer = CirclePacker(room_vertices, grid_size=1.0)
    
    # Execute packing
    circles, coverage = packer.pack_circles()
    
    print(f"Packed {len(circles)} circles")
    print(f"Coverage ratio: {coverage:.2%}")
    
    # Visualize results
    packer.visualize()