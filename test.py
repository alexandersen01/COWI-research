import pulp
import numpy as np
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt
from shapely.ops import unary_union

class CircleCoverageSolver:
    def __init__(self, room_vertices, grid_size=1.0, circle_radius=1.0, area_cell_size=0.1):
        """
        Initialize the circle coverage optimizer.
        
        Args:
            room_vertices: List of (x,y) tuples defining the room corners
            grid_size: Size of the grid spacing for circle placement
            circle_radius: Radius of each circle
            area_cell_size: Size of cells for discretizing area coverage
        """
        self.room = Polygon(room_vertices)
        self.grid_size = grid_size
        self.circle_radius = circle_radius
        self.area_cell_size = area_cell_size
        self.room_area = self.room.area
        
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
    
    def can_cover(self, circle_center, cell_center):
        """Check if a circle at given center can cover a cell."""
        dx = circle_center[0] - cell_center[0]
        dy = circle_center[1] - cell_center[1]
        return dx*dx + dy*dy <= (self.circle_radius + self.area_cell_size/2)**2
    
    def calculate_exact_coverage(self, circles):
        """Calculate exact area coverage using shapely."""
        if not circles:
            return 0, 0
            
        # Create circle polygons
        circle_polygons = [Point(x, y).buffer(self.circle_radius) for x, y in circles]
        
        # Create union of all circles
        coverage = unary_union(circle_polygons)
        
        # Intersect with room
        covered_area = coverage.intersection(self.room)
        
        # Calculate areas
        total_covered_area = covered_area.area
        coverage_percentage = (total_covered_area / self.room_area) * 100
        
        return total_covered_area, coverage_percentage
    
    def solve(self, coverage_weight=1.0, circle_weight=0.1):
        """
        Solve the optimization problem.
        """
        grid_points = self.generate_grid_points()
        area_cells = self.generate_area_cells()
        
        prob = pulp.LpProblem("CircleCoverage", pulp.LpMaximize)
        
        circle_vars = pulp.LpVariable.dicts("circle",
                                          ((x, y) for x, y in grid_points),
                                          cat='Binary')
        
        cell_vars = pulp.LpVariable.dicts("cell",
                                         ((x, y) for x, y in area_cells),
                                         cat='Binary')
        
        prob += (coverage_weight * sum(cell_vars[x,y] for x,y in area_cells) -
                circle_weight * sum(circle_vars[x,y] for x,y in grid_points))
        
        for cell_x, cell_y in area_cells:
            covering_circles = []
            for grid_x, grid_y in grid_points:
                if self.can_cover((grid_x, grid_y), (cell_x, cell_y)):
                    covering_circles.append(circle_vars[grid_x, grid_y])
            
            if covering_circles:
                prob += cell_vars[cell_x, cell_y] <= sum(covering_circles)
        
        prob.solve()
        
        selected_circles = []
        for x, y in grid_points:
            if circle_vars[x,y].value() > 0.5:
                selected_circles.append((x, y))
        
        covered_cells = []
        for x, y in area_cells:
            if cell_vars[x,y].value() > 0.5:
                covered_cells.append((x, y))
        
        # Calculate exact coverage
        covered_area, coverage_percentage = self.calculate_exact_coverage(selected_circles)
        
        return selected_circles, covered_cells, covered_area, coverage_percentage
    
    def visualize(self, circles, covered_cells, covered_area, coverage_percentage):
        """Visualize the solution with detailed metrics."""
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Plot room
        room_x, room_y = self.room.exterior.xy
        ax.plot(room_x, room_y, 'k-', label='Room boundary')
        
        # Plot covered cells
        for x, y in covered_cells:
            rect = plt.Rectangle((x - self.area_cell_size/2, y - self.area_cell_size/2),
                               self.area_cell_size, self.area_cell_size,
                               color='lightblue', alpha=0.3)
            ax.add_patch(rect)
        
        # Plot circles
        for x, y in circles:
            circle = plt.Circle((x, y), self.circle_radius,
                              fill=False, color='blue', alpha=0.5)
            ax.add_patch(circle)
        
        # Add metrics text
        metrics_text = (
            f'Room Area: {self.room_area:.2f} sq units\n'
            f'Covered Area: {covered_area:.2f} sq units\n'
            f'Coverage: {coverage_percentage:.2f}%\n'
            f'Number of Circles: {len(circles)}\n'
            f'Circle Radius: {self.circle_radius}'
        )
        
        plt.text(0.02, 0.98, metrics_text,
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_aspect('equal')
        plt.grid(True)
        plt.title('Circle Coverage Optimization Results')
        
        # Set limits with some padding
        bounds = self.room.bounds
        plt.xlim(bounds[0] - 1, bounds[2] + 1)
        plt.ylim(bounds[1] - 1, bounds[3] + 1)
        
        plt.show()

# Example usage
if __name__ == "__main__":
    # Define a simple L-shaped room
    room_vertices = [(0,0), (10,0), (10,5), (5,5), (5,10), (0,10)]
    
    solver = CircleCoverageSolver(room_vertices, 
                                grid_size=1.0,
                                circle_radius=2.0,
                                area_cell_size=0.2)
    
    circles, cells, covered_area, coverage_percentage = solver.solve(
        coverage_weight=1.0,
        circle_weight=0.1
    )
    
    print(f"\nOptimization Results:")
    print(f"Number of circles: {len(circles)}")
    print(f"Coverage: {coverage_percentage:.2f}%")
    print(f"Covered area: {covered_area:.2f} square units")
    print(f"Room area: {solver.room_area:.2f} square units")
    
    solver.visualize(circles, cells, covered_area, coverage_percentage)