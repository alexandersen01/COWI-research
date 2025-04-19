import numpy as np
import pulp
from shapely.geometry import Polygon, Point
from math import log
from matplotlib.colors import LinearSegmentedColormap
import json

# Extracts the json rooms from COWI
class RoomModel:
    """Model for room data and operations."""
    def __init__(self):
        self.rooms = {}
        
    def load_json_rooms(self, possible_paths=None):
        """Load rooms from the JSON file."""
        if possible_paths is None:
            possible_paths = [
                'spatial_elements_boundaries.json',
                './spatial_elements_boundaries.json',
                '../spatial_elements_boundaries.json',
                'spatial_elements_boundaries'  # Without extension
            ]
            
        try:
            data = None
            for path in possible_paths:
                try:
                    with open(path, 'r') as f:
                        data = json.load(f)
                    return True, f"Found and loaded JSON file: {path}", data
                except FileNotFoundError:
                    continue
            
            if data is None:
                return False, f"JSON file not found in any of the tried paths", None
            
        except json.JSONDecodeError:
            return False, f"Invalid JSON format in file", None
        except Exception as e:
            return False, f"Error loading JSON file: {str(e)}", None
    
    def process_room_data(self, data):
        """Process raw room data into room coordinates."""
        try:
            # Process rooms
            for room_id, room_data in data.items():
                # Extract 2D coordinates (ignore z-coordinate)
                coordinates = [(point[0], point[1]) for point in room_data[0]]
                self.rooms[room_id] = coordinates
            
            if self.rooms:
                return True, f"Successfully loaded {len(self.rooms)} rooms from JSON"
            else:
                return False, "No rooms were successfully parsed from the JSON file"
        except Exception as e:
            return False, f"Error processing room data: {str(e)}"
    
    def get_room_ids(self):
        """Return list of room IDs."""
        return list(self.rooms.keys())
    
    def get_room_coordinates(self, room_id):
        """Return coordinates for a specific room ID."""
        return self.rooms.get(room_id, [])
    
    def normalize_coordinates(self, coords, scale_factor=10, meters_per_unit=0.2):
        """
        Normalize room coordinates to a reasonable size.
        
        Args:
            coords: List of coordinate points
            scale_factor: Division factor to reduce the size of coordinates
            meters_per_unit: How many meters each unit in the output should represent
        
        Returns:
            Normalized coordinates where 1 unit = meters_per_unit
        """
        if not coords:
            return []
        
        # Calculate min values for x and y to shift to origin
        min_x = min(point[0] for point in coords)
        min_y = min(point[1] for point in coords)
        
        # Shift to origin, scale down, and adjust to match meters_per_unit
        normalized = []
        for point in coords:
            x = (point[0] - min_x) / scale_factor / meters_per_unit
            y = (point[1] - min_y) / scale_factor / meters_per_unit
            normalized.append((x, y))
        
        return normalized

class BaseSolverModel:
    """Base class for light placement solvers."""
    def __init__(self, room_vertices):
        self.room = Polygon(room_vertices)
        self.room_area = self.room.area
        
        # Create custom colormap from green to blue with more intermediate colors
        colors = [
            (0, 0, 1, 0.1),  # blue (lowest)
            (0, 0.5, 0.5, 0.4),  # blue-green (low)
            (0, 0.8, 0.2, 0.7),  # mostly green (medium)
            (0, 1, 0, 1),  # pure green (high)
        ]
        self.gradient_cmap = LinearSegmentedColormap.from_list("custom", colors)
    
    def get_coverage_value(self, light_center, point):
        """Calculate gradient coverage value based on distance from light center."""
        dx = light_center[0] - point[0]
        dy = light_center[1] - point[1]
        dz = -1
        distance = np.sqrt(dx**2 + dy**2 + dz**2)
        

        cos_theta = abs(dz) / distance

        return (1 * cos_theta) / distance**2

class GridPlacementModel(BaseSolverModel):
    """Model for grid-based light placement."""
    def __init__(self, room_vertices, wall_distance=1.2, 
                 horizontal_spacing=2.5, vertical_spacing=2.5):
        super().__init__(room_vertices)
        self.wall_distance = wall_distance
        self.horizontal_spacing = horizontal_spacing
        self.vertical_spacing = vertical_spacing
    
    def generate_grid_placement(self):
        """Generate light placements in a grid pattern with specified spacing from walls."""
        bounds = self.room.bounds
        x_min, y_min, x_max, y_max = bounds
        
        # Calculate starting positions (wall_distance away from boundaries)
        x_start = x_min + self.wall_distance
        y_start = y_min + self.wall_distance
        
        # Calculate grid points
        x_positions = np.arange(x_start, x_max, self.horizontal_spacing)
        y_positions = np.arange(y_start, y_max, self.vertical_spacing)
        
        # Generate all possible combinations of x and y
        light_candidates = []
        for x in x_positions:
            for y in y_positions:
                point = Point(x, y)
                if self.room.contains(point):
                    light_candidates.append((x, y))
        
        return light_candidates
    
    def calculate_coverage(self, lights):
        """Calculate the coverage for a set of lights using a fine grid."""
        # Generate a fine grid of points to evaluate coverage
        bounds = self.room.bounds
        x_min, y_min, x_max, y_max = bounds
        step = min(self.horizontal_spacing, self.vertical_spacing) / 10  # Fine grid
        
        x_coords = np.arange(x_min, x_max + step, step)
        y_coords = np.arange(y_min, y_max + step, step)
        
        cell_coverage = {}
        for x in x_coords:
            for y in y_coords:
                point = Point(x, y)
                if self.room.contains(point):
                    # Calculate total coverage at this point
                    total_coverage = 0
                    for light_x, light_y in lights:
                        coverage = self.get_coverage_value((light_x, light_y), (x, y))
                        total_coverage += coverage
                    
                    cell_coverage[(x, y)] = total_coverage
        
        return cell_coverage
    
    def solve(self):
        """Generate a grid-based placement of lights."""
        lights = self.generate_grid_placement()
        cell_coverage = self.calculate_coverage(lights)
        
        return lights, cell_coverage

class OptimizationModel(BaseSolverModel):
    """Model for optimized light placement."""
    def __init__(self, room_vertices, grid_size=1.0, area_cell_size=0.1):
        super().__init__(room_vertices)
        self.grid_size = grid_size
        self.area_cell_size = area_cell_size

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

    def get_cells_in_grid_cell(self, grid_x, grid_y, area_cells):
        """Get all small area cells that fall within a given grid cell."""
        min_x = grid_x - self.grid_size / 2
        max_x = grid_x + self.grid_size / 2
        min_y = grid_y - self.grid_size / 2
        max_y = grid_y + self.grid_size / 2

        return [
            (x, y) for x, y in area_cells if min_x <= x < max_x and min_y <= y < max_y
        ]

    def solve(self, min_light_level=0.2, min_light_spacing=1, status_callback=None):
        """
        Solve the optimization problem to minimize number of lights while maintaining
        minimum average coverage in each grid cell, with adjustable spacing between lights.

        Args:
            min_light_level (float): Minimum average light level required in each grid cell (default: 0.2)
            min_light_spacing (int): Minimum number of grid cells between lights (default: 1)
            status_callback (callable): Optional callback function to report status
        """
        grid_points = self.generate_grid_points()
        area_cells = self.generate_area_cells()

        prob = pulp.LpProblem("MinimumlightCoverage", pulp.LpMinimize)

        # Binary variables for light placement
        light_vars = pulp.LpVariable.dicts(
            "light", ((x, y) for x, y in grid_points), cat="Binary"
        )

        # Continuous variables for cell light levels
        cell_vars = pulp.LpVariable.dicts(
            "cell", ((x, y) for x, y in area_cells), lowBound=0
        )

        # Objective: Minimize number of lights
        prob += pulp.lpSum(light_vars[x, y] for x, y in grid_points)

        # Set up constraints for small cell values based on light coverage
        for cell_x, cell_y in area_cells:
            covering_lights = []
            for grid_x, grid_y in grid_points:
                coverage_value = self.get_coverage_value(
                    (grid_x, grid_y), (cell_x, cell_y)
                )
                if coverage_value > 0:
                    covering_lights.append(
                        coverage_value * light_vars[grid_x, grid_y]
                    )

            prob += cell_vars[cell_x, cell_y] == pulp.lpSum(covering_lights)

        # Add minimum average constraint for each grid cell using the parameter
        for grid_x, grid_y in grid_points:
            grid_cells = self.get_cells_in_grid_cell(grid_x, grid_y, area_cells)
            if grid_cells:  # Only add constraint if grid cell has area cells
                prob += pulp.lpSum(
                    cell_vars[x, y] for x, y in grid_cells
                ) >= min_light_level * len(grid_cells)
        
        # Add constraint to prevent lights from being too close to each other
        if min_light_spacing > 0:
            for x1, y1 in grid_points:
                # Calculate neighbors based on spacing parameter
                neighbors = []
                for x2, y2 in grid_points:
                    if (x1, y1) != (x2, y2):  # Don't compare with itself
                        dx = abs(x1 - x2)
                        dy = abs(y1 - y2)
                        # Calculate distance in grid cells
                        distance_in_grid_cells = max(
                            dx / self.grid_size, 
                            dy / self.grid_size
                        )
                        # If the distance is less than the minimum spacing
                        if distance_in_grid_cells <= min_light_spacing:
                            neighbors.append((x2, y2))
                
                # For each close neighbor, add constraint
                for x2, y2 in neighbors:
                    # Add constraint: at most one of these two cells can have a light
                    prob += light_vars[x1, y1] + light_vars[x2, y2] <= 1

        # Display a status message if callback provided
        if status_callback:
            status_callback("info", "Solving optimization problem...")
        
        status = prob.solve()
        status_message = f"Solver status: {pulp.LpStatus[status]}"
        
        if status != 1:  # If not optimal
            status_message += f"\nWarning: Solver did not find an optimal solution. Try adjusting parameters."
            if status_callback:
                status_callback("warning", status_message)
        else:
            if status_callback:
                status_callback("success", status_message)

        # Extract results
        selected_lights = []
        for x, y in grid_points:
            if light_vars[x, y].value() > 0.5:
                selected_lights.append((x, y))

        # Calculate light levels for visualization and verification
        cell_coverage = {}
        grid_averages = {}
        for grid_x, grid_y in grid_points:
            grid_cells = self.get_cells_in_grid_cell(grid_x, grid_y, area_cells)
            grid_total = 0

            for cell_x, cell_y in grid_cells:
                cell_total = 0
                for light_x, light_y in selected_lights:
                    coverage = self.get_coverage_value(
                        (light_x, light_y), (cell_x, cell_y)
                    )
                    cell_total += coverage
                cell_coverage[(cell_x, cell_y)] = cell_total
                grid_total += cell_total

            if grid_cells:
                grid_avg = grid_total / len(grid_cells)
                grid_averages[(grid_x, grid_y)] = grid_avg

        return selected_lights, cell_coverage, grid_averages
