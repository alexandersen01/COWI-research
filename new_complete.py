import streamlit as st
import pulp
import numpy as np
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt
from shapely.ops import unary_union
from math import log
from matplotlib.colors import LinearSegmentedColormap

# Set up the page
st.set_page_config(page_title="Circle Coverage Optimizer", layout="wide")
st.title("Gradient Circle Coverage Optimization")
st.write("Optimize the placement of circles to provide gradient coverage in a room.")

# === Simple Light Placement Model ===

class GridPlacementSolver:
    def __init__(self, room_vertices, circle_radius=1.0, wall_distance=1.2, 
                 horizontal_spacing=2.5, vertical_spacing=2.5):
        self.room = Polygon(room_vertices)
        self.circle_radius = circle_radius
        self.wall_distance = wall_distance
        self.horizontal_spacing = horizontal_spacing
        self.vertical_spacing = vertical_spacing
        self.room_area = self.room.area
        
        # Create custom colormap from green to blue with more intermediate colors
        colors = [
            (0, 0, 1, 0.1),  # blue (lowest)
            (0, 0.5, 0.5, 0.4),  # blue-green (low)
            (0, 0.8, 0.2, 0.7),  # mostly green (medium)
            (0, 1, 0, 1),
        ]  # pure green (high)
        self.gradient_cmap = LinearSegmentedColormap.from_list("custom", colors)
        
    def generate_grid_placement(self):
        """Generate circle placements in a grid pattern with specified spacing from walls."""
        bounds = self.room.bounds
        x_min, y_min, x_max, y_max = bounds
        
        # Calculate starting positions (wall_distance away from boundaries)
        x_start = x_min + self.wall_distance
        y_start = y_min + self.wall_distance
        
        # Calculate grid points
        x_positions = np.arange(x_start, x_max, self.horizontal_spacing)
        y_positions = np.arange(y_start, y_max, self.vertical_spacing)
        
        # Generate all possible combinations of x and y
        circle_candidates = []
        for x in x_positions:
            for y in y_positions:
                point = Point(x, y)
                if self.room.contains(point):
                    circle_candidates.append((x, y))
        
        return circle_candidates
    
    def get_coverage_value(self, circle_center, point):
        """Calculate gradient coverage value based on distance from circle center."""
        dx = circle_center[0] - point[0]
        dy = circle_center[1] - point[1]
        distance = np.sqrt(dx * dx + dy * dy)
        
        if distance <= self.circle_radius:
            normalized_distance = distance / self.circle_radius
            return 1 / (1 + 2 * normalized_distance)
        
        return 0
    
    def calculate_coverage(self, circles):
        """Calculate the coverage for a set of circles using a fine grid."""
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
                    for circle_x, circle_y in circles:
                        coverage = self.get_coverage_value((circle_x, circle_y), (x, y))
                        total_coverage += coverage
                    
                    cell_coverage[(x, y)] = total_coverage
        
        return cell_coverage
    
    def solve(self):
        """Generate a grid-based placement of circles."""
        circles = self.generate_grid_placement()
        cell_coverage = self.calculate_coverage(circles)
        
        return circles, cell_coverage
    
    def visualize(self, circles, cell_coverage):
        """Visualize the solution with gradient coverage."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot room boundary
        room_x, room_y = self.room.exterior.xy
        ax.plot(room_x, room_y, "k-", label="Room boundary")
        
        # Plot gradient coverage
        max_coverage = max(cell_coverage.values()) if cell_coverage else 1.0
        if max_coverage == 0:
            max_coverage = 1.0
        
        # Determine cell size for visualization
        bounds = self.room.bounds
        x_min, y_min, x_max, y_max = bounds
        cell_size = min(self.horizontal_spacing, self.vertical_spacing) / 10
        
        for (x, y), coverage in cell_coverage.items():
            normalized_coverage = coverage / max_coverage
            rect = plt.Rectangle(
                (x - cell_size / 2, y - cell_size / 2),
                cell_size,
                cell_size,
                color=self.gradient_cmap(normalized_coverage),
                alpha=0.8,
            )
            ax.add_patch(rect)
        
        # Plot grid lines
        x_positions = np.arange(x_min + self.wall_distance, x_max, self.horizontal_spacing)
        y_positions = np.arange(y_min + self.wall_distance, y_max, self.vertical_spacing)
        
        for x in x_positions:
            plt.axvline(x, color='gray', linestyle='--', alpha=0.3)
        for y in y_positions:
            plt.axhline(y, color='gray', linestyle='--', alpha=0.3)
        
        # Plot circles
        for x, y in circles:
            circle = plt.Circle(
                (x, y), self.circle_radius, fill=False, color="red", alpha=0.7, linewidth=2
            )
            ax.add_patch(circle)
            # Add a smaller circle at the center for better visibility
            center = plt.Circle((x, y), 0.1, fill=True, color="red")
            ax.add_patch(center)
        
        # Add metrics text
        metrics_text = (
            f"Room Area: {self.room_area:.2f} sq units\n"
            f"Number of Circles: {len(circles)}\n"
            f"Circle Radius: {self.circle_radius}\n"
            f"Wall Distance: {self.wall_distance}\n"
            f"Horizontal Spacing: {self.horizontal_spacing}\n"
            f"Vertical Spacing: {self.vertical_spacing}"
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
        plt.grid(False)  # Turn off default grid to avoid conflict with our custom grid lines
        plt.title("Grid Placement Results")
        
        # Set limits with padding
        plt.xlim(bounds[0] - 1, bounds[2] + 1)
        plt.ylim(bounds[1] - 1, bounds[3] + 1)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=self.gradient_cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label="Coverage Intensity")
        
        return fig

# === Optimization Model ===

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

    def solve(self, min_light_level=0.2, min_circle_spacing=1):
        """
        Solve the optimization problem to minimize number of circles while maintaining
        minimum average coverage in each grid cell, with adjustable spacing between circles.

        Args:
            min_light_level (float): Minimum average light level required in each grid cell (default: 0.2)
            min_circle_spacing (int): Minimum number of grid cells between circles (default: 1)
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
        
        # Add constraint to prevent circles from being too close to each other
        if min_circle_spacing > 0:
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
                        if distance_in_grid_cells <= min_circle_spacing:
                            neighbors.append((x2, y2))
                
                # For each close neighbor, add constraint
                for x2, y2 in neighbors:
                    # Add constraint: at most one of these two cells can have a circle
                    prob += circle_vars[x1, y1] + circle_vars[x2, y2] <= 1

        # Display a status message in Streamlit
        status_placeholder = st.empty()
        status_placeholder.info("Solving optimization problem...")
        
        status = prob.solve()
        status_message = f"Solver status: {pulp.LpStatus[status]}"
        
        if status != 1:  # If not optimal
            status_message += f"\nWarning: Solver did not find an optimal solution. Try adjusting parameters."
            status_placeholder.warning(status_message)
        else:
            status_placeholder.success(status_message)

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

        return selected_circles, cell_coverage, grid_averages

    def visualize(self, circles, cell_coverage):
        """Visualize the solution with gradient coverage."""
        fig, ax = plt.subplots(figsize=(10, 8))

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
                (x, y), self.circle_radius, fill=False, color="red", alpha=0.7, linewidth=2
            )
            ax.add_patch(circle)
            # Add a smaller circle at the center for better visibility
            center = plt.Circle((x, y), 0.1, fill=True, color="red")
            ax.add_patch(center)

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
        cbar = plt.colorbar(sm, ax=ax, label="Coverage Intensity")

        return fig

# ==== Streamlit Application ====

# Sidebar for room shape selection and parameters
st.sidebar.header("Configuration")

# Choose solution approach
solution_approach = st.sidebar.radio(
    "Solution Approach",
    ["Optimization Model", "Grid Placement Model"]
)

# Room shape selection
room_shape = st.sidebar.selectbox(
    "Room Shape",
    ["L-shaped Room", "Rectangle", "T-shaped Room", "Custom"]
)

# Room configuration based on selection
if room_shape == "L-shaped Room":
    room_vertices = [(0, 0), (10, 0), (10, 5), (5, 5), (5, 10), (0, 10)]
elif room_shape == "Rectangle":
    width = st.sidebar.slider("Room Width", 5, 20, 10)
    height = st.sidebar.slider("Room Height", 5, 20, 10)
    room_vertices = [(0, 0), (width, 0), (width, height), (0, height)]
elif room_shape == "T-shaped Room":
    room_vertices = [(0, 0), (15, 0), (15, 5), (10, 5), (10, 15), (5, 15), (5, 5), (0, 5)]
else:  # Custom
    st.sidebar.text("Enter vertices as x,y pairs (one per line):")
    custom_vertices = st.sidebar.text_area(
        "Format: x,y",
        "0,0\n10,0\n10,5\n5,5\n5,10\n0,10"
    )
    try:
        room_vertices = [tuple(map(float, line.split(','))) for line in custom_vertices.strip().split('\n')]
    except:
        st.sidebar.error("Invalid vertex format. Please use 'x,y' format, one pair per line.")
        room_vertices = [(0, 0), (10, 0), (10, 5), (5, 5), (5, 10), (0, 10)]  # Default to L-shape

# Add min_circle_spacing parameter to sidebar for Optimization Model
if solution_approach == "Optimization Model":
    st.sidebar.header("Optimization Parameters")
    grid_size = st.sidebar.slider("Grid Size", 0.5, 2.0, 1.0, 0.1)
    circle_radius = st.sidebar.slider("Circle Radius", 0.5, 3.0, 1.5, 0.1)
    area_cell_size = st.sidebar.slider("Area Cell Size", 0.1, 0.5, 0.2, 0.05)
    min_light_level = st.sidebar.slider("Minimum Light Level", 0.1, 1.0, 0.3, 0.05)
    min_circle_spacing = st.sidebar.slider("Minimum Circle Spacing (grid cells)", 0, 3, 1, 1,
                                         help="Minimum number of grid cells between circles. Set to 0 to allow adjacent placement.")
else:  # Grid Placement Model
    st.sidebar.header("Grid Placement Parameters")
    circle_radius = st.sidebar.slider("Circle Radius", 0.5, 3.0, 1.5, 0.1)
    wall_distance = st.sidebar.slider("Distance from Walls", 0.5, 3.0, 1.2, 0.1)
    horizontal_spacing = st.sidebar.slider("Horizontal Spacing", 1.0, 5.0, 2.5, 0.1)
    vertical_spacing = st.sidebar.slider("Vertical Spacing", 1.0, 5.0, 2.5, 0.1)

# Initialize solvers based on selected approach
if solution_approach == "Optimization Model":
    solver = GradientCircleCoverageSolver(
        room_vertices, 
        grid_size=grid_size, 
        circle_radius=circle_radius, 
        area_cell_size=area_cell_size
    )
else:  # Grid Placement Model
    solver = GridPlacementSolver(
        room_vertices,
        circle_radius=circle_radius,
        wall_distance=wall_distance,
        horizontal_spacing=horizontal_spacing,
        vertical_spacing=vertical_spacing
    )

# Main content
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Room and Coverage Visualization")
    
    if solution_approach == "Optimization Model":
        # Run optimization when button is clicked
        if st.button("Run Optimization"):
            with st.spinner("Optimizing circle placement..."):
                circles, cell_coverage, grid_averages = solver.solve(min_light_level=min_light_level)
                
                # Visualization
                fig = solver.visualize(circles, cell_coverage)
                st.pyplot(fig)
                
                # Save results in session state for tables
                st.session_state['circles'] = circles
                st.session_state['grid_averages'] = grid_averages
                st.session_state['cell_coverage'] = cell_coverage
                st.session_state['is_optimized'] = True
                st.session_state['is_grid_placed'] = False
        elif not st.session_state.get('is_optimized', False):
            # Show an initial room visualization without circles
            fig, ax = plt.subplots(figsize=(10, 8))
            room_x, room_y = solver.room.exterior.xy
            ax.plot(room_x, room_y, "k-", linewidth=2)
            ax.set_aspect('equal')
            plt.grid(True)
            plt.title("Room Layout")
            
            # Set limits with padding
            bounds = solver.room.bounds
            plt.xlim(bounds[0] - 1, bounds[2] + 1)
            plt.ylim(bounds[1] - 1, bounds[3] + 1)
            
            st.pyplot(fig)
            st.info("Click 'Run Optimization' to find the optimal circle placement.")
        else:
            # Show previously generated optimization results if they exist
            circles = st.session_state.get('circles', [])
            cell_coverage = st.session_state.get('cell_coverage', {})
            fig = solver.visualize(circles, cell_coverage)
            st.pyplot(fig)
    
    else:  # Grid Placement Model
        # Run grid placement when button is clicked
        if st.button("Generate Grid Placement"):
            with st.spinner("Generating grid placement..."):
                circles, cell_coverage = solver.solve()
                
                # Visualization
                fig = solver.visualize(circles, cell_coverage)
                st.pyplot(fig)
                
                # Save results in session state for tables
                st.session_state['circles'] = circles
                st.session_state['cell_coverage'] = cell_coverage
                st.session_state['is_grid_placed'] = True
                st.session_state['is_optimized'] = False
                
                # Calculate average coverage per region (for display purposes)
                bounds = solver.room.bounds
                x_min, y_min, x_max, y_max = bounds
                grid_averages = {}
                
                # Create grid cells for averaging
                cell_size = min(horizontal_spacing, vertical_spacing)
                for x in np.arange(x_min, x_max, cell_size):
                    for y in np.arange(y_min, y_max, cell_size):
                        cell_points = [(px, py) for (px, py), cov in cell_coverage.items() 
                                     if x <= px < x + cell_size and y <= py < y + cell_size]
                        if cell_points:
                            cell_values = [cell_coverage[(px, py)] for px, py in cell_points]
                            grid_averages[(x + cell_size/2, y + cell_size/2)] = sum(cell_values) / len(cell_values)
                
                st.session_state['grid_averages'] = grid_averages
        
        elif not st.session_state.get('is_grid_placed', False):
            # Show an initial room visualization without circles
            fig, ax = plt.subplots(figsize=(10, 8))
            room_x, room_y = solver.room.exterior.xy
            ax.plot(room_x, room_y, "k-", linewidth=2)
            ax.set_aspect('equal')
            plt.grid(True)
            plt.title("Room Layout")
            
            # Set limits with padding
            bounds = solver.room.bounds
            plt.xlim(bounds[0] - 1, bounds[2] + 1)
            plt.ylim(bounds[1] - 1, bounds[3] + 1)
            
            st.pyplot(fig)
            st.info("Click 'Generate Grid Placement' to create a grid-based light placement.")
        else:
            # Show previously generated grid placement results if they exist
            circles = st.session_state.get('circles', [])
            cell_coverage = st.session_state.get('cell_coverage', {})
            fig = solver.visualize(circles, cell_coverage)
            st.pyplot(fig)

with col2:
    st.subheader("Results")
    if 'circles' in st.session_state:
        st.metric("Number of Circles", len(st.session_state['circles']))
        st.metric("Room Area", f"{solver.room_area:.2f} sq units")
        
        # Display circle positions
        st.subheader("Circle Positions")
        circle_df_data = [{"X": x, "Y": y} for x, y in st.session_state['circles']]
        if circle_df_data:
            st.dataframe(circle_df_data, height=200)
        
        # Display average coverage
        st.subheader("Grid Cell Coverage")
        if st.session_state['grid_averages']:
            avg_df_data = [{"X": x, "Y": y, "Avg Coverage": f"{avg:.3f}"} 
                          for (x, y), avg in st.session_state['grid_averages'].items()]
            st.dataframe(avg_df_data, height=200)
            
            # Coverage statistics
            coverage_values = list(st.session_state['grid_averages'].values())
            min_coverage = min(coverage_values)
            max_coverage = max(coverage_values)
            avg_coverage = sum(coverage_values) / len(coverage_values)
            
            st.metric("Min Coverage", f"{min_coverage:.3f}")
            st.metric("Max Coverage", f"{max_coverage:.3f}")
            st.metric("Avg Coverage", f"{avg_coverage:.3f}")
    else:
        st.info("Results will appear here after optimization")

# Add explanations based on solution approach
if solution_approach == "Optimization Model":
    st.markdown("""
    ## How the Optimization Model Works

    This app optimizes the placement of circles to provide gradient coverage in a room:

    1. **Problem**: Place the minimum number of circles while ensuring each grid cell has at least the specified minimum light level.
    2. **Gradient Coverage**: Light intensity decreases with distance from the circle center.
    3. **Optimization**: Uses integer linear programming to find the optimal solution.
    4. **Spacing Constraint**: Controls the minimum distance between any two circles.

    ## Parameters Explained

    - **Circle Radius**: Radius of each coverage circle
    - **Area Cell Size**: Granularity for measuring coverage (smaller = more accurate but slower)
    - **Minimum Light Level**: Required average light intensity in each grid cell
    - **Minimum Circle Spacing**: Controls how far apart circles must be placed (in grid cells)
    """)
else:  # Grid Placement Model
    st.markdown("""
    ## How the Grid Placement Model Works

    This approach places circles in a regular grid pattern:

    1. **Wall Distance**: Circles are placed at a fixed distance from walls
    2. **Regular Spacing**: Circles are placed with fixed horizontal and vertical spacing
    3. **Gradient Coverage**: Light intensity decreases with distance from each circle
    4. **Simple Algorithm**: No optimization - just regular grid placement

    ## Parameters Explained

    - **Circle Radius**: Radius of each coverage circle
    - **Distance from Walls**: How far to place lights from the walls
    - **Horizontal Spacing**: Distance between lights in the horizontal direction
    - **Vertical Spacing**: Distance between lights in the vertical direction
    """)

# Comparison section when both models have been run
if st.session_state.get('is_optimized', False) and st.session_state.get('is_grid_placed', False):
    st.markdown("---")
    st.subheader("Model Comparison")
    
    opt_circles = len(st.session_state.get('circles', []) if st.session_state.get('is_optimized', False) else [])
    grid_circles = len(st.session_state.get('circles', []) if st.session_state.get('is_grid_placed', False) else [])
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Optimization Model", f"{opt_circles} circles")
    with col2:
        st.metric("Grid Placement Model", f"{grid_circles} circles")
    
    st.markdown("""
    ### Key Differences
    
    - **Optimization Model**: Finds the minimum number of circles needed to meet coverage requirements
    - **Grid Placement Model**: Uses a fixed pattern based on spacing parameters
    
    Try adjusting parameters of both models to see how they affect coverage and efficiency!
    """)

# Footer
st.markdown("---")
st.caption("Circle Coverage Optimization Tool")