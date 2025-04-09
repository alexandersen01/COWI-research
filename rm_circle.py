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



# === Optimization Model ===
class GradientCircleCoverageSolver:
    def __init__(self, room_vertices, grid_size=1.0, area_cell_size=0.1, lamp_lumen=1000):
        self.room = Polygon(room_vertices)
        self.grid_size = grid_size
        self.area_cell_size = area_cell_size
        self.room_area = self.room.area
        self.lamp_lumen = lamp_lumen

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
        """Calculate illuminance based on lamp lumens and distance from light source.
        Uses the inverse square law for light propagation."""
        dx = circle_center[0] - cell_center[0]
        dy = circle_center[1] - cell_center[1]
        r = np.sqrt(dx * dx + dy * dy)
        
        # Avoid division by zero by using a small minimum distance
        #r = max(r, 0.01)
        
        # Calculate illuminance using the inverse square law:
        # E = Φ/(4πr²) where:
        # E is illuminance in lux
        # Φ is luminous flux in lumens
        # r is distance in meters
        illuminance = self.lamp_lumen / (4 * np.pi * r**2)
        
        return min(illuminance, 10000)



    def get_cells_in_grid_cell(self, grid_x, grid_y, area_cells):
        """Get all small area cells that fall within a given grid cell."""
        min_x = grid_x - self.grid_size / 2
        max_x = grid_x + self.grid_size / 2
        min_y = grid_y - self.grid_size / 2
        max_y = grid_y + self.grid_size / 2

        return [
            (x, y) for x, y in area_cells if min_x <= x < max_x and min_y <= y < max_y
        ]

    def solve(self, min_light_level=300, min_circle_spacing=1):
        """
        Solve the optimization problem to minimize number of circles while maintaining
        minimum average coverage in each grid cell, with adjustable spacing between circles.

        Args:
            min_light_level (float): Minimum average light level in lux required in each grid cell (default: 300)
            min_circle_spacing (int): Minimum number of grid cells between circles (default: 1)
        """
        grid_points = self.generate_grid_points()
        area_cells = self.generate_area_cells()

        prob = pulp.LpProblem("MinimumCircleCoverage", pulp.LpMinimize)

        # Binary variables for circle placement
        circle_vars = pulp.LpVariable.dicts(
            "circle", ((x, y) for x, y in grid_points), cat="Binary"
        )

        # Continuous variables for cell light levels (in lux)
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
                # Only include coverage values that provide meaningful contribution
                if coverage_value > 1.0:  # Threshold to consider for optimization in lux
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
                            dx / self.grid_size, dy / self.grid_size
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
        """Visualize the solution with actual lux values."""
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot room boundary
        room_x, room_y = self.room.exterior.xy
        ax.plot(room_x, room_y, "k-", label="Room boundary")

        # Determine lux range for visualization
        # Set scale to match typical indoor illuminance values: 0-1000 lux
        min_lux = 0
        max_lux = 1000  # Cap for visualization
        
        # Plot illuminance levels with actual lux values
        for (x, y), lux in cell_coverage.items():
            # Clip lux to range for visualization
            norm_lux = lux / max_lux
            rect = plt.Rectangle(
                (x - self.area_cell_size / 2, y - self.area_cell_size / 2),
                self.area_cell_size,
                self.area_cell_size,
                color=self.gradient_cmap(norm_lux),
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

        # Plot light sources
        for x, y in circles:
            # Draw light source indicators
            center = plt.Circle((x, y), 0.2, fill=True, color="white", edgecolor="black", linewidth=1.5)
            ax.add_patch(center)

            # Add light rays
            for angle in range(0, 360, 30):
                angle_rad = np.radians(angle)
                dx = 0.5 * np.cos(angle_rad)
                dy = 0.5 * np.sin(angle_rad)
                ax.plot([x, x + dx], [y, y + dy], 'yellow', linewidth=1.2)

        # Add metrics text
        max_actual_lux = max(cell_coverage.values()) if cell_coverage else 0
        metrics_text = (
            f"Room Area: {self.room_area:.2f} sq units\n"
            f"Number of Light Sources: {len(circles)}\n"
            f"Lamp Output: {self.lamp_lumen} lumens\n"
            f"Maximum Illuminance: {max_actual_lux:.1f} lux"
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
        plt.title("Illuminance Optimization Results")

        # Set limits with padding
        bounds = self.room.bounds
        plt.xlim(bounds[0] - 1, bounds[2] + 1)
        plt.ylim(bounds[1] - 1, bounds[3] + 1)

        # Add colorbar with actual lux values
        norm = plt.Normalize(min_lux, max_lux)
        sm = plt.cm.ScalarMappable(cmap=self.gradient_cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label="Illuminance (lux)")
        
        # Add tick labels to colorbar with lux values
        # cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        #cbar.set_ticklabels(['0', '200', '400', '600', '800', '1000+'])

        return fig


# ==== Streamlit Application ====

# Sidebar for room shape selection and parameters
st.sidebar.header("Configuration")

# Choose solution approach
solution_approach = st.sidebar.radio(
    "Solution Approach", ["Optimization Model"]
)

# Room shape selection
room_shape = st.sidebar.selectbox(
    "Room Shape", ["L-shaped Room", "Rectangle", "T-shaped Room", "Custom"]
)

# Room configuration based on selection
if room_shape == "L-shaped Room":
    room_vertices = [(0, 0), (10, 0), (10, 5), (5, 5), (5, 10), (0, 10)]
elif room_shape == "Rectangle":
    width = st.sidebar.slider("Room Width", 5, 20, 10)
    height = st.sidebar.slider("Room Height", 5, 20, 10)
    room_vertices = [(0, 0), (width, 0), (width, height), (0, height)]
elif room_shape == "T-shaped Room":
    room_vertices = [
        (0, 0),
        (15, 0),
        (15, 5),
        (10, 5),
        (10, 15),
        (5, 15),
        (5, 5),
        (0, 5),
    ]
else:  # Custom
    st.sidebar.text("Enter vertices as x,y pairs (one per line):")
    custom_vertices = st.sidebar.text_area(
        "Format: x,y", "0,0\n10,0\n10,5\n5,5\n5,10\n0,10"
    )
    try:
        room_vertices = [
            tuple(map(float, line.split(",")))
            for line in custom_vertices.strip().split("\n")
        ]
    except:
        st.sidebar.error(
            "Invalid vertex format. Please use 'x,y' format, one pair per line."
        )
        room_vertices = [
            (0, 0),
            (10, 0),
            (10, 5),
            (5, 5),
            (5, 10),
            (0, 10),
        ]  # Default to L-shape

# Add min_circle_spacing parameter to sidebar for Optimization Model
st.sidebar.header("Optimization Parameters")
grid_size = st.sidebar.slider("Grid Size", 0.5, 2.0, 1.0, 0.1)
area_cell_size = st.sidebar.slider("Area Cell Size", 0.1, 0.5, 0.2, 0.05)
min_light_level = st.sidebar.slider("Minimum Light Level (lux)", 300, 500, 100, 10)
lamp_lumen = st.sidebar.slider("Lamp Luminous Flux (lumens)", 500, 1000, 2000, 100)
min_circle_spacing = st.sidebar.slider(
    "Minimum Circle Spacing (grid cells)",
    0,
    3,
    1,
    1,
    help="Minimum number of grid cells between circles. Set to 0 to allow adjacent placement.",
)


solver = GradientCircleCoverageSolver(
    room_vertices, grid_size=grid_size, area_cell_size=area_cell_size, lamp_lumen=lamp_lumen
)

# Main content
col, col2 = st.columns([3, 1])

with col:
    st.subheader("Room and Coverage Visualization")

    # Run optimization when button is clicked
    if st.button("Run Optimization"):
        with st.spinner("Optimizing circle placement..."):
            circles, cell_coverage, grid_averages = solver.solve(
                min_light_level=min_light_level
            )

            # Visualization
            fig = solver.visualize(circles, cell_coverage)
            st.pyplot(fig)

            # Save results in session state for tables
            st.session_state["circles"] = circles
            st.session_state["grid_averages"] = grid_averages
            st.session_state["cell_coverage"] = cell_coverage
            st.session_state["is_optimized"] = True
            st.session_state["is_grid_placed"] = False
    elif not st.session_state.get("is_optimized", False):
        # Show an initial room visualization without circles
        fig, ax = plt.subplots(figsize=(10, 8))
        room_x, room_y = solver.room.exterior.xy
        ax.plot(room_x, room_y, "k-", linewidth=2)
        ax.set_aspect("equal")
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
        circles = st.session_state.get("circles", [])
        cell_coverage = st.session_state.get("cell_coverage", {})
        fig = solver.visualize(circles, cell_coverage)
        st.pyplot(fig)

with col2:
    st.subheader("Results")
    if "circles" in st.session_state:
        st.metric("Number of Circles", len(st.session_state["circles"]))
        st.metric("Room Area", f"{solver.room_area:.2f} sq units")

        # Display circle positions
        st.subheader("Circle Positions")
        circle_df_data = [{"X": x, "Y": y} for x, y in st.session_state["circles"]]
        if circle_df_data:
            st.dataframe(circle_df_data, height=200)

        # Display average coverage
        st.subheader("Grid Cell Coverage")
        if st.session_state["grid_averages"]:
            avg_df_data = [
                {"X": x, "Y": y, "Avg Coverage": f"{avg:.3f}"}
                for (x, y), avg in st.session_state["grid_averages"].items()
            ]
            st.dataframe(avg_df_data, height=200)

            # Coverage statistics
            coverage_values = list(st.session_state["grid_averages"].values())
            min_coverage = min(coverage_values)
            max_coverage = max(coverage_values)
            avg_coverage = sum(coverage_values) / len(coverage_values)

            st.metric("Min Coverage", f"{min_coverage:.3f}")
            st.metric("Max Coverage", f"{max_coverage:.3f}")
            st.metric("Avg Coverage", f"{avg_coverage:.3f}")
    else:
        st.info("Results will appear here after optimization")

# Add explanations based on solution approach
st.markdown("""
## How the Optimization Model Works

This app optimizes the placement of circles to provide gradient coverage in a room:

1. **Problem**: Place the minimum number of circles while ensuring each grid cell has at least the specified minimum light level.
2. **Gradient Coverage**: Light intensity decreases with the square of distance from the light source.
3. **Optimization**: Uses integer linear programming to find the optimal solution.
4. **Spacing Constraint**: Controls the minimum distance between any two circles.

## Parameters Explained

- **Grid Size**: Size of each grid cell for potential light placement
- **Area Cell Size**: Granularity for measuring coverage (smaller = more accurate but slower)
- **Minimum Light Level**: Required average light intensity in each grid cell
- **Minimum Circle Spacing**: Controls how far apart circles must be placed (in grid cells)
""")

# Comparison section when both models have been run
if st.session_state.get("is_optimized", False) and st.session_state.get(
    "is_grid_placed", False
):
    st.markdown("---")
    st.subheader("Model Comparison")

    opt_circles = len(
        st.session_state.get("circles", [])
        if st.session_state.get("is_optimized", False)
        else []
    )
    grid_circles = len(
        st.session_state.get("circles", [])
        if st.session_state.get("is_grid_placed", False)
        else []
    )

    col= st.columns(1)
    with col:
        st.metric("Optimization Model", f"{opt_circles} circles")

    st.markdown("""
    ### Key Differences
    
    - **Optimization Model**: Finds the minimum number of circles needed to meet coverage requirements
    
    Try adjusting parameters of both models to see how they affect coverage and efficiency!
    """)

# Footer
st.markdown("---")
st.caption("Circle Coverage Optimization Tool")
