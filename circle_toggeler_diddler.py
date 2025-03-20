import pulp
import numpy as np
from shapely.geometry import Polygon, Point
import plotly.graph_objects as go
from shapely.ops import unary_union
from math import log
import plotly.express as px


class GradientCircleCoverageSolver:
    def __init__(
        self, room_vertices, grid_size=1.0, circle_radius=1.0, area_cell_size=0.1
    ):
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
            return 1 / (1 + 3 * normalized_distance)

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

    def visualize_plotly(self, circles, cell_coverage):
        """Visualize the solution with Plotly and toggle functionality."""
        # Room boundary coordinates
        # Convert Shapely array.array to list for Plotly compatibility
        room_x, room_y = list(self.room.exterior.xy[0]), list(self.room.exterior.xy[1])
        
        # Create a base figure
        fig = go.Figure()
        
        # Add room boundary
        fig.add_trace(go.Scatter(
            x=room_x, y=room_y,
            mode='lines',
            line=dict(color='black', width=2),
            name='Room Boundary',
            hoverinfo='skip'
        ))
        
        # Create gradient coverage traces (initially visible)
        gradient_traces = []
        max_coverage = max(cell_coverage.values()) if cell_coverage else 1.0
        if max_coverage == 0:
            max_coverage = 1.0
            
        # Group cells by their coverage values for more efficient plotting
        coverage_groups = {}
        for (x, y), coverage in cell_coverage.items():
            normalized_coverage = min(1.0, coverage / max_coverage)
            # Round to 2 decimal places for grouping
            rounded_coverage = round(normalized_coverage, 2)
            if rounded_coverage not in coverage_groups:
                coverage_groups[rounded_coverage] = []
            coverage_groups[rounded_coverage].append((x, y))
        
        # Create a better colorscale from blue to green that mimics matplotlib's quality
        colorscale = [
            [0, 'rgba(65, 105, 225, 0.2)'],      # royal blue (lowest)
            [0.2, 'rgba(30, 144, 255, 0.3)'],    # dodger blue (low)
            [0.4, 'rgba(0, 191, 191, 0.5)'],     # turquoise (low-medium)
            [0.6, 'rgba(46, 204, 113, 0.7)'],    # emerald green (medium)
            [0.8, 'rgba(39, 174, 96, 0.85)'],    # medium green (medium-high)
            [1, 'rgba(0, 255, 0, 1)']            # bright green (highest)
        ]
        
        # Add gradient cells by coverage group
        for coverage_value, cells in coverage_groups.items():
            if not cells:
                continue
                
            x_cells = []
            y_cells = []
            for x, y in cells:
                # Create the 4 corners of each cell
                half_cell = self.area_cell_size / 2
                x_cells.extend([x-half_cell, x+half_cell, x+half_cell, x-half_cell, None])
                y_cells.extend([y-half_cell, y-half_cell, y+half_cell, y+half_cell, None])
            
            # Get color based on coverage
            color_idx = min(int(coverage_value * len(colorscale)), len(colorscale) - 1)
            color = colorscale[color_idx][1]
            
            # Add trace for this coverage group
            fig.add_trace(go.Scatter(
                x=x_cells, y=y_cells,
                fill='toself',
                fillcolor=color,
                line=dict(color='rgba(0,0,0,0)'),
                mode='lines',
                name=f'Coverage: {coverage_value:.2f}',
                legendgroup='gradient',
                showlegend=False,
                hoverinfo='text',
                text=[f'Coverage: {coverage_value:.2f}'] * (len(x_cells) // 5),
                visible=True
            ))
        
        # Create grid cells for reference
        grid_points = self.generate_grid_points()
        grid_x = []
        grid_y = []
        
        for x, y in grid_points:
            half_grid = self.grid_size / 2
            grid_x.extend([x-half_grid, x+half_grid, x+half_grid, x-half_grid, x-half_grid, None])
            grid_y.extend([y-half_grid, y-half_grid, y+half_grid, y+half_grid, y-half_grid, None])
        
        fig.add_trace(go.Scatter(
            x=grid_x, y=grid_y,
            mode='lines',
            line=dict(color='gray', width=1, dash='dash'),
            name='Grid Cells',
            hoverinfo='skip'
        ))
        
        # Add black squares for circle centers (initially hidden)
        for x, y in circles:
            square_size = self.grid_size / 2
            # Create square coordinates for each circle center
            square_x = [x-square_size, x+square_size, x+square_size, x-square_size, x-square_size]
            square_y = [y-square_size, y-square_size, y+square_size, y+square_size, y-square_size]
            
            fig.add_trace(go.Scatter(
                x=square_x, y=square_y,
                fill='toself',
                fillcolor='black',
                line=dict(color='black', width=1),
                mode='lines',
                name='Light Positions',
                legendgroup='positions',
                showlegend=(x == circles[0][0] and y == circles[0][1]),  # Show only one in legend
                hoverinfo='text',
                text=f'Light at ({x:.1f}, {y:.1f})',
                visible=False  # Initially hidden, will be toggled
            ))
        
        # Add circles with nicer red outline
        for x, y in circles:
            # Generate points for a circle
            theta = np.linspace(0, 2*np.pi, 100)
            circle_x = x + self.circle_radius * np.cos(theta)
            circle_y = y + self.circle_radius * np.sin(theta)
            
            fig.add_trace(go.Scatter(
                x=circle_x, y=circle_y,
                mode='lines',
                line=dict(color='rgba(255, 0, 0, 0.7)', width=2),
                name=f'Circle at ({x:.1f}, {y:.1f})',
                legendgroup='circles',
                showlegend=False,
                hoverinfo='text',
                text=f'Circle at ({x:.1f}, {y:.1f})'
            ))
        
        # Create more reliable toggle mechanism
        # First count how many gradient traces we have
        gradient_trace_count = 0
        for _ in coverage_groups.keys():
            gradient_trace_count += 1
        
        # Calculate which traces should be visible in each mode
        all_traces_visible = [True] * len(fig.data)
        
        # For simple view - hide all gradient traces, show black squares 
        simple_view_visibility = []
        black_squares_index = None
        
        # Map which traces are gradient traces vs other elements
        for i, trace in enumerate(fig.data):
            if i == 0:  # First trace is room boundary
                simple_view_visibility.append(True)
            elif "Coverage: " in trace.name:
                simple_view_visibility.append(False)  # Hide gradient
            elif "Light Positions" in trace.name:
                simple_view_visibility.append(True)  # Show black squares
                black_squares_index = i
            else:
                simple_view_visibility.append(True)  # Show everything else
        
        # Make sure black squares are visible in simple view
        if black_squares_index is not None:
            fig.data[black_squares_index].visible = True
        
        # Add custom buttons for toggling between gradient and simple view
        updatemenus = [
            dict(
                type="buttons",
                direction="right",
                active=0,
                buttons=[
                    dict(
                        label="Show Gradient",
                        method="update",
                        args=[
                            {"visible": all_traces_visible},
                            {"title": "Gradient Coverage Visualization"}
                        ]
                    ),
                    dict(
                        label="Show Positions Only",
                        method="update",
                        args=[
                            {"visible": simple_view_visibility},
                            {"title": "Circle Placement Visualization"}
                        ]
                    ),
                ],
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.15,
                yanchor="top",
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor="rgba(0, 0, 0, 0.5)",
                font=dict(size=12)
            )
        ]
        
        # Add metrics annotation
        metrics_text = (
            f"Room Area: {self.room_area:.2f} sq units<br>"
            f"Number of Circles: {len(circles)}<br>"
            f"Circle Radius: {self.circle_radius}"
        )
        
        fig.add_annotation(
            text=metrics_text,
            align="left",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.01, y=0.99,
            bgcolor="white",
            bordercolor="black",
            borderwidth=1
        )
        
        # Set axis properties and layout
        bounds = self.room.bounds
        fig.update_layout(
            title={
                'text': "Gradient Coverage Optimization Results",
                'font': {'size': 24, 'color': 'black'}
            },
            updatemenus=updatemenus,
            xaxis=dict(
                range=[bounds[0] - 1, bounds[2] + 1],
                gridcolor='lightgray',
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor='gray',
                title={'text': '', 'font': {'size': 16}}
            ),
            yaxis=dict(
                range=[bounds[1] - 1, bounds[3] + 1],
                gridcolor='lightgray',
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor='gray',
                scaleanchor="x",
                scaleratio=1,
                title={'text': '', 'font': {'size': 16}}
            ),
            legend=dict(
                x=0.01,
                y=0.01,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="rgba(0, 0, 0, 0.5)",
                borderwidth=1,
                font=dict(size=12)
            ),
            margin=dict(l=20, r=20, t=80, b=20),
            plot_bgcolor='rgba(245, 245, 245, 0.8)',
            paper_bgcolor='white',
            height=800,  # Larger figure for better visibility
            width=1000
        )
        
        # Add a better colorbar
        fig.update_layout(
            coloraxis=dict(
                colorscale=colorscale,
                colorbar=dict(
                    title="Coverage Intensity",
                    titleside="right",
                    titlefont=dict(size=14),
                    ticks="outside",
                    tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                    ticktext=["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"],
                    tickfont=dict(size=12),
                    x=1.02,
                    len=0.8,
                    thickness=20
                )
            )
        )
        
        # Add a custom update function to fix toggle behavior
        fig.update_traces(
            selector=dict(name='Light Positions'),
            visible=False  # Initially hidden, toggled by buttons
        )
        
        return fig


if __name__ == "__main__":
    # Simple L-shaped room
    room_vertices = [(0, 0), (10, 0), (10, 5), (5, 5), (5, 10), (0, 10)]
    # room_vertices = [(0, 0), (0, 10), (10, 10), (10, 0)]

    solver = GradientCircleCoverageSolver(
        room_vertices, grid_size=1, circle_radius=4, area_cell_size=0.1
    )
    
    MIN_LUX = 300
    LAMP_LUX = 318
    MIN_AVG_LUX = MIN_LUX / LAMP_LUX

    # Now you can specify different minimum light levels
    circles, cell_coverage = solver.solve(min_light_level=MIN_AVG_LUX)

    print(f"\nOptimization Results:")
    print(f"Number of circles: {len(circles)}")
    print(f"Room area: {solver.room_area:.2f} square units")

    # Create and show the interactive Plotly figure
    fig = solver.visualize_plotly(circles, cell_coverage)
    fig.show()