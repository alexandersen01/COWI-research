import matplotlib.pyplot as plt
import numpy as np

class VisualizationView:
    """Base class for visualization views."""
    def create_empty_room_visualization(self, room_polygon):
        """Create an empty room visualization."""
        fig, ax = plt.subplots(figsize=(10, 8))
        room_x, room_y = room_polygon.exterior.xy
        ax.plot(room_x, room_y, "k-", linewidth=2)
        ax.set_aspect('equal')
        plt.grid(True)
        plt.title("Room Layout")
        
        # Set limits with padding
        bounds = room_polygon.bounds
        plt.xlim(bounds[0] - 1, bounds[2] + 1)
        plt.ylim(bounds[1] - 1, bounds[3] + 1)
        
        return fig

class GridPlacementView(VisualizationView):
    """Visualization view for grid placement model."""
    def visualize(self, model, lights, cell_coverage):
        """Visualize the solution with gradient coverage."""
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot room boundary
        room_x, room_y = model.room.exterior.xy
        ax.plot(room_x, room_y, "k-", label="Room boundary")

        # Plot gradient coverage
        max_coverage = max(cell_coverage.values()) if cell_coverage else 1.0
        if max_coverage == 0:
            max_coverage = 1.0

        # Determine cell size for visualization
        bounds = model.room.bounds
        x_min, y_min, x_max, y_max = bounds
        cell_size = min(model.horizontal_spacing, model.vertical_spacing) / 10

        for (x, y), coverage in cell_coverage.items():
            normalized_coverage = coverage / max_coverage
            rect = plt.Rectangle(
                (x - cell_size / 2, y - cell_size / 2),
                cell_size,
                cell_size,
                color=model.gradient_cmap(normalized_coverage),
                alpha=0.8,
            )
            ax.add_patch(rect)

        # Plot grid lines
        x_positions = np.arange(x_min + model.wall_distance, x_max, model.horizontal_spacing)
        y_positions = np.arange(y_min + model.wall_distance, y_max, model.vertical_spacing)

        for x in x_positions:
            plt.axvline(x, color='gray', linestyle='--', alpha=0.3)
        for y in y_positions:
            plt.axhline(y, color='gray', linestyle='--', alpha=0.3)

        # Plot lights
        for x, y in lights:
            light = plt.Circle((x, y), 0.1, fill=True, color="red")
            ax.add_patch(light)

        # Add metrics text
        metrics_text = (
            f"Room Area: {model.room_area:.2f} sq units\n"
            f"Number of Lights: {len(lights)}\n"
            f"Wall Distance: {model.wall_distance}\n"
            f"Horizontal Spacing: {model.horizontal_spacing}\n"
            f"Vertical Spacing: {model.vertical_spacing}"
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
        sm = plt.cm.ScalarMappable(cmap=model.gradient_cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label="Coverage Intensity")

        return fig

class OptimizationView(VisualizationView):
    """Visualization view for optimization model."""
    def visualize(self, model, lights, cell_coverage):
        """Visualize the solution with gradient coverage."""
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot room boundary
        room_x, room_y = model.room.exterior.xy
        ax.plot(room_x, room_y, "k-", label="Room boundary")

        # Plot gradient coverage
        max_coverage = max(cell_coverage.values()) if cell_coverage else 1.0
        if max_coverage == 0:
            max_coverage = 1.0

        for (x, y), coverage in cell_coverage.items():
            normalized_coverage = coverage / max_coverage
            rect = plt.Rectangle(
                (x - model.area_cell_size / 2, y - model.area_cell_size / 2),
                model.area_cell_size,
                model.area_cell_size,
                color=model.gradient_cmap(normalized_coverage),
                alpha=0.8,
            )
            ax.add_patch(rect)

        # Plot grid cells for reference
        grid_points = model.generate_grid_points()
        for x, y in grid_points:
            rect = plt.Rectangle(
                (x - model.grid_size / 2, y - model.grid_size / 2),
                model.grid_size,
                model.grid_size,
                fill=False,
                color="gray",
                linestyle="--",
                alpha=0.5,
            )
            ax.add_patch(rect)

        # Plot lights
        for x, y in lights:
            light = plt.Circle((x, y), 0.1, fill=True, color="red")
            ax.add_patch(light)

        # Add metrics text
        metrics_text = (
            f"Room Area: {model.room_area:.2f} sq units\n"
            f"Number of Lights: {len(lights)}\n"
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
        bounds = model.room.bounds
        plt.xlim(bounds[0] - 1, bounds[2] + 1)
        plt.ylim(bounds[1] - 1, bounds[3] + 1)

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=model.gradient_cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label="Coverage Intensity")

        return fig
