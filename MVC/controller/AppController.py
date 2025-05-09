import streamlit as st
import json
import numpy as np
from model.LightOptimizationModel import RoomModel, OptimizationModel, GridPlacementModel 
from view.LightOptimizationView import OptimizationView, GridPlacementView

class AppController:
    """Main controller for the application."""
    def __init__(self):
        self.room_model = RoomModel()
        
        # Initialize session state if needed
        if 'is_optimized' not in st.session_state:
            st.session_state['is_optimized'] = False
        if 'is_grid_placed' not in st.session_state:
            st.session_state['is_grid_placed'] = False
        if 'json_loaded' not in st.session_state:
            st.session_state['json_loaded'] = False
        if 'room_data' not in st.session_state:
            st.session_state['room_data'] = None
    
    def setup_sidebar(self):
        """Set up the sidebar configuration."""
        st.sidebar.header("Configuration")
        
        # Choose solution approach
        solution_approach = st.sidebar.radio(
            "Solution Approach",
            ["Optimization Model", "Grid Placement Model"]
        )
        
        # Room source selection
        room_source = st.sidebar.radio(
            "Room Source",
            ["Predefined Shapes", "JSON File"]
        )
        
        return solution_approach, room_source
    
    def load_room_from_json(self):
        """Load room data from JSON file."""
        # Only process the file if we haven't already or a new file is uploaded
        if ('json_loaded' not in st.session_state or not st.session_state.json_loaded):
            if 'uploaded_json' in st.session_state and st.session_state.uploaded_json is not None:
                try:
                    # Read the content of the uploaded file
                    file_content = st.session_state.uploaded_json.getvalue().decode('utf-8')
                    data = json.loads(file_content)
                    process_success, process_message = self.room_model.process_room_data(data)
                
                    if process_success:
                        st.sidebar.success("JSON file uploaded and processed successfully")
                        # Store the processed room data in session state
                        st.session_state['room_data'] = self.room_model.rooms
                        st.session_state.json_loaded = True
                        return True
                    else:
                        st.sidebar.warning(f"Error processing JSON data: {process_message}")
                        return False
                except Exception as e:
                    st.sidebar.error(f"Error loading JSON file: {str(e)}")
                    return False
            else:
                # Try standard file loading as fallback
                success, message, data = self.room_model.load_json_rooms()
                if success and data:
                    process_success, process_message = self.room_model.process_room_data(data)
                    if process_success:
                        st.sidebar.success(message)
                        st.sidebar.success(process_message)
                        # Store the processed room data in session state
                        st.session_state['room_data'] = self.room_model.rooms
                        st.session_state.json_loaded = True
                        return True
                    else:
                        st.sidebar.warning(process_message)
                        return False
                else:
                    st.sidebar.warning(message)
                    return False
        # If the data is already loaded, restore it to the room_model
        elif st.session_state.json_loaded and 'room_data' in st.session_state:
            self.room_model.rooms = st.session_state['room_data']
            return True
    
        return st.session_state.json_loaded
    
    def _try_load_from_paths(self):
        """Try to load JSON from predefined paths."""
        possible_paths = [
            'spatial_elements_boundaries.json',
            './spatial_elements_boundaries.json',
            '../spatial_elements_boundaries.json',
            'spatial_elements_boundaries'  # Without extension
        ]
    
        for path in possible_paths:
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                return True, f"Found and loaded JSON file: {path}", data
            except FileNotFoundError:
                continue
            except json.JSONDecodeError:
                return False, f"Invalid JSON format in file: {path}", None
            except Exception as e:
                return False, f"Error loading JSON file {path}: {str(e)}", None
    
        return False, "No JSON file found or uploaded. Please upload a file.", None
    
    def get_room_vertices(self, room_source):
        """Get room vertices based on selected source."""
        if room_source == "Predefined Shapes":
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
                    "0,0\n10,0\n10,5\n5,10\n0,10"
                )
                try:
                    room_vertices = [tuple(map(float, line.split(','))) for line in custom_vertices.strip().split('\n')]
                except:
                    st.sidebar.error("Invalid vertex format. Please use 'x,y' format, one pair per line.")
                    room_vertices = [(0, 0), (10, 0), (10, 5), (5, 5), (5, 10), (0, 10)]  # Default to L-shape
        else:  # JSON File
            # Room selection from JSON
            room_ids = self.room_model.get_room_ids()
            
            if room_ids:
                selected_room_id = st.sidebar.selectbox(
                    "Select Room ID",
                    room_ids
                )
                
                # Get room coordinates
                raw_coords = self.room_model.get_room_coordinates(selected_room_id)
                
                # Normalize coordinates to a reasonable scale
                room_vertices = self.room_model.normalize_coordinates(raw_coords)
                
                # Display room info
                st.sidebar.info(f"Room ID: {selected_room_id}")
                st.sidebar.info(f"Number of points: {len(raw_coords)}")
            else:
                st.sidebar.warning("No rooms loaded from JSON file.")
                # Default to L-shape as fallback
                room_vertices = [(0, 0), (10, 0), (10, 5), (5, 5), (5, 10), (0, 10)]
        
        return room_vertices
    
    def setup_optimization_params(self):
        """Set up parameters for optimization model."""
        st.sidebar.header("Optimization Parameters")
        circle_radius = st.sidebar.slider("Circle Radius", 0.5, 3.0, 1.5, 0.1)
        area_cell_size = st.sidebar.slider("Area Cell Size", 0.1, 0.5, 0.2, 0.05)
        min_light_level = st.sidebar.slider("Minimum Light Level", 0.1, 1.0, 0.3, 0.05)
        min_circle_spacing = st.sidebar.slider("Minimum Circle Spacing (grid cells)", 0, 3, 1, 1,
        help="Minimum number of grid cells between circles. Set to 0 to allow adjacent placement.")
        grid_size = st.sidebar.slider("Grid Size", 0.5, 2.0, 1.0, 0.1)
        
        return {
            'circle_radius': circle_radius,
            'area_cell_size': area_cell_size,
            'min_light_level': min_light_level,
            'min_circle_spacing': min_circle_spacing,
            'grid_size': grid_size
        }
    
    def setup_grid_placement_params(self):
        """Set up parameters for grid placement model."""
        st.sidebar.header("Grid Placement Parameters")
        circle_radius = st.sidebar.slider("Circle Radius", 0.5, 3.0, 1.5, 0.1)
        wall_distance = st.sidebar.slider("Distance from Walls", 0.5, 3.0, 1.2, 0.1)
        horizontal_spacing = st.sidebar.slider("Horizontal Spacing", 1.0, 5.0, 2.5, 0.1)
        vertical_spacing = st.sidebar.slider("Vertical Spacing", 1.0, 5.0, 2.5, 0.1)
        
        return {
            'circle_radius': circle_radius,
            'wall_distance': wall_distance,
            'horizontal_spacing': horizontal_spacing,
            'vertical_spacing': vertical_spacing
        }
    
    def create_model(self, solution_approach, room_vertices, params):
        """Create appropriate model based on solution approach."""
        if solution_approach == "Optimization Model":
            return OptimizationModel(
                room_vertices,
                grid_size=params['grid_size'],
                circle_radius=params['circle_radius'],
                area_cell_size=params['area_cell_size']
            ), OptimizationView()
        else:  # Grid Placement Model
            return GridPlacementModel(
                room_vertices,
                circle_radius=params['circle_radius'],
                wall_distance=params['wall_distance'],
                horizontal_spacing=params['horizontal_spacing'],
                vertical_spacing=params['vertical_spacing']
            ), GridPlacementView()
    
    def handle_optimization_model(self, model, view, params):
        """Handle the optimization model workflow."""
        # Run optimization when button is clicked
        if st.button("Run Optimization"):
            with st.spinner("Optimizing circle placement..."):
                status_placeholder = st.empty()
                
                def status_callback(status_type, message):
                    if status_type == "info":
                        status_placeholder.info(message)
                    elif status_type == "warning":
                        status_placeholder.warning(message)
                    elif status_type == "success":
                        status_placeholder.success(message)
                
                circles, cell_coverage, grid_averages = model.solve(
                    min_light_level=params['min_light_level'],
                    min_circle_spacing=params['min_circle_spacing'],
                    status_callback=status_callback
                )
                
                # Visualization
                fig = view.visualize(model, circles, cell_coverage)
                st.pyplot(fig)
                
                # Save results in session state for tables
                st.session_state['circles'] = circles
                st.session_state['grid_averages'] = grid_averages
                st.session_state['cell_coverage'] = cell_coverage
                st.session_state['is_optimized'] = True
                st.session_state['is_grid_placed'] = False
        elif not st.session_state.get('is_optimized', False):
            # Show an initial room visualization without circles
            fig = view.create_empty_room_visualization(model.room)
            st.pyplot(fig)
            st.info("Click 'Run Optimization' to find the optimal circle placement.")
        else:
            # Show previously generated optimization results if they exist
            circles = st.session_state.get('circles', [])
            cell_coverage = st.session_state.get('cell_coverage', {})
            fig = view.visualize(model, circles, cell_coverage)
            st.pyplot(fig)
    
    def handle_grid_placement_model(self, model, view):
        """Handle the grid placement model workflow."""
        # Run grid placement when button is clicked
        if st.button("Generate Grid Placement"):
            with st.spinner("Generating grid placement..."):
                circles, cell_coverage = model.solve()
                
                # Visualization
                fig = view.visualize(model, circles, cell_coverage)
                st.pyplot(fig)
                
                # Save results in session state for tables
                st.session_state['circles'] = circles
                st.session_state['cell_coverage'] = cell_coverage
                st.session_state['is_grid_placed'] = True
                st.session_state['is_optimized'] = False
                
                # Calculate average coverage per region (for display purposes)
                bounds = model.room.bounds
                x_min, y_min, x_max, y_max = bounds
                grid_averages = {}
                
                # Create grid cells for averaging
                cell_size = min(model.horizontal_spacing, model.vertical_spacing)
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
            fig = view.create_empty_room_visualization(model.room)
            st.pyplot(fig)
            st.info("Click 'Generate Grid Placement' to create a grid-based light placement.")
        else:
            # Show previously generated grid placement results if they exist
            circles = st.session_state.get('circles', [])
            cell_coverage = st.session_state.get('cell_coverage', {})
            fig = view.visualize(model, circles, cell_coverage)
            st.pyplot(fig)
    
    def display_results(self, model):
        """Display results in the sidebar."""
        st.subheader("Results")
        if 'circles' in st.session_state:
            st.metric("Number of Circles", len(st.session_state['circles']))
            st.metric("Room Area", f"{model.room_area:.2f} sq units")
            
            # Display circle positions
            st.subheader("Circle Positions")
            circle_df_data = [{"X": x, "Y": y} for x, y in st.session_state['circles']]
            if circle_df_data:
                st.dataframe(circle_df_data, height=200)
            
            # Display average coverage
            st.subheader("Grid Cell Coverage")
            if st.session_state.get('grid_averages', {}):
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
    
    def display_model_explanation(self, solution_approach):
        """Display explanation of the selected model."""
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
                        
            ## Model Constraints:
            1. Each grid cell must have a minimum average light level.
            2. Circles cannot be placed too close to each other (based on the minimum circle spacing).
            3. Circles must be placed within the room boundaries.
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
    
    def display_comparison(self):
        """Display comparison when both models have been run."""
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
    
    def run(self):
        """Run the application."""
        # Setup the page
        st.set_page_config(page_title="Circle Coverage Optimizer", layout="wide")
        st.title("Gradient Circle Coverage Optimization")
        st.write("Optimize the placement of circles to provide gradient coverage in a room.")
        
        # Setup sidebar
        solution_approach, room_source = self.setup_sidebar()
        
        # Load room data if JSON source selected
        if room_source == "JSON File":
            self.load_room_from_json()
        
        # Get room vertices
        room_vertices = self.get_room_vertices(room_source)
        
        # Setup parameters based on selected approach
        if solution_approach == "Optimization Model":
            params = self.setup_optimization_params()
        else:
            params = self.setup_grid_placement_params()
        
        # Create model and view
        model, view = self.create_model(solution_approach, room_vertices, params)
        
        # Main content layout
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader("Room and Coverage Visualization")
            
            if solution_approach == "Optimization Model":
                self.handle_optimization_model(model, view, params)
            else:
                self.handle_grid_placement_model(model, view)
        
        with col2:
            self.display_results(model)
        
        # Add explanations
        self.display_model_explanation(solution_approach)
        
        # Comparison section
        self.display_comparison()
        
        # Footer
        st.markdown("---")
        st.caption("Circle Coverage Optimization Tool")