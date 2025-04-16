import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

class MazeUI:
    """User interface components for the Maze Simulator."""
    def __init__(self, master):
        self.master = master
        
        # Set up variables
        self.size_var = tk.IntVar(value=15)  # Default maze size 15x15
        self.algorithm_var = tk.StringVar(value="A*")
        self.speed_var = tk.IntVar(value=500)
        self.status_var = tk.StringVar(value="Ready")
        self.car_position = (0, 0)  # Initial car position at the start (0, 0)
        
        # Create UI components
        self.create_widgets()
        
        # Load car image
        self.car_image = Image.open(r"E:/projects\Algorithm_Simulator/assest/car.png")  # Replace with the path to your car image
        
    def create_widgets(self):
        """Create all the UI components."""
        # Main frame
        main_frame = ttk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Left panel for controls
        left_panel = ttk.Frame(main_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))
        
        # Maze size selection
        size_frame = ttk.LabelFrame(left_panel, text="Maze Size")
        size_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Radiobutton(size_frame, text="10x10", variable=self.size_var, value=10, command=self.master.change_maze_size).pack(anchor=tk.W, padx=10, pady=5)
        ttk.Radiobutton(size_frame, text="15x15", variable=self.size_var, value=15, command=self.master.change_maze_size).pack(anchor=tk.W, padx=10, pady=5)
        ttk.Radiobutton(size_frame, text="20x20", variable=self.size_var, value=20, command=self.master.change_maze_size).pack(anchor=tk.W, padx=10, pady=5)
        ttk.Radiobutton(size_frame, text="30x30", variable=self.size_var, value=30, command=self.master.change_maze_size).pack(anchor=tk.W, padx=10, pady=5)
        ttk.Radiobutton(size_frame, text="40x40", variable=self.size_var, value=40, command=self.master.change_maze_size).pack(anchor=tk.W, padx=10, pady=5)

        # Algorithm selection
        algo_frame = ttk.LabelFrame(left_panel, text="Algorithm")
        algo_frame.pack(fill=tk.X, pady=(0, 10))
        
        algorithms = ["A*", "Breadth-First Search (BFS)", "Depth-First Search (DFS)", "Greedy Search", "Uniform Cost Search"]
        algo_dropdown = ttk.Combobox(algo_frame, textvariable=self.algorithm_var, values=algorithms, state="readonly")
        algo_dropdown.pack(padx=10, pady=10, fill=tk.X)
        # Bind the combobox selection event
        algo_dropdown.bind("<<ComboboxSelected>>", self.master.algorithm_changed)
        
        # Control buttons
        control_frame = ttk.Frame(left_panel)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(control_frame, text="Generate New Maze", command=self.master.generate_new_maze).pack(fill=tk.X, pady=5)
        ttk.Button(control_frame, text="Start Simulation", command=self.master.start_simulation).pack(fill=tk.X, pady=5)
        ttk.Button(control_frame, text="Reset", command=self.master.reset_simulation).pack(fill=tk.X, pady=5)
        
        # Animation speed control
        speed_frame = ttk.LabelFrame(left_panel, text="Animation Speed")
        speed_frame.pack(fill=tk.X, pady=(0, 10))
        
        speed_scale = ttk.Scale(speed_frame, from_=1000, to=0.1, orient=tk.HORIZONTAL, variable=self.speed_var, command=self.master.update_speed)
        speed_scale.pack(padx=10, pady=10, fill=tk.X)
        ttk.Label(speed_frame, text="Slow").pack(side=tk.LEFT, padx=10)
        ttk.Label(speed_frame, text="Fast").pack(side=tk.RIGHT, padx=10)
        
        # Right panel for maze display and solution
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Maze display
        maze_frame = ttk.LabelFrame(right_panel, text="Maze")
        maze_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.maze_canvas = tk.Canvas(maze_frame, bg="white")
        self.maze_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Solution display
        solution_frame = ttk.LabelFrame(right_panel, text="Solution")
        solution_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.solution_text = tk.Text(solution_frame, height=10, wrap=tk.WORD, state=tk.DISABLED)
        self.solution_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def draw_maze(self, maze, current_pos=None, path=None, explored_cells=None):
        """Draw the maze on the canvas."""
        # Only update canvas size if window size has changed
        window_width = self.master.winfo_width()
        window_height = self.master.winfo_height()
        canvas_size = min(window_width * 0.6, window_height * 0.6)  # Use 60% of the smallest window dimension
        
        current_width = self.maze_canvas.winfo_width()
        current_height = self.maze_canvas.winfo_height()
        
        if abs(current_width - canvas_size) > 5 or abs(current_height - canvas_size) > 5:
            self.maze_canvas.config(width=canvas_size, height=canvas_size)
            self.maze_canvas.update()
            self.maze_canvas.delete("all")  # Only clear if size changed
        else:
            # Clear only dynamic elements (car and path)
            self.maze_canvas.delete("car", "path", "explored")
        
        # Calculate the size of each cell
        canvas_width = self.maze_canvas.winfo_width()
        canvas_height = self.maze_canvas.winfo_height()
        
        # Ensure the canvas has a minimum size
        if canvas_width < 50 or canvas_height < 50:
            canvas_width = max(400, canvas_width)
            canvas_height = max(400, canvas_height)
            self.maze_canvas.config(width=canvas_width, height=canvas_height)
        
        rows = len(maze)
        cols = len(maze[0]) if rows > 0 else 0
        
        cell_width = canvas_width / cols if cols > 0 else 0
        cell_height = canvas_height / rows if rows > 0 else 0
        
        # Resize the car image based on cell size
        car_size = (int(cell_width * 0.8), int(cell_height * 0.8))  # Resize the image to fit the cell
        self.car_image = self.car_image.resize(car_size, Image.Resampling.LANCZOS)
        self.car_photo = ImageTk.PhotoImage(self.car_image)
        
        # Draw the maze cells
        for i in range(rows):
            for j in range(cols):
                x1 = j * cell_width
                y1 = i * cell_height
                x2 = x1 + cell_width
                y2 = y1 + cell_height
                
                # Determine cell color
                if maze[i][j] == 1:  # Wall
                    fill_color = "black"
                elif (i, j) == (0, 0):  # Start
                    fill_color = "green"
                elif (i, j) == (rows-1, cols-1):  # End
                    fill_color = "red"
                elif explored_cells and (i, j) in explored_cells:  # Explored cell
                    fill_color = "#FFE4B5"  # Light orange for explored cells
                else:  # Unexplored path
                    fill_color = "white"
                
                # Draw static cells only if canvas was cleared
                if abs(current_width - canvas_size) > 5 or abs(current_height - canvas_size) > 5:
                    self.maze_canvas.create_rectangle(x1, y1, x2, y2, fill=fill_color, outline="gray")
                elif explored_cells and (i, j) in explored_cells:
                    # Draw explored cells with tag for easy removal
                    self.maze_canvas.create_rectangle(x1, y1, x2, y2, fill=fill_color, outline="gray", tags="explored")
        
        # Draw the current position if provided (moving car)
        if current_pos:
            i, j = current_pos
            x1 = j * cell_width
            y1 = i * cell_height
            # Draw the car image with tag for easy removal
            self.maze_canvas.create_image(x1 + cell_width / 2, y1 + cell_height / 2, 
                                        image=self.car_photo, tags="car")
        
        # Draw the path if provided
        if path:
            for pos in path:
                i, j = pos
                x1 = j * cell_width
                y1 = i * cell_height
                x2 = x1 + cell_width
                y2 = y1 + cell_height
                
                # Draw path circles with tag for easy removal
                self.maze_canvas.create_oval(x1 + cell_width/3, y1 + cell_height/3, 
                                           x2 - cell_width/3, y2 - cell_height/3, 
                                           fill="yellow", tags="path")
    
    def update_solution_text(self, text):
        """Update the solution text display."""
        self.solution_text.config(state=tk.NORMAL)
        self.solution_text.delete(1.0, tk.END)
        self.solution_text.insert(tk.END, text)
        self.solution_text.config(state=tk.DISABLED)
