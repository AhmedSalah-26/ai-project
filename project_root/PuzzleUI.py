import tkinter as tk
from tkinter import ttk

class PuzzleUI:
    """User interface components for the Puzzle Simulator."""
    def __init__(self, master):
        self.master = master
        
        # Set up variables
        self.size_var = tk.IntVar(value=3)
        self.algorithm_var = tk.StringVar(value="A*")
        self.speed_var = tk.IntVar(value=500)
        self.status_var = tk.StringVar(value="Ready")
        
        # Create UI components
        self.create_widgets()
        
    def create_widgets(self):
        """Create all the UI components."""
        # Main frame
        main_frame = ttk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Left panel for controls
        left_panel = ttk.Frame(main_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))
        
        # Puzzle size selection
        size_frame = ttk.LabelFrame(left_panel, text="Puzzle Size")
        size_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Radiobutton(size_frame, text="3x3", variable=self.size_var, value=3, command=self.master.change_puzzle_size).pack(anchor=tk.W, padx=10, pady=5)
        ttk.Radiobutton(size_frame, text="4x4", variable=self.size_var, value=4, command=self.master.change_puzzle_size).pack(anchor=tk.W, padx=10, pady=5)
        
        # Algorithm selection
        algo_frame = ttk.LabelFrame(left_panel, text="Algorithm")
        algo_frame.pack(fill=tk.X, pady=(0, 10))
        
        algorithms = ["A*", "Breadth-First Search (BFS)", "Depth-First Search (DFS)", "Greedy Search", "Uniform Cost Search", "Genetic Algorithm"]
        algo_dropdown = ttk.Combobox(algo_frame, textvariable=self.algorithm_var, values=algorithms, state="readonly")
        algo_dropdown.pack(padx=10, pady=10, fill=tk.X)
        # Bind the combobox selection event
        algo_dropdown.bind("<<ComboboxSelected>>", self.master.algorithm_changed)
        
        # Control buttons
        control_frame = ttk.Frame(left_panel)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(control_frame, text="Generate Random Puzzle", command=self.master.generate_random_puzzle).pack(fill=tk.X, pady=5)
        ttk.Button(control_frame, text="Manual Mode", command=self.master.toggle_manual_mode).pack(fill=tk.X, pady=5)
        ttk.Button(control_frame, text="Start Simulation", command=self.master.start_simulation).pack(fill=tk.X, pady=5)
        ttk.Button(control_frame, text="Reset", command=self.master.reset_simulation).pack(fill=tk.X, pady=5)
        
        # Animation speed control
        speed_frame = ttk.LabelFrame(left_panel, text="Animation Speed")
        speed_frame.pack(fill=tk.X, pady=(0, 10))
        
        speed_scale = ttk.Scale(speed_frame, from_=100, to=1000, orient=tk.HORIZONTAL, variable=self.speed_var, command=self.master.update_speed)
        speed_scale.pack(padx=10, pady=10, fill=tk.X)
        ttk.Label(speed_frame, text="Fast").pack(side=tk.RIGHT, padx=10)
        ttk.Label(speed_frame, text="Slow").pack(side=tk.LEFT, padx=10)
        
        # Right panel for puzzle display and solution
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Puzzle display
        puzzle_frame = ttk.LabelFrame(right_panel, text="Puzzle")
        puzzle_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.puzzle_canvas = tk.Canvas(puzzle_frame, bg="white")
        self.puzzle_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Solution display
        solution_frame = ttk.LabelFrame(right_panel, text="Solution")
        solution_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.solution_text = tk.Text(solution_frame, height=10, wrap=tk.WORD, state=tk.DISABLED)
        self.solution_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Status bar
        status_bar = ttk.Label(self.master, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def draw_puzzle(self, board, puzzle_size, manual_mode=False, selected_tile=None):
        """Draw the puzzle on the canvas."""
        self.puzzle_canvas.delete("all")
        
        # Calculate the size of each tile
        canvas_width = self.puzzle_canvas.winfo_width()
        canvas_height = self.puzzle_canvas.winfo_height()
        
        # Ensure the canvas has a minimum size
        if canvas_width < 50 or canvas_height < 50:
            canvas_width = max(300, canvas_width)
            canvas_height = max(300, canvas_height)
            self.puzzle_canvas.config(width=canvas_width, height=canvas_height)
        
        tile_size = min(canvas_width, canvas_height) // puzzle_size
        
        # Draw the tiles
        for i in range(puzzle_size):
            for j in range(puzzle_size):
                value = board[i][j]
                x1 = j * tile_size
                y1 = i * tile_size
                x2 = x1 + tile_size
                y2 = y1 + tile_size
                
                if value != 0:  # Skip the empty space for drawing the tile
                    # Determine tile color based on selection in manual mode
                    fill_color = "lightblue"
                    if manual_mode and selected_tile == (i, j):
                        fill_color = "lightgreen"  # Highlight selected tile
                    
                    # Draw the tile
                    self.puzzle_canvas.create_rectangle(x1, y1, x2, y2, fill=fill_color, outline="black")
                    
                    # Draw the number
                    self.puzzle_canvas.create_text((x1 + x2) // 2, (y1 + y2) // 2, text=str(value), font=("Arial", 20, "bold"))
                else:
                    # Draw a light outline for the empty space in manual mode
                    if manual_mode:
                        self.puzzle_canvas.create_rectangle(x1, y1, x2, y2, fill="white", outline="gray", dash=(2, 2))
                
                # Add click binding for manual mode
                if manual_mode:
                    # Create a transparent rectangle for click detection
                    tag = f"tile_{i}_{j}"
                    self.puzzle_canvas.create_rectangle(x1, y1, x2, y2, fill="", outline="", tags=tag)
                    self.puzzle_canvas.tag_bind(tag, "<Button-1>", lambda event, row=i, col=j: self.master.on_tile_click(row, col))
    
    def update_solution_text(self, text):
        """Update the solution text display."""
        self.solution_text.config(state=tk.NORMAL)
        self.solution_text.delete(1.0, tk.END)
        self.solution_text.insert(tk.END, text)
        self.solution_text.config(state=tk.DISABLED)
