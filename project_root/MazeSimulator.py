import tkinter as tk
from MazeUI import MazeUI
from maze_logic import MazeGenerator, MazeSolver

class MazeSimulator(tk.Tk):
    """Main application class for the maze simulator."""
    def __init__(self):
        super().__init__()
        
        self.title("Algorithm Simulator - Maze Pathfinding")
        self.geometry("800x600")
        self.resizable(True, True)
        
        # Set up variables
        self.maze_size = 15  # 15x15 maze by default
        self.maze = None
        self.solution_path = None
        self.current_step = 0
        self.animation_speed = 500  # milliseconds
        self.after_ids = []  # To track animation steps for cancellation
        
        # Create the UI components
        self.ui = MazeUI(self)
        
        # Generate a random maze to start
        self.generate_new_maze()
    
    def change_maze_size(self):
        """Change the maze size and generate a new maze."""
        new_size = self.ui.size_var.get()
        if new_size != self.maze_size:
            self.maze_size = new_size
            self._reset_state()
            self.generate_new_maze()
            size_text = f"Maze size changed to {self.maze_size}x{self.maze_size}"
            self._update_display(
                f"{size_text}. Click 'Start Simulation' to solve.",
                size_text
            )
    
    def generate_new_maze(self):
        """Generate a new random maze."""
        # Create a maze generator and generate a new maze
        generator = MazeGenerator(self.maze_size, self.maze_size)
        self.maze = generator.generate_dfs_maze()
        
        # Reset the simulation state
        self._reset_state()
        
        # Update the display
        self._update_display(
            "New maze generated. Select an algorithm to start the simulation.",
            "New maze generated"
        )
    
    def start_simulation(self):
        """Start the simulation with the selected algorithm."""
        import time
        import tkinter.messagebox as messagebox
        
        # Cancel any ongoing animation
        for after_id in self.after_ids:
            self.after_cancel(after_id)
        self.after_ids = []
        
        # Reset the simulation state
        self._reset_state()
        
        # Get the selected algorithm
        algorithm = self.ui.algorithm_var.get()
        
        # Create a solver and find a solution
        # Set the start and goal positions
        start_pos = (0, 0)  # Top-left corner
        goal_pos = (self.maze_size-1, self.maze_size-1)  # Bottom-right corner
        
        solver = MazeSolver(self.maze, start_pos, goal_pos)
        start_time = time.time()
        
        # Run the selected algorithm
        if "A*" in algorithm:
            solution, error = solver.solve_a_star()
        elif "Breadth-First" in algorithm:
            solution, error = solver.solve_bfs()
        elif "Depth-First" in algorithm:
            solution, error = solver.solve_dfs()
        elif "Greedy" in algorithm:
            solution, error = solver.solve_greedy()
        elif "Uniform Cost" in algorithm:
            solution, error = solver.solve_uniform_cost()
        else:
            messagebox.showerror("Error", f"Algorithm '{algorithm}' not implemented yet.")
            return
        
        # Check if a solution was found
        if not solution:
            error_msg = "No solution found"
            if error:
                error_msg += f" ({error})"
            messagebox.showinfo("No Solution", error_msg)
            return
        
        # Store the solution path and explored cells
        self.solution_path = []
        self.explored_cells = solver.get_explored_cells()  # Get explored cells from solver
        # Convert the path of moves to a path of positions
        current_pos = start_pos
        self.solution_path.append(current_pos)
        
        # Follow the solution path to get all positions
        for move in solution:
            # Extract direction from move description (e.g., "Move right" -> "right")
            direction = move.split()[-1]
            
            # Update current position based on direction
            if direction == "right":
                current_pos = (current_pos[0], current_pos[1] + 1)
            elif direction == "down":
                current_pos = (current_pos[0] + 1, current_pos[1])
            elif direction == "left":
                current_pos = (current_pos[0], current_pos[1] - 1)
            elif direction == "up":
                current_pos = (current_pos[0] - 1, current_pos[1])
            
            # Verify the new position is valid (not a wall)
            if 0 <= current_pos[0] < self.maze_size and 0 <= current_pos[1] < self.maze_size and self.maze[current_pos[0]][current_pos[1]] != 1:
                self.solution_path.append(current_pos)
            else:
                # If we hit a wall, something is wrong with the solution
                # Log the error and continue
                print(f"Warning: Invalid move to {current_pos} (wall or out of bounds)")
                # Revert to previous position
                current_pos = self.solution_path[-1]
        
        # Update the solution text
        elapsed_time = time.time() - start_time
        solution_text = f"Algorithm: {algorithm}\n"
        solution_text += f"Path length: {len(solution)}\n"
        solution_text += f"Explored cells: {len(self.explored_cells)}\n"
        solution_text += f"Time taken: {elapsed_time:.4f} seconds\n"
        
        self.ui.update_solution_text(solution_text)
        
        # Start the animation
        self._animate_solution()
    
    def _animate_solution(self):
        """Animate the solution path."""
        if not self.solution_path or self.current_step >= len(self.solution_path):
            return
        
        # Get the current position
        current_pos = self.solution_path[self.current_step]
        
        # Draw the maze with the current position and explored cells
        self.ui.draw_maze(self.maze, current_pos, self.solution_path[:self.current_step], self.explored_cells)
        
        # Increment the step counter
        self.current_step += 1
        
        # Schedule the next animation step
        if self.current_step < len(self.solution_path):
            after_id = self.after(self.animation_speed, self._animate_solution)
            self.after_ids.append(after_id)
    
    def reset_simulation(self):
        """Reset the simulation to its initial state."""
        self._reset_state()
        self._update_display(
            "Simulation reset. Click 'Start Simulation' to solve again.",
            "Simulation reset"
        )
    
    def _reset_state(self):
        """Reset the simulation state variables."""
        # Cancel any ongoing animation
        for after_id in self.after_ids:
            self.after_cancel(after_id)
        self.after_ids = []
        
        # Reset the solution path, explored cells and current step
        self.solution_path = None
        self.explored_cells = None
        self.current_step = 0
        
        # Update the display
        if self.maze:
            self.ui.draw_maze(self.maze)
        self.ui.update_solution_text("")
    
    def update_speed(self, value):
        """Update the animation speed.
        The slider is set from 100 (fast) to 1000 (slow),
        so we need to use the value directly as the delay in milliseconds.
        """
        # Convert to float first, then to integer to handle floating point values from the scale widget
        self.animation_speed = int(float(value))  # Convert to integer as the scale passes a string
    
    def algorithm_changed(self, event):
        """Handle algorithm selection change."""
        algorithm = self.ui.algorithm_var.get()
        self.ui.status_var.set(f"Selected algorithm: {algorithm}")
    
    def _update_display(self, status_text, status_var_text):
        """Update the display with the given status text."""
        if self.maze:
            self.ui.draw_maze(self.maze)
        self.ui.status_var.set(status_var_text)

if __name__ == "__main__":
    app = MazeSimulator()
    app.mainloop()