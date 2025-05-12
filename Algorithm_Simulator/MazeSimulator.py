import tkinter as tk
from MazeUI import MazeUI
from maze_logic import MazeGenerator, MazeSolver

class MazeSimulator(tk.Tk):
    """Main application class for the maze simulator."""
    def __init__(self):
        super().__init__()
        
        self.title("Algorithm Simulator - Maze Pathfinding")
        self.geometry("1024x768")
        self.resizable(True, True)
        
  
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = (screen_width - 1024) // 2
        y = (screen_height - 768) // 2
        self.geometry(f"1024x768+{x}+{y}")
        
        # Set up variables
        self.maze_size = 15 
        self.maze = None
        self.solution_path = None
        self.current_step = 0
        self.animation_speed = 500  
        self.after_ids = []  
        
        self.bind("<Configure>", self._on_window_resize)
        
        self.ui = MazeUI(self)
        
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
        generator = MazeGenerator(self.maze_size, self.maze_size)
        self.maze = generator.generate_dfs_maze()
        
        self._reset_state()
        
        self._update_display(
            "New maze generated. Select an algorithm to start the simulation.",
            "New maze generated"
        )
    
    def start_simulation(self):
        """Start the simulation with the selected algorithm."""
        import time
        import tkinter.messagebox as messagebox
        
        for after_id in self.after_ids:
            self.after_cancel(after_id)
        self.after_ids = []
        
        self._reset_state()
        
        algorithm = self.ui.algorithm_var.get()
        

        # Set the start and goal positions
        start_pos = (0, 0)  # Top-left corner
        goal_pos = (self.maze_size-1, self.maze_size-1)  # Bottom-right corner
        
        solver = MazeSolver(self.maze, start_pos, goal_pos)
        start_time = time.time()
        
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
        elif "Q-Learning" in algorithm:
            solution, error = solver.solve_q_learning()
        else:
            messagebox.showerror("Error", f"Algorithm '{algorithm}' not implemented yet.")
            return
        
        if not solution:
            error_msg = "No solution found"
            if error:
                error_msg += f" ({error})"
            messagebox.showinfo("No Solution", error_msg)
            return
        
        self.solution_path = []
        self.explored_cells = solver.get_explored_cells() 
        current_pos = start_pos
        self.solution_path.append(current_pos)
        
        for move in solution:
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

                print(f"Warning: Invalid move to {current_pos} (wall or out of bounds)")
                current_pos = self.solution_path[-1]
        
        self.ui.update_solution_text("⏳ Finding solution...")
        self.update()
        
        elapsed_time = time.time() - start_time
        solution_text = f"Algorithm: {algorithm}\n"
        solution_text += f"Path length: {len(solution)}\n"
        solution_text += f"Explored cells: {len(self.explored_cells)}\n"
        solution_text += f"Time taken: {elapsed_time:.4f} seconds\n"
        solution_text += "\n🚗 Animating solution path..."
        
        self.ui.update_solution_text(solution_text)
        
        self._animate_solution()
    
    def _animate_solution(self):
        """Animate the solution path."""
        if not self.solution_path or self.current_step >= len(self.solution_path):
            return
        
        current_pos = self.solution_path[self.current_step]
        
        self.ui.draw_maze(self.maze, current_pos, self.solution_path[:self.current_step], self.explored_cells)
        
        self.current_step += 1
        
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
        for after_id in self.after_ids:
            self.after_cancel(after_id)
        self.after_ids = []
        
        self.solution_path = None
        self.explored_cells = None
        self.current_step = 0
        
        if self.maze:
            self.ui.draw_maze(self.maze)
        self.ui.update_solution_text("")
    
    def update_speed(self, value):
        """Update the animation speed.
        The slider is set from 100 (fast) to 1000 (slow),
        so we need to use the value directly as the delay in milliseconds.
        """
        self.animation_speed = int(float(value))  
    
    def algorithm_changed(self, event):
        """Handle algorithm selection change."""
        algorithm = self.ui.algorithm_var.get()
        self.ui.status_var.set(f"Selected algorithm: {algorithm}")
    
    def _update_display(self, status_text, status_var_text):
        """Update the display with the given status text."""
        if self.maze:
            self.ui.draw_maze(self.maze)
        self.ui.status_var.set(status_var_text)

    def _on_window_resize(self, event):
        """Handle window resize events."""
        if self.maze and event.widget == self:
            # Redraw the maze with current state
            current_pos = self.solution_path[self.current_step - 1] if self.solution_path and self.current_step > 0 else None
            self.ui.draw_maze(self.maze, current_pos, 
                            self.solution_path[:self.current_step] if self.solution_path else None, 
                            self.explored_cells)

    def return_to_home(self):
        """Return to the home page."""
        self.destroy()
        from simulator import SimulatorStartPage
        app = SimulatorStartPage()
        app.mainloop()

if __name__ == "__main__":
    app = MazeSimulator()
    app.mainloop()