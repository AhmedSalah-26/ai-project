import tkinter as tk
from PuzzleUI import PuzzleUI
from puzzle_logic import PuzzleState ,PuzzleSolver

class PuzzleSimulator(tk.Tk):
    """Main application class for the puzzle simulator."""
    def __init__(self):
        super().__init__()
        
        self.title("Algorithm Simulator - Number Puzzle Sliding Game")
        self.geometry("800x600")
        self.resizable(True, True)
        
        # Set up variables
        self.puzzle_size = 3  
        self.board = None
        self.initial_board = None 
        self.goal_board = self.create_goal_board()
        self.solution_path = None
        self.current_step = 0
        self.animation_speed = 500  
        self.after_ids = []  
        self.manual_mode = False  
        self.selected_tile = None  
        
        self.ui = PuzzleUI(self)
        
        self.generate_random_puzzle()
    
    def create_goal_board(self):
        """Create the goal board configuration."""
        board = []
        for i in range(self.puzzle_size):
            row = []
            for j in range(self.puzzle_size):
                value = i * self.puzzle_size + j + 1
                if value == self.puzzle_size ** 2:
                    value = 0  
                row.append(value)
            board.append(row)
        return board
    
    def change_puzzle_size(self):
        """Change the puzzle size and reset the game."""
        new_size = self.ui.size_var.get()
        if new_size != self.puzzle_size:
            self.puzzle_size = new_size
            self.goal_board = self.create_goal_board()
            self._reset_state()
            self.generate_random_puzzle()
            size_text = f"Puzzle size changed to {self.puzzle_size}x{self.puzzle_size}"
            self._update_display(
                f"{size_text}. Click 'Start Simulation' to solve.",
                size_text
            )
    
    def _find_empty_position(self):
        """Helper method to find the empty tile position."""
        for i in range(self.puzzle_size):
            for j in range(self.puzzle_size):
                if self.board[i][j] == 0:
                    return i, j
        return None, None

    def _make_random_moves(self, empty_i, empty_j):
        """Helper method to make random moves on the board."""
        import random
        moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
        for _ in range(100):  # Make 100 random moves
            random.shuffle(moves)
            for dx, dy in moves:
                new_i, new_j = empty_i + dx, empty_j + dy
                if 0 <= new_i < self.puzzle_size and 0 <= new_j < self.puzzle_size:
                    self.board[empty_i][empty_j] = self.board[new_i][new_j]
                    self.board[new_i][new_j] = 0
                    empty_i, new_i = new_i, empty_i
                    empty_j, new_j = new_j, empty_j
                    break
        return empty_i, empty_j

    def generate_random_puzzle(self):
        """Generate a random, solvable puzzle configuration."""
        import copy
        self.board = copy.deepcopy(self.goal_board)
        empty_i, empty_j = self._find_empty_position()
        self._make_random_moves(empty_i, empty_j)
        self.initial_board = copy.deepcopy(self.board)
        self._reset_state()
        self._update_display(
            "Generate a random puzzle and select an algorithm to start the simulation.",
            "Random puzzle generated"
        )
    
    def start_simulation(self):
        """Start the simulation with the selected algorithm."""
        import time
        import tkinter.messagebox as messagebox
        import threading
        import queue
        
        if self.board is None:
            messagebox.showerror("Error", "Please generate a puzzle first.")
            return
        
        # Check for Q-Learning impracticality
        algorithm = self.ui.algorithm_var.get()
        if algorithm == "Q-Learning" and self.puzzle_size in (3, 4):
            messagebox.showwarning(
                "Q-Learning Not Practical",
                "Q-Learning is not practical for 3x3 or 4x4 puzzles due to the enormous state space.\n"
                "It is included for demonstration purposes only and will not solve the puzzle efficiently.\n\n"
                "Please use A*, BFS, or another search algorithm for actual puzzle solving."
            )
            return
        
        result_queue = queue.Queue()
        
        initial_state = PuzzleState(self.board)
        goal_state = PuzzleState(self.goal_board)
        
        solver = PuzzleSolver(initial_state, goal_state)
        if self.puzzle_size == 4:
            solver.timeout = 120  
        
        # To solve the puzzle with the selected algorithm
        self.ui.status_var.set(f"Solving with {algorithm}...")
        self.update()
        
        start_time = time.time()
        
        def solve_puzzle():
            try:
                if algorithm == "A*":
                    solution = solver.solve_a_star()
                elif algorithm == "Breadth-First Search (BFS)":
                    solution = solver.solve_bfs()
                elif algorithm == "Depth-First Search (DFS)":
                    solution = solver.solve_dfs()
                elif algorithm == "Greedy Search":
                    solution = solver.solve_greedy()
                elif algorithm == "Uniform Cost Search":
                    solution = solver.solve_uniform_cost()
                elif algorithm == "Genetic Algorithm":
                    solution = solver.solve_genetic()
                elif algorithm == "Q-Learning":
                    solution = solver.solve_q_learning()
                result_queue.put((True, solution))
            except Exception as e:
                result_queue.put((False, e))
        
        # To start solving in background thread
        solver_thread = threading.Thread(target=solve_puzzle)
        solver_thread.daemon = True  # The thread will be terminated when main program exits
        solver_thread.start()
        
        def check_solution():
            if solver_thread.is_alive():
                self.after(100, check_solution)
                return
            
            try:
                success, result = result_queue.get_nowait()
                end_time = time.time()
                elapsed_time = end_time - start_time
                
                if success:
                    self.solution_path, stop_reason = result
                    if self.solution_path:
                        # To display the solution
                        solution_text = f"Solution found with {algorithm} in {elapsed_time:.2f} seconds\n"
                        solution_text += f"Total Steps: {len(self.solution_path)}\n\nPath:\n"
                        
                        for i, move in enumerate(self.solution_path, 1):
                            solution_text += f"{i}. {move}\n"
                        
                        self.ui.update_solution_text(solution_text)
                        self.ui.status_var.set(f"Solution found with {len(self.solution_path)} steps")
                        
                        self.current_step = 0
                        self.animate_solution()
                    else:
                        if stop_reason == "timeout":
                            message = f"Timeout reached after {elapsed_time:.2f} seconds. The {algorithm} algorithm couldn't find a solution in time."
                            if self.puzzle_size == 4:
                                message += "\n\nNote: 4x4 puzzles have a much larger state space and may take longer to solve."
                                message += "\nTry using a different algorithm or generating a simpler puzzle."
                        elif stop_reason == "state_limit":
                            message = f"State limit reached after exploring {solver.max_states:,} states. The {algorithm} algorithm stopped to prevent excessive memory usage."
                            if self.puzzle_size == 4:
                                message += "\n\nNote: 4x4 puzzles have a much larger state space and require more memory."
                                message += "\nTry using a different algorithm or generating a simpler puzzle."
                        else:
                            message = f"No solution found with {algorithm} in {elapsed_time:.2f} seconds."
                        
                        self.ui.update_solution_text(message)
                        self.ui.status_var.set("No solution found")
                else:
                    error = result
                    if isinstance(error, MemoryError):
                        message = f"Out of memory error while solving with {algorithm} after {elapsed_time:.2f} seconds.\n"
                        if self.puzzle_size == 4:
                            message += "4x4 puzzles require significantly more memory.\n\n"
                            message += "Recommendations:\n"
                            message += "1. Try a different algorithm (A* or Greedy usually work best)\n"
                            message += "2. Try a 3x3 puzzle instead\n"
                            message += "3. Generate a new random puzzle that might be easier to solve"
                        self.ui.update_solution_text(message)
                        self.ui.status_var.set("Out of memory error")
                        print(f"Memory error in {algorithm} for {self.puzzle_size}x{self.puzzle_size} puzzle")
                    else:
                        message = f"Error occurred while solving with {algorithm} after {elapsed_time:.2f} seconds.\n"
                        message += f"Error details: {str(error)}\n\n"
                        if self.puzzle_size == 4:
                            message += "Note: 4x4 puzzles are much more complex and may cause errors.\n"
                            message += "Try using a different algorithm or generating a simpler puzzle configuration."
                        self.ui.update_solution_text(message)
                        self.ui.status_var.set("Error occurred")
                        print(f"Error in {algorithm} for {self.puzzle_size}x{self.puzzle_size} puzzle: {error}")
                        import traceback
                        traceback.print_exc()
            except queue.Empty:
                self.ui.status_var.set("Error: No result from solver")
        
        # Start checking for solution
        check_solution()
    
    def animate_solution(self):
        """Animate the solution step by step."""
        if not self.solution_path or self.current_step >= len(self.solution_path):
            return
        
        move = self.solution_path[self.current_step]
        self.ui.status_var.set(f"Step {self.current_step + 1}/{len(self.solution_path)}: {move}")
        
        parts = move.split()
        tile_value = int(parts[1])
        direction = parts[2]
        
        tile_i, tile_j = None, None
        for i in range(self.puzzle_size):
            for j in range(self.puzzle_size):
                if self.board[i][j] == tile_value:
                    tile_i, tile_j = i, j
                    break
        
        empty_i, empty_j = None, None
        for i in range(self.puzzle_size):
            for j in range(self.puzzle_size):
                if self.board[i][j] == 0:
                    empty_i, empty_j = i, j
                    break
        
        self.board[empty_i][empty_j] = tile_value
        self.board[tile_i][tile_j] = 0
        
        # Update the display
        self.ui.draw_puzzle(self.board, self.puzzle_size, self.manual_mode, self.selected_tile)
        
        self.current_step += 1
        
        if self.current_step < len(self.solution_path):
            after_id = self.after(self.animation_speed, self.animate_solution)
            self.after_ids.append(after_id)
        else:
            self.ui.status_var.set(f"Animation complete. Total steps: {len(self.solution_path)}")
            self.after_ids = []
    
    def _reset_state(self):
        """Helper method to reset the puzzle state."""
        self.solution_path = None
        self.current_step = 0

    def _update_display(self, solution_text, status_text):
        """Helper method to update the UI display."""
        self.ui.draw_puzzle(self.board, self.puzzle_size, self.manual_mode, self.selected_tile)
        self.ui.update_solution_text(solution_text)
        self.ui.status_var.set(status_text)

    def reset_simulation(self):
        """Reset the simulation to the initial state."""
        import copy
        if self.board is None or self.initial_board is None:
            return
        
        self._reset_state()
        self.board = copy.deepcopy(self.initial_board)
        self._update_display(
            "Puzzle reset to initial state. Click 'Start Simulation' to solve.",
            "Simulation reset to initial puzzle"
        )
    
    def update_speed(self, value):
        """Update the animation speed."""
        try:
            value = int(float(value))
        except (ValueError, TypeError):
            pass
        
        if self.ui.speed_var.get() != value:
            self.ui.speed_var.set(value)
            
        self.animation_speed = 1100 - value
        
        # Update status bar
        self.ui.status_var.set(f"Animation speed updated: {1100 - self.animation_speed} (higher = faster)")
        
        # IApply the new speed 
        if self.solution_path and self.current_step < len(self.solution_path):
            for after_id in self.after_ids:
                self.after_cancel(after_id)
            self.animate_solution()

    def algorithm_changed(self, event=None):
        """Handle algorithm selection changes."""
        algorithm = self.ui.algorithm_var.get()
        self.ui.status_var.set(f"Algorithm changed to {algorithm}")
        if self.solution_path:
            self.solution_path = None
            self.current_step = 0
            self.ui.update_solution_text(f"Algorithm changed to {algorithm}. Click 'Start Simulation' to solve with the new algorithm.")
    
    def toggle_manual_mode(self):
        """Toggle between manual mode and normal mode."""
        self.manual_mode = not self.manual_mode
        
        if self.manual_mode:
            self.ui.status_var.set("Manual Mode: Click tiles to rearrange them")
            self.ui.update_solution_text("Manual Mode Instructions:\n\n"
                                    "1. Click on a tile to select it (highlighted in green)\n"
                                    "2. Click on another tile or the empty space to swap positions\n"
                                    "3. Create your custom puzzle configuration\n"
                                    "4. Click 'Manual Mode' again to exit and confirm your puzzle\n\n"
                                    "Note: The puzzle must be solvable to run a simulation.")
            # Reset selection
            self.selected_tile = None
        else:
            if self.is_valid_puzzle():
                self.ui.status_var.set("Manual Mode: Custom puzzle confirmed")
                self.ui.update_solution_text("Custom puzzle created. Click 'Start Simulation' to solve.")
            else:
                self.ui.status_var.set("Warning: Puzzle may not be solvable")
                self.ui.update_solution_text("Warning: The current puzzle configuration may not be solvable.\n"
                                        "You can still try to run the simulation, but it might not find a solution.")
        
        self.ui.draw_puzzle(self.board, self.puzzle_size, self.manual_mode, self.selected_tile)
    
    def on_tile_click(self, row, col):
        """Handle tile clicks in manual mode."""
        if not self.manual_mode:
            return
        
        if self.selected_tile is None:
            self.selected_tile = (row, col)
            self.ui.status_var.set(f"Tile {self.board[row][col]} selected. Click another tile to swap.")
        else:
            prev_row, prev_col = self.selected_tile
            
            self.board[prev_row][prev_col], self.board[row][col] = self.board[row][col], self.board[prev_row][prev_col]
            
            self.selected_tile = None
            self.ui.status_var.set("Tiles swapped. Select another tile or confirm the puzzle.")
        
        self.solution_path = None
        self.current_step = 0
        
        self.ui.draw_puzzle(self.board, self.puzzle_size, self.manual_mode, self.selected_tile)
    
    def is_valid_puzzle(self):
        """Check if the current puzzle configuration is valid and likely solvable."""
        flat_board = [item for row in self.board for item in row]
        expected_values = list(range(self.puzzle_size * self.puzzle_size))
        
        if sorted(flat_board) != expected_values:
            return False
        
        if self.puzzle_size == 3:
            empty_row = 0
            for i in range(self.puzzle_size):
                for j in range(self.puzzle_size):
                    if self.board[i][j] == 0:
                        empty_row = self.puzzle_size - i
                        break
            
            inversions = 0
            for i in range(len(flat_board)):
                if flat_board[i] == 0:
                    continue
                for j in range(i + 1, len(flat_board)):
                    if flat_board[j] == 0:
                        continue
                    if flat_board[i] > flat_board[j]:
                        inversions += 1
            
            # For a 3x3 puzzle:
            # If the empty tile is on an odd row from the bottom, the number of inversions must be even
            # If the empty tile is on an even row from the bottom, the number of inversions must be odd
            return (empty_row % 2 == 1 and inversions % 2 == 0) or (empty_row % 2 == 0 and inversions % 2 == 1)
        
        # For 4x4 puzzles we'll just assume it's valid to avoid complex calculations
        return True

    def return_to_home(self):
        """Return to the home page."""
        self.destroy()
        from simulator import SimulatorStartPage
        app = SimulatorStartPage()
        app.mainloop()


if __name__ == "__main__":
    app = PuzzleSimulator()
    app.mainloop()
