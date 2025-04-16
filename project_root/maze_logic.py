import random
import time
import heapq
from collections import deque
import copy

class MazeState:
    """Represents a state in the maze."""
    def __init__(self, position, parent=None, move=None, depth=0):
        self.position = position  # (row, col)
        self.parent = parent
        self.move = move  # The move that led to this state
        self.depth = depth  # Depth in the search tree
    
    def get_neighbors(self, maze):
        """Generate all possible next states by moving in valid directions."""
        neighbors = []
        moves = [(0, 1, 'right'), (1, 0, 'down'), (0, -1, 'left'), (-1, 0, 'up')]
        
        for dx, dy, direction in moves:
            new_x, new_y = self.position[0] + dx, self.position[1] + dy
            
            # Check if the new position is valid (within bounds and not a wall)
            if (0 <= new_x < len(maze) and 0 <= new_y < len(maze[0]) and 
                maze[new_x][new_y] != 1):  # 1 represents a wall
                
                # Create a new state with the new position
                move_description = f"Move {direction}"
                neighbors.append(MazeState((new_x, new_y), self, move_description, self.depth + 1))
        
        return neighbors
    
    def __eq__(self, other):
        return self.position == other.position
    
    def __hash__(self):
        # Convert position to a hashable tuple to ensure proper hashing
        return hash((self.position[0], self.position[1]))
    
    def __lt__(self, other):
        # For priority queue in A* algorithm
        return self.depth < other.depth

class MazeGenerator:
    """Generates random mazes using various algorithms."""
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
    
    def generate_dfs_maze(self):
        """Generate a maze using Depth-First Search algorithm."""
        # Initialize maze with all walls
        maze = [[1 for _ in range(self.cols)] for _ in range(self.rows)]
        
        # Choose random starting point (must be odd coordinates)
        start_row = random.randrange(1, self.rows, 2)
        start_col = random.randrange(1, self.cols, 2)
        
        # Set starting point as path
        maze[start_row][start_col] = 0
        
        # Stack for backtracking
        stack = [(start_row, start_col)]
        
        # Directions: right, down, left, up
        directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
        
        while stack:
            current_row, current_col = stack[-1]
            
            # Find unvisited neighbors
            unvisited = []
            random.shuffle(directions)
            
            for dr, dc in directions:
                new_row, new_col = current_row + dr, current_col + dc
                
                if (0 <= new_row < self.rows and 0 <= new_col < self.cols and 
                    maze[new_row][new_col] == 1):
                    unvisited.append((new_row, new_col, dr, dc))
            
            if unvisited:
                # Choose a random unvisited neighbor
                new_row, new_col, dr, dc = unvisited[0]
                
                # Remove the wall between current cell and chosen cell
                maze[current_row + dr//2][current_col + dc//2] = 0
                
                # Mark the chosen cell as visited
                maze[new_row][new_col] = 0
                
                # Push the chosen cell to the stack
                stack.append((new_row, new_col))
            else:
                # Backtrack
                stack.pop()
        
        # Set entrance and exit
        # Always set the top-left corner as entrance
        maze[0][0] = 0  # Set entrance
        
        # Always set the bottom-right corner as exit
        maze[self.rows-1][self.cols-1] = 0  # Set exit
        
        # Make sure there's a path from entrance to exit
        # Connect entrance to first row path
        if maze[1][0] == 1:  # If wall below entrance
            maze[1][0] = 0   # Create path
        
        # Connect exit to last row path
        if maze[self.rows-2][self.cols-1] == 1:  # If wall above exit
            maze[self.rows-2][self.cols-1] = 0   # Create path
        
        return maze
    
    def generate_random_maze(self):
        """Generate a simple random maze."""
        # Initialize maze with all paths
        maze = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        
        # Add random walls
        wall_density = 0.3  # Adjust this value to control maze difficulty
        
        for row in range(1, self.rows-1):
            for col in range(1, self.cols-1):
                if random.random() < wall_density:
                    maze[row][col] = 1  # Add a wall
        
        # Ensure start and end are clear
        maze[0][0] = 0  # Start
        maze[self.rows-1][self.cols-1] = 0  # End
        
        # Make sure there's a path from start to end
        # This is a simple approach - in a real implementation, you might want to check
        # if the maze is solvable and regenerate if not
        
        return maze

class MazeSolver:
    """Contains algorithms to solve mazes."""
    def __init__(self, maze, start_pos, goal_pos):
        self.maze = maze
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.timeout = 30  # Default timeout in seconds
        self.start_time = None
        self.max_states = 100000  # Maximum number of states to explore
    
    def manhattan_distance(self, state):
        """Calculate the Manhattan distance heuristic."""
        return (abs(state.position[0] - self.goal_pos[0]) + 
                abs(state.position[1] - self.goal_pos[1]))
    
    def solve_a_star(self):
        """Solve the maze using A* algorithm."""
        self.start_time = time.time()
        open_set = []
        closed_set = set()
        states_explored = 0  # Counter for states explored
        self.explored_cells = set()  # Reset explored cells
        
        # Create initial state
        initial_state = MazeState(self.start_pos)
        
        # Add the initial state to the open set with f-score = g-score (0) + h-score
        h_score = self.manhattan_distance(initial_state)
        heapq.heappush(open_set, (h_score, 0, initial_state))
        counter = 1  # Tie-breaker for states with the same priority
        
        try:
            while open_set:
                # Check for timeout
                if time.time() - self.start_time > self.timeout:
                    return None, "timeout"  # Timeout reached
                
                # Check if we've reached the state limit
                if states_explored >= self.max_states:
                    return None, "state_limit"  # State limit reached
                
                # Get the state with the lowest f-score
                f_score, _, current_state = heapq.heappop(open_set)
                states_explored += 1  # Increment state counter
                
                # Skip if already in closed set
                if hash(current_state) in closed_set:
                    continue
                
                self.explored_cells.add(current_state.position)  # Track explored cell
                
                # Check if we've reached the goal
                if current_state.position == self.goal_pos:
                    return self.reconstruct_path(current_state), None
                
                # Add the current state to the closed set
                closed_set.add(hash(current_state))
                
                # Explore neighbors
                for neighbor in current_state.get_neighbors(self.maze):
                    if hash(neighbor) in closed_set:
                        continue
                    
                    # Calculate the f-score (g + h)
                    g_score = current_state.depth + 1  # Cost to reach neighbor
                    h_score = self.manhattan_distance(neighbor)
                    f_score = g_score + h_score
                    
                    # Update neighbor's depth
                    neighbor.depth = g_score
                    
                    # Add the neighbor to the open set
                    heapq.heappush(open_set, (f_score, counter, neighbor))
                    counter += 1
        except Exception as e:
            print(f"Error in A* algorithm: {e}")
            return None, "error"
        
        return None, "no_solution"  # No solution found
    
    def solve_bfs(self):
        """Solve the maze using Breadth-First Search."""
        self.start_time = time.time()
        queue = deque([MazeState(self.start_pos)])
        visited = set([hash(MazeState(self.start_pos))])
        states_explored = 0  # Counter for states explored
        self.explored_cells = set()  # Reset explored cells
        
        try:
            while queue:
                # Check for timeout
                if time.time() - self.start_time > self.timeout:
                    return None, "timeout"  # Timeout reached
                
                # Check if we've reached the state limit
                if states_explored >= self.max_states:
                    return None, "state_limit"  # State limit reached
                
                current_state = queue.popleft()
                states_explored += 1  # Increment state counter
                self.explored_cells.add(current_state.position)  # Track explored cell
                
                # Check if we've reached the goal
                if current_state.position == self.goal_pos:
                    return self.reconstruct_path(current_state), None
                
                # Explore neighbors
                for neighbor in current_state.get_neighbors(self.maze):
                    if hash(neighbor) not in visited:
                        visited.add(hash(neighbor))
                        queue.append(neighbor)
        except Exception as e:
            print(f"Error in BFS algorithm: {e}")
            return None, "error"
        
        return None, "no_solution"  # No solution found
    
    def solve_dfs(self, max_depth=100):
        """Solve the maze using Depth-First Search with a depth limit."""
        self.start_time = time.time()
        states_explored = 0  # Counter for states explored
        self.explored_cells = set()  # Reset explored cells
        
        stack = [MazeState(self.start_pos)]
        visited = set([hash(MazeState(self.start_pos))])
        
        try:
            while stack:
                # Check for timeout
                if time.time() - self.start_time > self.timeout:
                    return None, "timeout"  # Timeout reached
                
                # Check if we've reached the state limit
                if states_explored >= self.max_states:
                    return None, "state_limit"  # State limit reached
                
                current_state = stack.pop()
                states_explored += 1  # Increment state counter
                self.explored_cells.add(current_state.position)  # Track explored cell
                
                # Check if we've reached the goal
                if current_state.position == self.goal_pos:
                    return self.reconstruct_path(current_state), None
                
                # Check depth limit
                if current_state.depth >= max_depth:
                    continue
                
                # Explore neighbors in reverse order (to prioritize up, left, down, right)
                for neighbor in reversed(current_state.get_neighbors(self.maze)):
                    if hash(neighbor) not in visited:
                        visited.add(hash(neighbor))
                        stack.append(neighbor)
        except Exception as e:
            print(f"Error in DFS algorithm: {e}")
            return None, "error"
        
        return None, "no_solution"  # No solution found
    
    def solve_greedy(self):
        """Solve the maze using Greedy Best-First Search."""
        self.start_time = time.time()
        open_set = []
        closed_set = set()
        states_explored = 0  # Counter for states explored
        self.explored_cells = set()  # Reset explored cells
        
        # Create initial state
        initial_state = MazeState(self.start_pos)
        
        # Add the initial state to the open set
        heapq.heappush(open_set, (self.manhattan_distance(initial_state), 0, initial_state))
        counter = 1  # Tie-breaker for states with the same priority
        
        try:
            while open_set:
                # Check for timeout
                if time.time() - self.start_time > self.timeout:
                    return None, "timeout"  # Timeout reached
                
                # Check if we've reached the state limit
                if states_explored >= self.max_states:
                    return None, "state_limit"  # State limit reached
                
                # Get the state with the lowest heuristic value
                _, _, current_state = heapq.heappop(open_set)
                states_explored += 1  # Increment state counter
                self.explored_cells.add(current_state.position)  # Track explored cell
                self.explored_cells.add(current_state.position)  # Track explored cell
                
                # Check if we've reached the goal
                if current_state.position == self.goal_pos:
                    return self.reconstruct_path(current_state), None
                
                # Add the current state to the closed set
                closed_set.add(hash(current_state))
                
                # Explore neighbors
                for neighbor in current_state.get_neighbors(self.maze):
                    if hash(neighbor) in closed_set:
                        continue
                    
                    # Calculate the heuristic value
                    h_score = self.manhattan_distance(neighbor)
                    
                    # Add the neighbor to the open set
                    heapq.heappush(open_set, (h_score, counter, neighbor))
                    counter += 1
        except Exception as e:
            print(f"Error in Greedy algorithm: {e}")
            return None, "error"
        
        return None, "no_solution"  # No solution found
    
    def solve_uniform_cost(self):
        """Solve the maze using Uniform Cost Search."""
        self.start_time = time.time()
        open_set = []
        closed_set = set()
        states_explored = 0  # Counter for states explored
        self.explored_cells = set()  # Reset explored cells
        
        # Create initial state
        initial_state = MazeState(self.start_pos)
        
        # Add the initial state to the open set
        heapq.heappush(open_set, (0, 0, initial_state))  # (cost, counter, state)
        counter = 1  # Tie-breaker for states with the same priority
        
        try:
            while open_set:
                # Check for timeout
                if time.time() - self.start_time > self.timeout:
                    return None, "timeout"  # Timeout reached
                
                # Check if we've reached the state limit
                if states_explored >= self.max_states:
                    return None, "state_limit"  # State limit reached
                
                # Get the state with the lowest cost
                cost, _, current_state = heapq.heappop(open_set)
                states_explored += 1  # Increment state counter
                self.explored_cells.add(current_state.position)  # Track explored cell
                
                # Check if we've reached the goal
                if current_state.position == self.goal_pos:
                    return self.reconstruct_path(current_state), None
                
                # Add the current state to the closed set
                closed_set.add(hash(current_state))
                
                # Explore neighbors
                for neighbor in current_state.get_neighbors(self.maze):
                    if hash(neighbor) in closed_set:
                        continue
                    
                    # Calculate the new cost
                    new_cost = cost + 1
                    
                    # Add the neighbor to the open set
                    heapq.heappush(open_set, (new_cost, counter, neighbor))
                    counter += 1
        except Exception as e:
            print(f"Error in Uniform Cost algorithm: {e}")
            return None, "error"
        
        return None, "no_solution"  # No solution found
    
    def get_explored_cells(self):
        """Return the set of cells that were explored during the search."""
        return self.explored_cells

    def reconstruct_path(self, state):
        """Reconstruct the path from the initial state to the goal state."""
        path = []
        current = state
        
        while current.parent:
            path.append(current.move)
            current = current.parent
        
        path.reverse()  # Reverse to get the path from start to goal
        return path