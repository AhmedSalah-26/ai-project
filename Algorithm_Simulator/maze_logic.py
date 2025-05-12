import random
import time
import heapq
from collections import deque
import copy
import numpy as np

class MazeState:
    """Represents a state in the maze."""
    def __init__(self, position, parent=None, move=None, depth=0):
        self.position = position  
        self.parent = parent
        self.move = move  
        self.depth = depth  
    
    def get_neighbors(self, maze):
        """Generate all possible next states by moving in valid directions."""
        neighbors = []
        moves = [(0, 1, 'right'), (1, 0, 'down'), (0, -1, 'left'), (-1, 0, 'up')]
        
        for dx, dy, direction in moves:
            new_x, new_y = self.position[0] + dx, self.position[1] + dy
            
            
            if (0 <= new_x < len(maze) and 0 <= new_y < len(maze[0]) and 
                maze[new_x][new_y] != 1): 
                
                
                move_description = f"Move {direction}"
                neighbors.append(MazeState((new_x, new_y), self, move_description, self.depth + 1))
        
        return neighbors
    
    def __eq__(self, other):
        return self.position == other.position
    
    def __hash__(self):
        
        return hash((self.position[0], self.position[1]))
    
    def __lt__(self, other):
        
        return self.depth < other.depth

class MazeGenerator:
    """Generates random mazes using various algorithms."""
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
    
    def generate_dfs_maze(self):
        """Generate a maze using Depth-First Search algorithm."""
        # Initialize maze 
        maze = [[1 for _ in range(self.cols)] for _ in range(self.rows)]
        
        # Choose random starting point 
        start_row = random.randrange(1, self.rows, 2)
        start_col = random.randrange(1, self.cols, 2)
        

        maze[start_row][start_col] = 0
        

        stack = [(start_row, start_col)]
        
        # Directions: right, down, left, up
        directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
        
        while stack:
            current_row, current_col = stack[-1]
            
            # For unvisited neighbors
            unvisited = []
            random.shuffle(directions)
            
            for dr, dc in directions:
                new_row, new_col = current_row + dr, current_col + dc
                
                if (0 <= new_row < self.rows and 0 <= new_col < self.cols and 
                    maze[new_row][new_col] == 1):
                    unvisited.append((new_row, new_col, dr, dc))
            
            if unvisited:

                new_row, new_col, dr, dc = unvisited[0]
                
                maze[current_row + dr//2][current_col + dc//2] = 0
                
                # Mark as visited
                maze[new_row][new_col] = 0
                
                stack.append((new_row, new_col))
            else:
                # 
                stack.pop()
        
        maze[0][0] = 0  
        
        maze[self.rows-1][self.cols-1] = 0  
        
        # Connect entrance to first row path
        if maze[1][0] == 1: 
            maze[1][0] = 0   
        
        # Connect exit to last row path
        if maze[self.rows-2][self.cols-1] == 1: 
            maze[self.rows-2][self.cols-1] = 0   
        
        return maze
    
    def generate_random_maze(self):
        """Generate a simple random maze."""
        maze = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        
        # Add random walls
        wall_density = 0.3  
        
        for row in range(1, self.rows-1):
            for col in range(1, self.cols-1):
                if random.random() < wall_density:
                    maze[row][col] = 1  
        

        maze[0][0] = 0  # Start
        maze[self.rows-1][self.cols-1] = 0  # End
        
        
        return maze

class MazeSolver:
    """Contains algorithms to solve mazes."""
    def __init__(self, maze, start_pos, goal_pos):
        self.maze = maze
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.timeout = 30  
        self.start_time = None
        self.max_states = 100000 
        self.q_table = {}  
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 0.1 
    
    def manhattan_distance(self, state):
        """Calculate the Manhattan distance heuristic."""
        return (abs(state.position[0] - self.goal_pos[0]) + 
                abs(state.position[1] - self.goal_pos[1]))
    
    def solve_a_star(self):
        """Solve the maze using A* algorithm."""
        self.start_time = time.time()
        open_set = []
        closed_set = set()
        states_explored = 0  
        self.explored_cells = set() 
        
        # Create initial state
        initial_state = MazeState(self.start_pos)
        
        h_score = self.manhattan_distance(initial_state)
        heapq.heappush(open_set, (h_score, 0, initial_state))
        counter = 1 
        
        try:
            while open_set:
                if time.time() - self.start_time > self.timeout:
                    return None, "timeout"  
                
                if states_explored >= self.max_states:
                    return None, "state_limit"  
                

                f_score, _, current_state = heapq.heappop(open_set)
                states_explored += 1  
                
                # Skip if already in closed set
                if hash(current_state) in closed_set:
                    continue
                
                self.explored_cells.add(current_state.position)  
                

                if current_state.position == self.goal_pos:
                    return self.reconstruct_path(current_state), None
                

                closed_set.add(hash(current_state))
                
                # Explore neighbors
                for neighbor in current_state.get_neighbors(self.maze):
                    if hash(neighbor) in closed_set:
                        continue
                    

                    g_score = current_state.depth + 1  
                    h_score = self.manhattan_distance(neighbor)
                    f_score = g_score + h_score
                    
                    neighbor.depth = g_score
                    

                    heapq.heappush(open_set, (f_score, counter, neighbor))
                    counter += 1
        except Exception as e:
            print(f"Error in A* algorithm: {e}")
            return None, "error"
        
        return None, "no_solution" 
    
    def solve_bfs(self):
        """Solve the maze using Breadth-First Search."""
        self.start_time = time.time()
        queue = deque([MazeState(self.start_pos)])
        visited = set([hash(MazeState(self.start_pos))])
        states_explored = 0  
        self.explored_cells = set()  
        
        try:
            while queue:

                if time.time() - self.start_time > self.timeout:
                    return None, "timeout"  
                
                if states_explored >= self.max_states:
                    return None, "state_limit"  
                
                current_state = queue.popleft()
                states_explored += 1  
                self.explored_cells.add(current_state.position)  
                
                if current_state.position == self.goal_pos:
                    return self.reconstruct_path(current_state), None
                
                for neighbor in current_state.get_neighbors(self.maze):
                    if hash(neighbor) not in visited:
                        visited.add(hash(neighbor))
                        queue.append(neighbor)
        except Exception as e:
            print(f"Error in BFS algorithm: {e}")
            return None, "error"
        
        return None, "no_solution"  
    
    def solve_dfs(self, max_depth=100):
        """Solve the maze using Depth-First Search with a depth limit."""
        self.start_time = time.time()
        states_explored = 0  
        self.explored_cells = set()  
        
        stack = [MazeState(self.start_pos)]
        visited = set([hash(MazeState(self.start_pos))])
        
        try:
            while stack:
                if time.time() - self.start_time > self.timeout:
                    return None, "timeout"  
                
                if states_explored >= self.max_states:
                    return None, "state_limit"  
                
                current_state = stack.pop()
                states_explored += 1  
                self.explored_cells.add(current_state.position) 
                
                if current_state.position == self.goal_pos:
                    return self.reconstruct_path(current_state), None
                

                if current_state.depth >= max_depth:
                    continue
                

                for neighbor in reversed(current_state.get_neighbors(self.maze)):
                    if hash(neighbor) not in visited:
                        visited.add(hash(neighbor))
                        stack.append(neighbor)
        except Exception as e:
            print(f"Error in DFS algorithm: {e}")
            return None, "error"
        
        return None, "no_solution"  
    
    def solve_greedy(self):
        """Solve the maze using Greedy Best-First Search."""
        self.start_time = time.time()
        open_set = []
        closed_set = set()
        states_explored = 0 
        self.explored_cells = set()  
        
        initial_state = MazeState(self.start_pos)
        
        heapq.heappush(open_set, (self.manhattan_distance(initial_state), 0, initial_state))
        counter = 1  
        
        try:
            while open_set:
                if time.time() - self.start_time > self.timeout:
                    return None, "timeout"  
                
                if states_explored >= self.max_states:
                    return None, "state_limit"  
                
                _, _, current_state = heapq.heappop(open_set)
                states_explored += 1  
                self.explored_cells.add(current_state.position) 
                self.explored_cells.add(current_state.position)  
                
                # Check if we've reached the goal
                if current_state.position == self.goal_pos:
                    return self.reconstruct_path(current_state), None
                
                # Add the current state to the closed set
                closed_set.add(hash(current_state))
                

                for neighbor in current_state.get_neighbors(self.maze):
                    if hash(neighbor) in closed_set:
                        continue
                    

                    h_score = self.manhattan_distance(neighbor)
                    
                    heapq.heappush(open_set, (h_score, counter, neighbor))
                    counter += 1
        except Exception as e:
            print(f"Error in Greedy algorithm: {e}")
            return None, "error"
        
        return None, "no_solution"  
    
    def solve_uniform_cost(self):
        """Solve the maze using Uniform Cost Search."""
        self.start_time = time.time()
        open_set = []
        closed_set = set()
        states_explored = 0  
        self.explored_cells = set()  
        
        initial_state = MazeState(self.start_pos)
        
        heapq.heappush(open_set, (0, 0, initial_state))  # (cost, counter, state)
        counter = 1  
        
        try:
            while open_set:
                # Check for timeout
                if time.time() - self.start_time > self.timeout:
                    return None, "timeout"  
                
                if states_explored >= self.max_states:
                    return None, "state_limit"  
                
                cost, _, current_state = heapq.heappop(open_set)
                states_explored += 1  
                self.explored_cells.add(current_state.position)  
                
                if current_state.position == self.goal_pos:
                    return self.reconstruct_path(current_state), None
                
                closed_set.add(hash(current_state))
                
                # Explore neighbors
                for neighbor in current_state.get_neighbors(self.maze):
                    if hash(neighbor) in closed_set:
                        continue
                    
                    new_cost = cost + 1
                    
                    heapq.heappush(open_set, (new_cost, counter, neighbor))
                    counter += 1
        except Exception as e:
            print(f"Error in Uniform Cost algorithm: {e}")
            return None, "error"
        
        return None, "no_solution"  
    
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
        
        path.reverse()  
        return path

    def get_state_key(self, position):
        """Convert position to a hashable key for Q-table."""
        return (position[0], position[1])

    def get_action_reward(self, next_pos):
        """Calculate reward for taking an action."""
        if next_pos == self.goal_pos:
            return 100  
        elif self.maze[next_pos[0]][next_pos[1]] == 1:
            return -100 
        else:
            return -1 

    def get_valid_actions(self, position):
        """Get all valid actions from current position."""
        actions = []
        moves = [(0, 1, 'right'), (1, 0, 'down'), (0, -1, 'left'), (-1, 0, 'up')]
        
        for dx, dy, direction in moves:
            new_x, new_y = position[0] + dx, position[1] + dy
            if (0 <= new_x < len(self.maze) and 0 <= new_y < len(self.maze[0]) and 
                self.maze[new_x][new_y] != 1):
                actions.append((new_x, new_y))
        return actions

    def solve_q_learning(self, episodes=1000):
        """Solve the maze using Q-learning algorithm."""
        self.start_time = time.time()
        self.explored_cells = set()
        
        # Initialize Q-table
        for row in range(len(self.maze)):
            for col in range(len(self.maze[0])):
                if self.maze[row][col] != 1:  
                    state = self.get_state_key((row, col))
                    self.q_table[state] = {}
                    for action in self.get_valid_actions((row, col)):
                        self.q_table[state][action] = 0.0

        try:
            for episode in range(episodes):
                if time.time() - self.start_time > self.timeout:
                    return None, "timeout"

                current_pos = self.start_pos
                path = [current_pos]
                
                while current_pos != self.goal_pos:
                    state = self.get_state_key(current_pos)
                    valid_actions = self.get_valid_actions(current_pos)
                    

                    if random.random() < self.epsilon:
                        next_pos = random.choice(valid_actions)
                    else:
                        next_pos = max(valid_actions, key=lambda x: self.q_table[state].get(x, 0))
                    
                    reward = self.get_action_reward(next_pos)
                    
                    # Update Q-value
                    next_state = self.get_state_key(next_pos)
                    next_max = max(self.q_table[next_state].values()) if self.q_table[next_state] else 0
                    current_q = self.q_table[state].get(next_pos, 0)
                    new_q = current_q + self.learning_rate * (reward + self.discount_factor * next_max - current_q)
                    self.q_table[state][next_pos] = new_q
                    
                    current_pos = next_pos
                    path.append(current_pos)
                    self.explored_cells.add(current_pos)
                    
                    if len(path) > len(self.maze) * len(self.maze[0]):  
                        break

            # Extract final path using learned Q-values
            current_pos = self.start_pos
            final_path = []
            
            while current_pos != self.goal_pos:
                state = self.get_state_key(current_pos)
                next_pos = max(self.get_valid_actions(current_pos), 
                            key=lambda x: self.q_table[state].get(x, 0))
                
                # Determine the direction of movement
                dx = next_pos[0] - current_pos[0]
                dy = next_pos[1] - current_pos[1]
                
                if dx == 0 and dy == 1:
                    move = "Move right"
                elif dx == 1 and dy == 0:
                    move = "Move down"
                elif dx == 0 and dy == -1:
                    move = "Move left"
                elif dx == -1 and dy == 0:
                    move = "Move up"
                
                final_path.append(move)
                current_pos = next_pos
                
                if len(final_path) > len(self.maze) * len(self.maze[0]):  # Prevent infinite loops
                    return None, "no_solution"

            return final_path, None

        except Exception as e:
            print(f"Error in Q-learning algorithm: {e}")
            return None, "error"