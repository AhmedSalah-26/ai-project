import random
import time
import heapq
from collections import deque
import copy
import operator

class PuzzleState:
    """Represents a state of the sliding puzzle."""
    def __init__(self, board, empty_pos=None, parent=None, move=None, depth=0):
        self.board = board
        self.size = len(board)
        if empty_pos:
            self.empty_pos = empty_pos
        else:
            # Find the empty position (represented by 0)
            for i in range(self.size):
                for j in range(self.size):
                    if board[i][j] == 0:
                        self.empty_pos = (i, j)
                        break
        self.parent = parent
        self.move = move  # The move that led to this state
        self.depth = depth  # Depth in the search tree
    
    def get_neighbors(self):
        """Generate all possible next states by moving the empty space."""
        neighbors = []
        moves = [(0, 1, 'right'), (1, 0, 'down'), (0, -1, 'left'), (-1, 0, 'up')]
        
        for dx, dy, direction in moves:
            new_x, new_y = self.empty_pos[0] + dx, self.empty_pos[1] + dy
            
            # Check if the new position is valid
            if 0 <= new_x < self.size and 0 <= new_y < self.size:
                # For 4x4 puzzles, use a more memory-efficient approach
                if self.size == 4:
                    # Create a new board by copying rows individually
                    new_board = []
                    for i, row in enumerate(self.board):
                        if i == self.empty_pos[0] or i == new_x:
                            # Only deep copy rows that will change
                            new_board.append(list(row))
                        else:
                            # For unchanged rows, just reference the original
                            new_board.append(row)
                else:
                    # For smaller puzzles, use the original method
                    new_board = copy.deepcopy(self.board)
                
                tile_value = new_board[new_x][new_y]
                
                # Swap the empty space with the tile
                new_board[self.empty_pos[0]][self.empty_pos[1]] = tile_value
                new_board[new_x][new_y] = 0
                
                # Create a new state with the moved tile
                move_description = f"Move {tile_value} {direction}"
                neighbors.append(PuzzleState(new_board, (new_x, new_y), self, move_description, self.depth + 1))
        
        return neighbors
    
    def __eq__(self, other):
        return self.board == other.board
    
    def __hash__(self):
        # More efficient hashing for larger boards
        # For 4x4 puzzles, use a more compact representation to save memory
        if self.size == 4:
            # Convert the 2D board to a single tuple for more efficient hashing
            flat_board = tuple(item for row in self.board for item in row)
            return hash(flat_board)
        else:
            # For smaller boards, use the original method
            return hash(tuple(tuple(row) for row in self.board))
    
    def __lt__(self, other):
        # For priority queue in A* algorithm
        return self.depth < other.depth

class PuzzleSolver:
    """Contains algorithms to solve the sliding puzzle."""
    def __init__(self, initial_state, goal_state):
        self.initial_state = initial_state
        self.goal_state = goal_state
        self.timeout = 30  # Default timeout in seconds
        self.start_time = None
        
        # Set state limits based on puzzle size
        if initial_state.size == 3:
            self.max_states = 500000  # 3x3 puzzles can explore more states
        else:
            self.max_states = 200000  # 4x4 puzzles need stricter limits
    
    def manhattan_distance(self, state):
        """Calculate the Manhattan distance heuristic."""
        distance = 0
        size = state.size
        
        for i in range(size):
            for j in range(size):
                if state.board[i][j] != 0:  # Skip the empty space
                    # Find the goal position of the current tile
                    value = state.board[i][j]
                    goal_i, goal_j = divmod(value - 1, size)
                    
                    # Add the Manhattan distance
                    distance += abs(i - goal_i) + abs(j - goal_j)
        
        return distance
    
    def solve_a_star(self):
        """Solve the puzzle using A* algorithm."""
        self.start_time = time.time()
        open_set = []
        closed_set = set()
        states_explored = 0  # Counter for states explored
        
        # Add the initial state to the open set
        heapq.heappush(open_set, (self.manhattan_distance(self.initial_state), 0, self.initial_state))
        counter = 1  # Tie-breaker for states with the same priority
        
        # Memory management - limit the size of open_set for 4x4 puzzles
        max_open_set_size = 100000 if self.initial_state.size == 4 else float('inf')
        
        try:
            while open_set:
                # Check for timeout
                if time.time() - self.start_time > self.timeout:
                    return None, "timeout"  # Timeout reached
                
                # Check if we've reached the state limit
                if states_explored >= self.max_states:
                    return None, "state_limit"  # State limit reached
                
                # Get the state with the lowest f-score
                _, _, current_state = heapq.heappop(open_set)
                states_explored += 1  # Increment state counter
                
                # Check if we've reached the goal
                if current_state.board == self.goal_state.board:
                    return self.reconstruct_path(current_state), None
                
                # Add the current state to the closed set
                closed_set.add(hash(current_state))
                
                # Explore neighbors
                for neighbor in current_state.get_neighbors():
                    if hash(neighbor) in closed_set:
                        continue
                    
                    # Calculate the f-score (g + h)
                    g_score = neighbor.depth
                    h_score = self.manhattan_distance(neighbor)
                    f_score = g_score + h_score
                    
                    # Add the neighbor to the open set if we haven't reached the limit
                    if len(open_set) < max_open_set_size:
                        heapq.heappush(open_set, (f_score, counter, neighbor))
                        counter += 1
                    else:
                        # If we've reached the limit, only add if better than worst in open_set
                        if open_set and f_score < open_set[0][0]:
                            heapq.heappop(open_set)  # Remove the worst state
                            heapq.heappush(open_set, (f_score, counter, neighbor))
                            counter += 1
        except MemoryError:
            return None, "memory_error"  # Out of memory
        except Exception as e:
            print(f"Error in A* algorithm: {e}")
            return None, "error"
        
        return None, "no_solution"  # No solution found
    
    def solve_bfs(self):
        """Solve the puzzle using Breadth-First Search."""
        self.start_time = time.time()
        queue = deque([self.initial_state])
        visited = set([hash(self.initial_state)])
        states_explored = 0  # Counter for states explored
        
        # Memory management - limit the size of queue for 4x4 puzzles
        max_queue_size = 100000 if self.initial_state.size == 4 else float('inf')
        
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
                
                # Check if we've reached the goal
                if current_state.board == self.goal_state.board:
                    return self.reconstruct_path(current_state), None
                
                # Explore neighbors
                for neighbor in current_state.get_neighbors():
                    if hash(neighbor) not in visited:
                        visited.add(hash(neighbor))
                        if len(queue) < max_queue_size:
                            queue.append(neighbor)
        except MemoryError:
            return None, "memory_error"  # Out of memory
        except Exception as e:
            print(f"Error in BFS algorithm: {e}")
            return None, "error"
        
        return None, "no_solution"  # No solution found
    
    def solve_dfs(self, max_depth=100):
        """Solve the puzzle using Depth-First Search with a depth limit."""
        self.start_time = time.time()
        states_explored = 0  # Counter for states explored
        
        # Adjust max_depth based on puzzle size
        if self.initial_state.size == 4:
            max_depth = 50  # Reduce depth for 4x4 puzzles to avoid excessive memory usage
        
        stack = [self.initial_state]
        visited = set([hash(self.initial_state)])
        
        # Memory management - limit the size of stack for 4x4 puzzles
        max_stack_size = 50000 if self.initial_state.size == 4 else float('inf')
        
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
                
                # Check if we've reached the goal
                if current_state.board == self.goal_state.board:
                    return self.reconstruct_path(current_state), None
                
                # Check depth limit
                if current_state.depth >= max_depth:
                    continue
                
                # Explore neighbors in reverse order (to prioritize up, left, down, right)
                for neighbor in reversed(current_state.get_neighbors()):
                    if hash(neighbor) not in visited:
                        visited.add(hash(neighbor))
                        if len(stack) < max_stack_size:
                            stack.append(neighbor)
        except MemoryError:
            return None, "memory_error"  # Out of memory
        except Exception as e:
            print(f"Error in DFS algorithm: {e}")
            return None, "error"
        
        return None, "no_solution"  # No solution found
    
    def solve_greedy(self):
        """Solve the puzzle using Greedy Best-First Search."""
        self.start_time = time.time()
        open_set = []
        closed_set = set()
        states_explored = 0  # Counter for states explored
        
        # Add the initial state to the open set
        heapq.heappush(open_set, (self.manhattan_distance(self.initial_state), 0, self.initial_state))
        counter = 1  # Tie-breaker for states with the same priority
        
        # Memory management - limit the size of open_set for 4x4 puzzles
        max_open_set_size = 100000 if self.initial_state.size == 4 else float('inf')
        
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
                
                # Check if we've reached the goal
                if current_state.board == self.goal_state.board:
                    return self.reconstruct_path(current_state), None
                
                # Add the current state to the closed set
                closed_set.add(hash(current_state))
                
                # Explore neighbors
                for neighbor in current_state.get_neighbors():
                    if hash(neighbor) in closed_set:
                        continue
                    
                    # Calculate the heuristic value
                    h_score = self.manhattan_distance(neighbor)
                    
                    # Add the neighbor to the open set if we haven't reached the limit
                    if len(open_set) < max_open_set_size:
                        heapq.heappush(open_set, (h_score, counter, neighbor))
                        counter += 1
                    else:
                        # If we've reached the limit, only add if better than worst in open_set
                        if open_set and h_score < open_set[0][0]:
                            heapq.heappop(open_set)  # Remove the worst state
                            heapq.heappush(open_set, (h_score, counter, neighbor))
                            counter += 1
        except MemoryError:
            return None, "memory_error"  # Out of memory
        except Exception as e:
            print(f"Error in Greedy algorithm: {e}")
            return None, "error"
        
        return None, "no_solution"  # No solution found
    
    def solve_uniform_cost(self):
        """Solve the puzzle using Uniform Cost Search."""
        self.start_time = time.time()
        open_set = []
        closed_set = set()
        states_explored = 0  # Counter for states explored
        
        # Add the initial state to the open set
        heapq.heappush(open_set, (0, 0, self.initial_state))  # (cost, counter, state)
        counter = 1  # Tie-breaker for states with the same priority
        
        # Memory management - limit the size of open_set for 4x4 puzzles
        max_open_set_size = 100000 if self.initial_state.size == 4 else float('inf')
        
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
                
                # Check if we've reached the goal
                if current_state.board == self.goal_state.board:
                    return self.reconstruct_path(current_state), None
                
                # Add the current state to the closed set
                closed_set.add(hash(current_state))
                
                # Explore neighbors
                for neighbor in current_state.get_neighbors():
                    if hash(neighbor) in closed_set:
                        continue
                    
                    # Calculate the new cost
                    new_cost = cost + 1
                    
                    # Add the neighbor to the open set if we haven't reached the limit
                    if len(open_set) < max_open_set_size:
                        heapq.heappush(open_set, (new_cost, counter, neighbor))
                        counter += 1
                    else:
                        # If we've reached the limit, only add if better than worst in open_set
                        if open_set and new_cost < open_set[0][0]:
                            heapq.heappop(open_set)  # Remove the worst state
                            heapq.heappush(open_set, (new_cost, counter, neighbor))
                            counter += 1
        except MemoryError:
            return None, "memory_error"  # Out of memory
        except Exception as e:
            print(f"Error in Uniform Cost algorithm: {e}")
            return None, "error"
        
        return None, "no_solution"  # No solution found
    
    def reconstruct_path(self, state):
        """Reconstruct the path from the initial state to the goal state."""
        path = []
        current = state
        
        while current.parent:
            path.append(current.move)
            current = current.parent
        
        path.reverse()  # Reverse to get the path from start to goal
        return path
    
    def solve_genetic(self, population_size=100, generations=50):
        """Solve the puzzle using a genetic algorithm approach."""
        self.start_time = time.time()
        states_explored = 0
        
        # Adjust parameters based on puzzle size
        if self.initial_state.size == 4:
            population_size = 200  # Larger population for 4x4 puzzles
            generations = 100      # More generations for 4x4 puzzles
        
        try:
            # Initialize population with random moves from initial state
            population = self.initialize_population(population_size)
            best_fitness = float('inf')
            best_state = None
            
            for generation in range(generations):
                # Check for timeout
                if time.time() - self.start_time > self.timeout:
                    return None, "timeout"  # Timeout reached
                
                # Check if we've reached the state limit
                if states_explored >= self.max_states:
                    return None, "state_limit"  # State limit reached
                
                # Evaluate fitness of each individual
                fitness_scores = []
                for individual in population:
                    fitness = self.calculate_fitness(individual)
                    fitness_scores.append((fitness, individual))
                    states_explored += 1
                    
                    # Check if we've found the goal
                    if fitness == 0:
                        return self.reconstruct_path(individual), None
                
                # Sort by fitness (lower is better)
                fitness_scores.sort(key=operator.itemgetter(0))
                
                # Keep track of the best solution found so far
                if fitness_scores[0][0] < best_fitness:
                    best_fitness = fitness_scores[0][0]
                    best_state = fitness_scores[0][1]
                    
                    # If we're very close to the solution, try to complete it with A*
                    if best_fitness <= 5 and self.initial_state.size == 3:
                        # Create a new solver with the current best state as initial
                        sub_solver = PuzzleSolver(best_state, self.goal_state)
                        sub_solver.timeout = 5  # Short timeout for the A* refinement
                        sub_path, sub_reason = sub_solver.solve_a_star()
                        
                        if sub_path:
                            # Combine paths
                            path = self.reconstruct_path(best_state)
                            return path + sub_path, None
                
                # Create the next generation
                next_population = []
                
                # Elitism: Keep the best individuals
                elite_count = max(1, population_size // 10)
                for i in range(elite_count):
                    next_population.append(fitness_scores[i][1])
                
                # Fill the rest of the population with offspring
                while len(next_population) < population_size:
                    # Tournament selection
                    parent1 = self.tournament_selection(fitness_scores, tournament_size=3)
                    parent2 = self.tournament_selection(fitness_scores, tournament_size=3)
                    
                    # Crossover
                    if random.random() < 0.8:  # 80% chance of crossover
                        child = self.crossover(parent1, parent2)
                    else:
                        child = parent1 if random.random() < 0.5 else parent2
                    
                    # Mutation
                    if random.random() < 0.2:  # 20% chance of mutation
                        child = self.mutate(child)
                    
                    next_population.append(child)
                
                population = next_population
            
            # If we've reached the maximum generations without finding the exact solution,
            # return the best solution found so far
            if best_state:
                return self.reconstruct_path(best_state), "partial_solution"
            
        except MemoryError:
            return None, "memory_error"  # Out of memory
        except Exception as e:
            print(f"Error in Genetic algorithm: {e}")
            return None, "error"
        
        return None, "no_solution"  # No solution found
    
    def initialize_population(self, population_size):
        """Initialize a population of puzzle states with random moves."""
        population = []
        for _ in range(population_size):
            # Start with the initial state
            current_state = copy.deepcopy(self.initial_state)
            
            # Apply a random number of random moves
            num_moves = random.randint(5, 20)
            for _ in range(num_moves):
                neighbors = current_state.get_neighbors()
                if neighbors:
                    current_state = random.choice(neighbors)
            
            population.append(current_state)
        
        return population
    
    def calculate_fitness(self, state):
        """Calculate the fitness of a state (lower is better)."""
        # Use Manhattan distance as the primary fitness measure
        manhattan = self.manhattan_distance(state)
        
        # Add a small penalty for the depth to encourage shorter paths
        depth_penalty = state.depth * 0.1
        
        return manhattan + depth_penalty
    
    def tournament_selection(self, fitness_scores, tournament_size):
        """Select an individual using tournament selection."""
        # Randomly select tournament_size individuals
        tournament = random.sample(fitness_scores, tournament_size)
        
        # Return the best individual from the tournament
        tournament.sort(key=operator.itemgetter(0))
        return tournament[0][1]
    
    def crossover(self, parent1, parent2):
        """Create a new individual by combining two parents."""
        # Start with a copy of the initial state
        child_state = copy.deepcopy(self.initial_state)
        
        # Get the move sequences that led to each parent
        parent1_path = self.reconstruct_path(parent1)
        parent2_path = self.reconstruct_path(parent2)
        
        # Choose a crossover point
        if parent1_path and parent2_path:
            crossover_point1 = random.randint(0, len(parent1_path))
            crossover_point2 = random.randint(0, len(parent2_path))
            
            # Create a new path by combining parts of both parents
            combined_path = parent1_path[:crossover_point1] + parent2_path[crossover_point2:]
            
            # Apply the combined path to the child state
            for move in combined_path:
                # Find the move direction and tile value
                if "up" in move.lower():
                    direction = "up"
                elif "down" in move.lower():
                    direction = "down"
                elif "left" in move.lower():
                    direction = "left"
                elif "right" in move.lower():
                    direction = "right"
                else:
                    continue  # Skip if we can't determine the direction
                
                # Try to apply the move
                neighbors = child_state.get_neighbors()
                for neighbor in neighbors:
                    if direction in neighbor.move.lower():
                        child_state = neighbor
                        break
        
        return child_state
    
    def mutate(self, state):
        """Apply a random mutation to the state."""
        # Apply a small number of random moves
        mutated_state = copy.deepcopy(state)
        num_mutations = random.randint(1, 3)
        
        for _ in range(num_mutations):
            neighbors = mutated_state.get_neighbors()
            if neighbors:
                mutated_state = random.choice(neighbors)
        
        return mutated_state