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
            for i in range(self.size):
                for j in range(self.size):
                    if board[i][j] == 0:
                        self.empty_pos = (i, j)
                        break
        self.parent = parent
        self.move = move  
        self.depth = depth  
    
    def get_neighbors(self):
        """Generate all possible next states by moving the empty space."""
        neighbors = []
        moves = [(0, 1, 'right'), (1, 0, 'down'), (0, -1, 'left'), (-1, 0, 'up')]
        
        for dx, dy, direction in moves:
            new_x, new_y = self.empty_pos[0] + dx, self.empty_pos[1] + dy
            
            # Check if the new position is valid
            if 0 <= new_x < self.size and 0 <= new_y < self.size:
                
                if self.size == 4:
                    new_board = []
                    for i, row in enumerate(self.board):
                        if i == self.empty_pos[0] or i == new_x:
                            new_board.append(list(row))
                        else:
                            new_board.append(row)
                else:
                    new_board = copy.deepcopy(self.board)
                
                tile_value = new_board[new_x][new_y]
                
                new_board[self.empty_pos[0]][self.empty_pos[1]] = tile_value
                new_board[new_x][new_y] = 0
                
                move_description = f"Move {tile_value} {direction}"
                neighbors.append(PuzzleState(new_board, (new_x, new_y), self, move_description, self.depth + 1))
        
        return neighbors
    
    def __eq__(self, other):
        return self.board == other.board
    
    def __hash__(self):

        if self.size == 4:
            flat_board = tuple(item for row in self.board for item in row)
            return hash(flat_board)
        else:
            return hash(tuple(tuple(row) for row in self.board))
    
    def __lt__(self, other):
        return self.depth < other.depth

class PuzzleSolver:
    """Contains algorithms to solve the sliding puzzle."""
    def __init__(self, initial_state, goal_state):
        self.initial_state = initial_state
        self.goal_state = goal_state
        self.timeout = 30 
        self.start_time = None
        

        if initial_state.size == 3:
            self.max_states = 500000  
        else:
            self.max_states = 200000  
            
        # Q-learning parameters
        self.q_table = {}  
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 0.1  
    
    def manhattan_distance(self, state):
        """Calculate the Manhattan distance heuristic."""
        distance = 0
        size = state.size
        
        for i in range(size):
            for j in range(size):
                if state.board[i][j] != 0: 
                    value = state.board[i][j]
                    goal_i, goal_j = divmod(value - 1, size)
                    
                    distance += abs(i - goal_i) + abs(j - goal_j)
        
        return distance
    
    def solve_a_star(self):
        """Solve the puzzle using A* algorithm."""
        self.start_time = time.time()
        open_set = []
        closed_set = set()
        states_explored = 0  
        
        heapq.heappush(open_set, (self.manhattan_distance(self.initial_state), 0, self.initial_state))
        counter = 1  
        
        max_open_set_size = 100000 if self.initial_state.size == 4 else float('inf')
        
        try:
            while open_set:
                if time.time() - self.start_time > self.timeout:
                    return None, "timeout"  
                
                if states_explored >= self.max_states:
                    return None, "state_limit"  
                
                _, _, current_state = heapq.heappop(open_set)
                states_explored += 1  
                
                # Check if we've reached the goal
                if current_state.board == self.goal_state.board:
                    return self.reconstruct_path(current_state), None
                
                # Add the current state to the closed set
                closed_set.add(hash(current_state))
                
                # Explore neighbors
                for neighbor in current_state.get_neighbors():
                    if hash(neighbor) in closed_set:
                        continue
                    
                    
                    g_score = neighbor.depth
                    h_score = self.manhattan_distance(neighbor)
                    f_score = g_score + h_score
                    
                    
                    if len(open_set) < max_open_set_size:
                        heapq.heappush(open_set, (f_score, counter, neighbor))
                        counter += 1
                    else:
                        if open_set and f_score < open_set[0][0]:
                            heapq.heappop(open_set)  # Remove the worst state
                            heapq.heappush(open_set, (f_score, counter, neighbor))
                            counter += 1
        except MemoryError:
            return None, "memory_error" 
        except Exception as e:
            print(f"Error in A* algorithm: {e}")
            return None, "error"
        
        return None, "no_solution"  
    
    def solve_bfs(self):
        """Solve the puzzle using Breadth-First Search."""
        self.start_time = time.time()
        queue = deque([self.initial_state])
        visited = set([hash(self.initial_state)])
        states_explored = 0  
        
        max_queue_size = 100000 if self.initial_state.size == 4 else float('inf')
        
        try:
            while queue:
                if time.time() - self.start_time > self.timeout:
                    return None, "timeout"  
                
                if states_explored >= self.max_states:
                    return None, "state_limit"  
                
                current_state = queue.popleft()
                states_explored += 1  
                
                if current_state.board == self.goal_state.board:
                    return self.reconstruct_path(current_state), None
                
                # Explore neighbors
                for neighbor in current_state.get_neighbors():
                    if hash(neighbor) not in visited:
                        visited.add(hash(neighbor))
                        if len(queue) < max_queue_size:
                            queue.append(neighbor)
        except MemoryError:
            return None, "memory_error"  
        except Exception as e:
            print(f"Error in BFS algorithm: {e}")
            return None, "error"
        
        return None, "no_solution" 
    
    def solve_dfs(self, max_depth=100):
        """Solve the puzzle using Depth-First Search with a depth limit."""
        self.start_time = time.time()
        states_explored = 0  
        
        
        if self.initial_state.size == 4:
            max_depth = 50  
        
        stack = [self.initial_state]
        visited = set([hash(self.initial_state)])
        
        max_stack_size = 50000 if self.initial_state.size == 4 else float('inf')
        
        try:
            while stack:
                if time.time() - self.start_time > self.timeout:
                    return None, "timeout"  
                
                if states_explored >= self.max_states:
                    return None, "state_limit"  
                
                current_state = stack.pop()
                states_explored += 1 
                
                if current_state.board == self.goal_state.board:
                    return self.reconstruct_path(current_state), None
                
                if current_state.depth >= max_depth:
                    continue
                
                for neighbor in reversed(current_state.get_neighbors()):
                    if hash(neighbor) not in visited:
                        visited.add(hash(neighbor))
                        if len(stack) < max_stack_size:
                            stack.append(neighbor)
        except MemoryError:
            return None, "memory_error"  
        except Exception as e:
            print(f"Error in DFS algorithm: {e}")
            return None, "error"
        
        return None, "no_solution"  
    
    def solve_greedy(self):
        """Solve the puzzle using Greedy Best-First Search."""
        self.start_time = time.time()
        open_set = []
        closed_set = set()
        states_explored = 0  
        
        # Add the initial state to the open set
        heapq.heappush(open_set, (self.manhattan_distance(self.initial_state), 0, self.initial_state))
        counter = 1 
        
        max_open_set_size = 100000 if self.initial_state.size == 4 else float('inf')
        
        try:
            while open_set:
                if time.time() - self.start_time > self.timeout:
                    return None, "timeout"  
                
                if states_explored >= self.max_states:
                    return None, "state_limit"  
                
                _, _, current_state = heapq.heappop(open_set)
                states_explored += 1 
                
                
                if current_state.board == self.goal_state.board:
                    return self.reconstruct_path(current_state), None
                
                closed_set.add(hash(current_state))
                
                for neighbor in current_state.get_neighbors():
                    if hash(neighbor) in closed_set:
                        continue
                    
                    h_score = self.manhattan_distance(neighbor)
                    
                    if len(open_set) < max_open_set_size:
                        heapq.heappush(open_set, (h_score, counter, neighbor))
                        counter += 1
                    else:
                        if open_set and h_score < open_set[0][0]:
                            heapq.heappop(open_set)  # Remove the worst state
                            heapq.heappush(open_set, (h_score, counter, neighbor))
                            counter += 1
        except MemoryError:
            return None, "memory_error" 
        except Exception as e:
            print(f"Error in Greedy algorithm: {e}")
            return None, "error"
        
        return None, "no_solution" 
    
    def solve_uniform_cost(self):
        """Solve the puzzle using Uniform Cost Search."""
        self.start_time = time.time()
        open_set = []
        closed_set = set()
        states_explored = 0 
        
        heapq.heappush(open_set, (0, 0, self.initial_state))  
        counter = 1 
        
        max_open_set_size = 100000 if self.initial_state.size == 4 else float('inf')
        
        try:
            while open_set:
                if time.time() - self.start_time > self.timeout:
                    return None, "timeout"  
                
                if states_explored >= self.max_states:
                    return None, "state_limit"  
                

                cost, _, current_state = heapq.heappop(open_set)
                states_explored += 1  
                
                if current_state.board == self.goal_state.board:
                    return self.reconstruct_path(current_state), None
                
                closed_set.add(hash(current_state))
                
                for neighbor in current_state.get_neighbors():
                    if hash(neighbor) in closed_set:
                        continue
                    
                    new_cost = cost + 1
                    
                    if len(open_set) < max_open_set_size:
                        heapq.heappush(open_set, (new_cost, counter, neighbor))
                        counter += 1
                    else:
                        if open_set and new_cost < open_set[0][0]:
                            heapq.heappop(open_set)  
                            heapq.heappush(open_set, (new_cost, counter, neighbor))
                            counter += 1
        except MemoryError:
            return None, "memory_error"  
        except Exception as e:
            print(f"Error in Uniform Cost algorithm: {e}")
            return None, "error"
        
        return None, "no_solution"  
    
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
        
        if self.initial_state.size == 4:
            population_size = 200  
            generations = 100     
        
        try:
            population = self.initialize_population(population_size)
            best_fitness = float('inf')
            best_state = None
            
            for generation in range(generations):
                if time.time() - self.start_time > self.timeout:
                    return None, "timeout"  
                
                if states_explored >= self.max_states:
                    return None, "state_limit"  
                
                fitness_scores = []
                for individual in population:
                    fitness = self.calculate_fitness(individual)
                    fitness_scores.append((fitness, individual))
                    states_explored += 1
                    
                    if fitness == 0:
                        return self.reconstruct_path(individual), None
                
                fitness_scores.sort(key=operator.itemgetter(0))
                
                if fitness_scores[0][0] < best_fitness:
                    best_fitness = fitness_scores[0][0]
                    best_state = fitness_scores[0][1]
                    
                    if best_fitness <= 5 and self.initial_state.size == 3:
                        # Create a new solver with the current best state as initial
                        sub_solver = PuzzleSolver(best_state, self.goal_state)
                        sub_solver.timeout = 5  
                        sub_path, sub_reason = sub_solver.solve_a_star()
                        
                        if sub_path:
                            path = self.reconstruct_path(best_state)
                            return path + sub_path, None
                
                # Create the next generation
                next_population = []
                
                elite_count = max(1, population_size // 10)
                for i in range(elite_count):
                    next_population.append(fitness_scores[i][1])
                
                while len(next_population) < population_size:
                    parent1 = self.tournament_selection(fitness_scores, tournament_size=3)
                    parent2 = self.tournament_selection(fitness_scores, tournament_size=3)
                    
                    if random.random() < 0.8:  
                        child = self.crossover(parent1, parent2)
                    else:
                        child = parent1 if random.random() < 0.5 else parent2
                    
                    # Mutation
                    if random.random() < 0.2:  
                        child = self.mutate(child)
                    
                    next_population.append(child)
                
                population = next_population
            
            # return the best solution found so far
            if best_state:
                return self.reconstruct_path(best_state), "partial_solution"
            
        except MemoryError:
            return None, "memory_error" 
        except Exception as e:
            print(f"Error in Genetic algorithm: {e}")
            return None, "error"
        
        return None, "no_solution"  
    
    def initialize_population(self, population_size):
        """Initialize a population of puzzle states with random moves."""
        population = []
        for _ in range(population_size):
            # Start with the initial state
            current_state = copy.deepcopy(self.initial_state)
            
            num_moves = random.randint(5, 20)
            for _ in range(num_moves):
                neighbors = current_state.get_neighbors()
                if neighbors:
                    current_state = random.choice(neighbors)
            
            population.append(current_state)
        
        return population
    
    def calculate_fitness(self, state):
        """Calculate the fitness of a state (lower is better)."""
        manhattan = self.manhattan_distance(state)
        
        depth_penalty = state.depth * 0.1
        
        return manhattan + depth_penalty
    
    def tournament_selection(self, fitness_scores, tournament_size):
        """Select an individual using tournament selection."""
        tournament = random.sample(fitness_scores, tournament_size)
        
        tournament.sort(key=operator.itemgetter(0))
        return tournament[0][1]
    
    def crossover(self, parent1, parent2):
        """Create a new individual by combining two parents."""
        child_state = copy.deepcopy(self.initial_state)
        
        parent1_path = self.reconstruct_path(parent1)
        parent2_path = self.reconstruct_path(parent2)
        
        # Choose a crossover point
        if parent1_path and parent2_path:
            crossover_point1 = random.randint(0, len(parent1_path))
            crossover_point2 = random.randint(0, len(parent2_path))
            
            combined_path = parent1_path[:crossover_point1] + parent2_path[crossover_point2:]
            
            for move in combined_path:
                if "up" in move.lower():
                    direction = "up"
                elif "down" in move.lower():
                    direction = "down"
                elif "left" in move.lower():
                    direction = "left"
                elif "right" in move.lower():
                    direction = "right"
                else:
                    continue 
                
                # To try apply the move
                neighbors = child_state.get_neighbors()
                for neighbor in neighbors:
                    if direction in neighbor.move.lower():
                        child_state = neighbor
                        break
        
        return child_state
    
    def mutate(self, state):
        """Apply a random mutation to the state."""
        mutated_state = copy.deepcopy(state)
        num_mutations = random.randint(1, 3)
        
        for _ in range(num_mutations):
            neighbors = mutated_state.get_neighbors()
            if neighbors:
                mutated_state = random.choice(neighbors)
        
        return mutated_state

    def get_state_key(self, state):
        """Convert state to a hashable key for Q-table."""
        if state.size == 4:
            return tuple(item for row in state.board for item in row)
        else:
            return tuple(tuple(row) for row in state.board)

    def get_action_reward(self, next_state):
        """Calculate reward for taking an action."""
        if next_state.board == self.goal_state.board:
            return 100  
        else:
            correct_tiles = sum(1 for i in range(next_state.size) 
                            for j in range(next_state.size)
                            if next_state.board[i][j] == self.goal_state.board[i][j])
            total_tiles = next_state.size * next_state.size
            return (correct_tiles / total_tiles) * 10  

    def get_valid_actions(self, state):
        """Get all valid actions from current state."""
        return state.get_neighbors()

    def solve_q_learning(self, episodes=1000):
        """Solve the puzzle using Q-learning algorithm."""
        self.start_time = time.time()
        
        # Initialize Q-table with the initial state
        initial_key = self.get_state_key(self.initial_state)
        self.q_table[initial_key] = {}
        for action in self.get_valid_actions(self.initial_state):
            self.q_table[initial_key][action] = 0.0

        try:
            for episode in range(episodes):
                if time.time() - self.start_time > self.timeout:
                    return None, "timeout"

                current_state = copy.deepcopy(self.initial_state)
                path = [current_state]
                visited_states = set([self.get_state_key(current_state)])
                
                while current_state.board != self.goal_state.board:
                    state_key = self.get_state_key(current_state)
                    valid_actions = self.get_valid_actions(current_state)
                    
                    # Initialize Q-values for new states
                    if state_key not in self.q_table:
                        self.q_table[state_key] = {}
                        for action in valid_actions:
                            self.q_table[state_key][action] = 0.0
                    
                    if random.random() < self.epsilon:
                        next_state = random.choice(valid_actions)
                    else:
                        unvisited_actions = [action for action in valid_actions 
                                        if self.get_state_key(action) not in visited_states]
                        if unvisited_actions:
                            next_state = max(unvisited_actions, 
                                        key=lambda x: self.q_table[state_key].get(x, 0))
                        else:
                            next_state = max(valid_actions, 
                                        key=lambda x: self.q_table[state_key].get(x, 0))
                    
                    reward = self.get_action_reward(next_state)
                    
                    # Update Q-value
                    next_key = self.get_state_key(next_state)
                    if next_key not in self.q_table:
                        self.q_table[next_key] = {}
                        for action in self.get_valid_actions(next_state):
                            self.q_table[next_key][action] = 0.0
                    
                    next_max = max(self.q_table[next_key].values()) if self.q_table[next_key] else 0
                    current_q = self.q_table[state_key].get(next_state, 0)
                    new_q = current_q + self.learning_rate * (reward + self.discount_factor * next_max - current_q)
                    self.q_table[state_key][next_state] = new_q
                    
                    current_state = next_state
                    path.append(current_state)
                    visited_states.add(self.get_state_key(current_state))
                    
                    if len(path) > self.max_states:  # Prevent infinite loops
                        break

            # Extract final path using learned Q-values
            current_state = copy.deepcopy(self.initial_state)
            final_path = []
            visited_states = set([self.get_state_key(current_state)])
            
            while current_state.board != self.goal_state.board:
                state_key = self.get_state_key(current_state)
                valid_actions = self.get_valid_actions(current_state)
                
                unvisited_actions = [action for action in valid_actions 
                                if self.get_state_key(action) not in visited_states]
                if unvisited_actions:
                    next_state = max(unvisited_actions, 
                                key=lambda x: self.q_table[state_key].get(x, 0))
                else:
                    next_state = max(valid_actions, 
                                key=lambda x: self.q_table[state_key].get(x, 0))
                
                final_path.append(next_state.move)
                current_state = next_state
                visited_states.add(self.get_state_key(current_state))
                
                if len(final_path) > self.max_states:  # Prevent infinite loops
                    return None, "no_solution"

            return final_path, None

        except Exception as e:
            print(f"Error in Q-learning algorithm: {e}")
            return None, "error"