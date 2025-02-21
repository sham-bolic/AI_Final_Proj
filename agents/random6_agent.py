# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time


@register_agent("random6_agent")
class Random6Agent(Agent):

    class MonteCarloTreeSearchNode() :
        def __init__ (self, chess_board, my_pos, adv_pos, max_steps, aggression = False, dir = None,parent=None, c = np.sqrt(2)) :
            self.chess_board = chess_board
            self.c = c
            self.p0_pos = my_pos
            self.p1_pos = adv_pos
            self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
            self.opposites = {0: 2, 1: 3, 2: 0, 3: 1}
            self.map = {0:(-1, 0), 1:(0, 1), 2:(1, 0), 3:(0, -1)}
            self.parent = parent
            self.children = []
            self.N = 0                  # number of times visited
            self.Q = 0                  # score of node (win: +1, loss: -1, tie: +0.5)
            self.aggression = aggression
            self.max_steps = max_steps
            self.all_moves = None
            self.all_moves = self.legal_moves(self.p0_pos, self.p1_pos, self.chess_board) # List of legal moves
            self.dir = dir      # parents decision
            self.simulation_moves = None

        def tree_policy(self):      
            UCT = [((child.Q/child.N) +                              # Exploitation
                    (child.c * np.sqrt(np.log(self.N/child.N))))    # Exploration
                      for child in self.children]                    # For each child node
            #breakpoint()    
            return self.children[np.argmax(UCT)]                     # Returning best child based on UCT            
        
        def selection (self):                           # Selection policy
            curr_node = self
            
            while not curr_node.is_terminal_node(curr_node.chess_board, curr_node.p0_pos, curr_node.p1_pos):     # While is not the end of game
                if (not curr_node.is_fully_expanded()):   # While there are still moves to be made
                    return curr_node.expand()         # Expand the node
                else:
                    curr_node = curr_node.tree_policy() # Choose best child and return
            return curr_node
        
        def manhatten_distance(self, p1, p2):
            x1, y1 = p1
            x2, y2 = p2
            return (np.abs(x1-x2)+np.abs(y1-y2))/2
                
        def expand(self):                               # Make next move and add as child
            next_move = self.all_moves.pop(np.random.randint(0, len(self.all_moves)))
            board, dir = self.move(self.chess_board, next_move)

            child = Random6Agent.MonteCarloTreeSearchNode(
                board, next_move, self.p1_pos, self.max_steps, dir = dir, parent = self
            )
            
            num_bar = len(Random6Agent.allowed_barriers(next_move, self.chess_board))            # Heuristic to avoid trapping self in
            if (num_bar == 1) :
                child.Q += -50
            elif (num_bar == 2) :
                child.Q += -5
           
            child.Q -= self.manhatten_distance(next_move, self.p1_pos)

            '''if (child.Q > 0):
                aggro = self.aggressive_playstyle(next_move, board)
                if (aggro > 0):
                    child.Q += aggro'''
            self.children.append(child)
            return child
                
        def move(self, board, pos):                            # takes position and simulates a move 
            chess_board = deepcopy(board)
            x, y = pos
            dir = self.barrier_sims(pos)

            # Set the barrier to True
            chess_board[x, y, dir] = True
            # Set the opposite barrier to True
            move = self.moves[dir]
            chess_board[x + move[0], y + move[1], self.opposites[dir]] = True
            return chess_board, dir
        
        def barrier_sims(self, move):
            p1 = move
            p2 = self.p1_pos
            board = deepcopy(self.chess_board) 
            original_player = True
            barrier_list = [-999,-999,-999,-999]

            for i in Random6Agent.allowed_barriers(p1, board):
                score = 0
                for _ in range(2):
                    current_state = deepcopy(self.simulation_moves)

                    while (not self.is_terminal_node(board, p1, p2)):       # While game is not over
                        p1, p2, board = self.simulation_step(board, p2, p1)           # Take turns playing
                        original_player = not original_player 
                    if (original_player): 
                        _, result = self.check_endgame(board, p1, p2)
                    else:
                        _, result = self.check_endgame(board, p2, p1)
                    score += result
                    self.simulation_moves = current_state      
                barrier_list[i] = score
            
            agressive_bars, pref_bar = self.aggressive_barrier(move)
            for i in agressive_bars:
                barrier_list[i] += 0.5
            barrier_list[pref_bar] += 0.5

            best_barrier = np.argmax(barrier_list)
            #breakpoint()
            self.backpropagate(barrier_list[best_barrier])
            return best_barrier
        
        def aggressive_barrier(self, move):
            p1 = move
            p2 = self.p1_pos
            board = deepcopy(self.chess_board)

            allowed_barriers = Random6Agent.allowed_barriers(p1, board)

            x1,y1 = p1
            x2,y2 = p2
            x_diff = x2 - x1        # > 0 r, < 0 l 
            y_diff = y2 - y1        # > 0 d, < 0 u
            
            preferred_dir = []

            if x_diff > 0:
                preferred_dir.append(2)
            else:
                preferred_dir.append(0)

            if y_diff > 0:
                preferred_dir.append(1)
            else: 
                preferred_dir.append(3)

            if np.abs(x_diff) > np.abs(y_diff):
                dir = preferred_dir[0]
            else:
                dir = preferred_dir[1]

            return preferred_dir, dir
            
            

        def random_barrier(self, pos):
            barriers = Random6Agent.allowed_barriers(pos, self.chess_board)
            return barriers[np.random.randint(0, len(barriers))]


        def legal_moves(self, o_pos, adv_pos, chess_board):                                  # find all possible moves
            steps = [[] for k in range(self.max_steps+1)]       # create 2D array to store possible moves for each step
            steps[0].append(o_pos)                                 # step 0 = original position
            for i in range(self.max_steps):
                for pos in steps[i]:
                    moves = Random6Agent.allowed_dirs(pos, adv_pos, chess_board)  
                    for move in moves:                          # iterates through legal moves given current position
                        new_move = tuple(np.add(pos, self.map[move]))           # getting new pos (x + a, y + b)
                        if new_move not in steps:               # checking if move is contained in array
                            steps[i+1].append(new_move)
            return list(set(sum(steps, [])))                    # flattening the array so it is only 1D
                         

        def is_fully_expanded(self):                            # checks if there are any possible moves left
            return len(self.all_moves) == 0
        
        def is_terminal_node(self, board, p1, p2):     # Matches endgame and extracts boolean
            x, _ = self.check_endgame(board, p1, p2)
            return x        
        
        def check_endgame(self, board, p1, p2):        # returns (isendgame, p1 score, p2 score)
            # Union-Find
            father = dict()
            for r in range(self.board_size()):
                for c in range(self.board_size()):
                    father[(r, c)] = (r, c)

            def find(pos):
                if father[pos] != pos:
                    father[pos] = find(father[pos])
                return father[pos]

            def union(pos1, pos2):
                father[pos1] = pos2

            for r in range(self.board_size()):
                for c in range(self.board_size()):
                    for dir, move in enumerate(
                        self.moves[1:3]
                    ):  # Only check down and right
                        if board[r, c, dir + 1]:
                            continue
                        pos_a = find((r, c))
                        pos_b = find((r + move[0], c + move[1]))
                        if pos_a != pos_b:
                            union(pos_a, pos_b)

            for r in range(self.board_size()):
                for c in range(self.board_size()):
                    find((r, c))
            p0_r = find(tuple(p1))
            p1_r = find(tuple(p2))
            p0_score = list(father.values()).count(p0_r)
            p1_score = list(father.values()).count(p1_r)
            if p0_r == p1_r:
                return False, -9999                    # Doesn't matter what number is returned with bool
            player_win = None
            if p0_score > p1_score:
                player_win = 1
            elif p0_score < p1_score:
                player_win = -1
            else:
                player_win = 0.5  # Tie
            return True, player_win                 # p1 = 0, p2 = 1, tie = -1
        
        def board_size(self):                       # Board is NxNx4 capturing N
            x, _, _ = np.shape(self.chess_board)
            return x
        
        def best_move(self):
            
            start_time = time.time()
            turn_time = 1.9             # set to 0.5 for testing, 1.9 for real min max
            end_time = start_time + turn_time
            sims = 0
            while(time.time()<end_time):
                current = self.selection()
                current.simulation_moves = current.legal_moves(current.p0_pos, self.p1_pos, self.chess_board)
                result = current.simulate()
                current.backpropagate(result)
                sims += 1
            #breakpoint()
            best_node = self.tree_policy()
            best_pos = best_node.p0_pos
            best_dir = best_node.dir
            return best_pos, best_dir


        def backpropagate(self, result):
            self.N += 1
            self.Q += result
            if (self.parent):
                self.parent.backpropagate(result)

        def simulate(self):
            board = deepcopy(self.chess_board) 
            p1 = self.p0_pos
            p2 = self.p1_pos
            original_player = True 
            while (not self.is_terminal_node(board, p1, p2)):       # While game is not over
                p1, p2, board = self.simulation_step(board, p2, p1)           # Take turns playing
                original_player = not original_player 
            if (original_player): 
              _, result = self.check_endgame(board, p1, p2)
            else:
              _, result = self.check_endgame(board, p2, p1)                  
            return result                                        # Update value of node depending on result
            

        def random_moves(self, board):                     # Literally random_agent
            '''chess_board = deepcopy(board)
            
            steps = np.random.randint(0, self.max_steps + 1)

            # Pick steps random but allowable moves
            for _ in range(steps):
                r, c = p1
                # Build a list of the moves we can make
                allowed_dirs = [ d                                
                    for d in range(0,4)                                      # 4 moves possible
                    if not board[r,c,d] and                       # chess_board True means wall
                    not p2 == (r+self.moves[d][0],c+self.moves[d][1])]  # cannot move through Adversary
                if len(allowed_dirs)==0:
                    # If no possible move, we must be enclosed by our Adversary
                    break
                random_dir = allowed_dirs[np.random.randint(0, len(allowed_dirs))]
                # This is how to update a row,col by the entries in moves 
                # to be consistent with game logic
                m_r, m_c = self.moves[random_dir]
                p1 = (r + m_r, c + m_c)
            dir = self.random_barrier(p1)
            x, y = p1
            # Set the barrier to True
            chess_board[x, y, dir] = True
            # Set the opposite barrier to True
            move = self.moves[dir]
            chess_board[x + move[0], y + move[1], self.opposites[dir]] = True
            
            return p1, p2, chess_board'''
            
            chess_board = deepcopy(board)
            steps = np.random.randint(0, self.max_steps + 1)
            barriers = [1]
            
            while(len(barriers) == 1 and len(self.simulation_moves) > 0):
                move = self.simulation_moves.pop(np.random.randint(len(self.simulation_moves)))
                #breakpoint()
                barriers = Random6Agent.allowed_barriers(move, chess_board)
            return move
        
        def simulation_step(self, chess_board, my_pos, adv_pos):
            self.simulation_moves = self.legal_moves(my_pos, adv_pos,chess_board)
            my_pos = self.random_moves(chess_board)
            walls = Random6Agent.allowed_barriers(my_pos, chess_board)
            dir = walls[np.random.randint(len(walls))]
            # Set the barrier to True
            x, y = my_pos
            chess_board[x, y, dir] = True
            # Set the opposite barrier to True
            move = self.moves[dir]
            chess_board[x + move[0], y + move[1], self.opposites[dir]] = True
            return my_pos, adv_pos, chess_board
        
        def aggressive_playstyle(self, next_move, chessboard):
            current_moves = len(self.adv_moves(self.p0_pos, self.chess_board))
            next_moves    = len(self.adv_moves(next_move, chessboard))
            return (current_moves - next_moves)

        def adv_moves(self, next_move,chessboard):
            return self.legal_moves(self.p1_pos, next_move, chessboard)

            

    def __init__(self):
        super(Random6Agent, self).__init__()
        self.name = "Random6Agent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }

    def allowed_dirs(my_pos, adv_pos, chess_board):
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))                      # Possible changes in x and y
        x, y = my_pos                                                   
        possible_moves = [ d                                            # storing d if there is not a wall nor adv in the way
            for d in range (0,4)
            if not chess_board[x, y, d] and
            not adv_pos == (x + moves[d][0], y + moves[d][1])
        ]
        return possible_moves

    def allowed_barriers(my_pos, chess_board):
        x, y = my_pos
        legal_walls = [i for i in range (0,4) if not chess_board[x, y, i]]
        return legal_walls                                              # returning the possible places that wall can be placed for given my_pos

    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """

        
        # Some simple code to help you with timing. Consider checking 
        # time_taken during your search and breaking with the best answer
        # so far when it nears 2 seconds.
        start_time = time.time()
        root = self.MonteCarloTreeSearchNode(chess_board, my_pos, adv_pos, max_step)
        #breakpoint()
        pos, dir = root.best_move()

        time_taken = time.time() - start_time
        
        print("My AI's turn took ", time_taken, "seconds.")

        # dummy return
        
        return pos, dir



     
