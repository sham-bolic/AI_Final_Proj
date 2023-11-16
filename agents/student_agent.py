# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
import math 
import random


@register_agent("student_agent")
class StudentAgent(Agent):

    class MonteCarloTreeSearchNode() :
        def __init__ (self, chess_board, my_pos, adv_pos, max_steps, parent=None, c = math.sqrt(2)) :
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
            self.A = 0                  # number of actions taken
            self.max_steps = max_steps
            self.all_moves = None
            self.all_moves = self.legal_moves() # List of legal moves
            self.dir = None
            self.dir = self.random_barrier(self.p0_pos)

        def tree_policy(self):          
            UCT = [((child.Q/child.N) +                              # Exploitation
                    (child.c * np.sqrt(np.log(child.N/child.A))))    # Exploration
                      for child in self.children]                    # For each child node
            return self.children[np.argmax(UCT)]                     # Returning best child based on UCT            
        
        def selection (self):                           # Selection policy
            curr_node = self
            while not curr_node.is_terminal_node(curr_node.chess_board, curr_node.p0_pos, curr_node.p1_pos):     # While is not the end of game
                if not curr_node.is_fully_expanded():   # While there are still moves to be made
                    return curr_node.expand()           # Expand the node
                else:
                    curr_node = curr_node.tree_policy() # Choose best child and return
            return curr_node
                
        def expand(self):                               # Make next move and add as child
            next_move = self.all_moves.pop(np.random.randint(0, len(self.all_moves)))
            board = self.move(self.chess_board, next_move)

            child = StudentAgent.MonteCarloTreeSearchNode(
                board, next_move, self.p1_pos, self.max_steps, parent = self
            )
            self.children.append(child)
            return child
                
        def move(self, board, pos):                            # takes position and simulates a move 
            chess_board = deepcopy(board)
            x, y = pos
            
            # Set the barrier to True
            chess_board[x, y, self.dir] = True
            # Set the opposite barrier to True
            move = self.moves[self.dir]
            chess_board[x + move[0], y + move[1], self.opposites[self.dir]] = True
            return chess_board
            
        def random_barrier(self, pos):
            barriers = StudentAgent.allowed_barriers(pos, self.chess_board)
            return barriers[np.random.randint(0, len(barriers))]


        def legal_moves(self):                                  # find all possible moves
            o_pos = deepcopy(self.p0_pos)
            steps = [[] for k in range(self.max_steps+1)]       # create 2D array to store possible moves for each step
            steps[0].append(o_pos)                                 # step 0 = original position
            for i in range(self.max_steps):
                for pos in steps[i]:
                    moves = StudentAgent.allowed_dirs(pos, self.p1_pos, self.chess_board)  
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
                return False, -9999                    # -5 if false
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
            n_simulation = 100

            for i in range(n_simulation):
                current = self.selection()
                result = current.simulate()
                current.backpropagate(result)
            best_node = self.tree_policy()
            best_pos = best_node.p0_pos
            best_dir = best_node.dir
            return best_pos, best_dir


        def backpropagate(self, result):
            self.N += 1
            self.Q += result
            if (self.parent):
                self.backpropagate()

        def simulate(self):
            board = deepcopy(self.chess_board) 
            p1 = self.p0_pos
            p2 = self.p1_pos
            original_player = True 
            while (not self.is_terminal_node(board, p1, p2)):       # While game is not over
                p1, p2 = self.random_moves(p2, p1, board)           # Take turns playing
                board = self.move(board, p1)                        # Update board
                original_player = not original_player 
            if (original_player): 
              _, result = self.check_endgame(board, p1, p2)
            else:
              _, result = self.check_endgame(board, p2, p1)                  
            sum += result                                        # Update value of node depending on result
            return sum

        def random_moves(self, p1, p2, board):                     # Literally random_agent
            steps = np.random.randint(0, self.max_step + 1)

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

            # Final portion, pick where to put our new barrier, at random
            r, c = p1
            # Possibilities, any direction such that chess_board is False
            # allowed_barriers=[i for i in range(0,4) if not board[r,c,i]]
            # Sanity check, no way to be fully enclosed in a square, else game already ended
            #assert len(allowed_barriers)>=1 
            #dir = allowed_barriers[np.random.randint(0, len(allowed_barriers))]

            return p1, p2

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
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

    def panic_room(walls):
        assert len(walls) >= 1                                          # insuring that were surrounded by 4 walls
          

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
        pos, dir = root.best_move()

        time_taken = time.time() - start_time
        
        print("My AI's turn took ", time_taken, "seconds.")

        # dummy return
        return pos, dir



     
