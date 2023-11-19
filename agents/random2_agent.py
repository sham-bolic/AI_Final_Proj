import numpy as np
from copy import deepcopy
from agents.agent import Agent
from store import register_agent

# Important: you should register your agent with a name
@register_agent("random2_agent")
class RandomAgent2(Agent):
    """
    Example of an agent which takes random decisions
    """

    def __init__(self):
        super(RandomAgent2, self).__init__()
        self.name = "RandomAgent"
        self.autoplay = True
        self.all_moves = None
        self.map = {0:(-1, 0), 1:(0, 1), 2:(1, 0), 3:(0, -1)}

    def step(self, chess_board, my_pos, adv_pos, max_step):
        self.all_moves = self.legal_moves(chess_board, my_pos, adv_pos, max_step)
        my_pos = self.random_moves(chess_board, max_step)
        walls = self.allowed_barriers(my_pos, chess_board)
        dir = walls[np.random.randint(len(walls))]
        return my_pos, dir
    
    def legal_moves(self, chess_board, my_pos, adv_pos, max_step):                                  # find all possible moves
        o_pos = deepcopy(my_pos)
        steps = [[] for k in range(max_step+1)]       # create 2D array to store possible moves for each step
        steps[0].append(o_pos)                                 # step 0 = original position
        for i in range(max_step):
            for pos in steps[i]:
                moves = RandomAgent2.allowed_dirs(pos, adv_pos, chess_board)  
                for move in moves:                          # iterates through legal moves given current position
                    new_move = tuple(np.add(pos, self.map[move]))           # getting new pos (x + a, y + b)
                    if new_move not in steps:               # checking if move is contained in array
                        steps[i+1].append(new_move)
        return list(set(sum(steps, []))) 
        
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
        return legal_walls     

    def random_moves(self, board, max_step):                     # Literally random_agent
            chess_board = deepcopy(board)
            steps = np.random.randint(0, max_step + 1)
            barriers = [1]
            
            while(len(barriers) == 1 and len(self.all_moves) > 0):
                move = self.all_moves.pop(np.random.randint(len(self.all_moves)))
                barriers = self.allowed_barriers(move, chess_board)
            return move
    
    def allowed_barriers(self, my_pos, chess_board):
        x, y = my_pos
        legal_walls = [i for i in range (0,4) if not chess_board[x, y, i]]
        return legal_walls 