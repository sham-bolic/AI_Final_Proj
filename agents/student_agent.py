# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
import math 
from collections import defaultdict


@register_agent("student_agent")
class StudentAgent(Agent):

    class MonteCarloTreeSearchNode() :
        def __init__ (self, state, parent, c = math.sqrt(2)) :
            self.state = state
            elf.parent = parent
            self.children = []
            self.N = defaultdict(int)           # number of times visited
            self.Q = defaultdict(int)           # value of node
            self.A = defaultdict(int)           # number of actions taken

        def tree_policy(self):
            return (self.Q) + ( self.c * math.sqrt(math.log(self.N) / self.A))  
        
        #def is_terminal_node(self): 

        

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
        time_taken = time.time() - start_time
        
        print("My AI's turn took ", time_taken, "seconds.")

        # dummy return
        return my_pos, self.dir_map["u"]



     
