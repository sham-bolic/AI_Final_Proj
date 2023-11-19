from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
import random


@register_agent("random4_agent")
class random4Agent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(random4Agent, self).__init__()
        self.name = "random4Agent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }

        # Moves (Up, Right, Down, Left)
        self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

        # Opposite Directions
        self.opposites = {0: 2, 1: 3, 2: 0, 3: 1}

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

        # Get all possible moves that we could make
        better_moves, losing_moves = self.get_possible_moves(chess_board, my_pos, adv_pos, max_step)

        # If we already lost, just play a random losing move
        if (len(better_moves) == 0):
            pos, dir = losing_moves[0]
            return pos, dir
        
        # If not, get the list of best possible moves based on our heuristic
        list_of_moves = self.get_best_moves(chess_board, adv_pos, max_step, better_moves)
        copy_chess_board = deepcopy(chess_board)
        print(f'NUMBER OF MOVES I CAN MAKE: {len(list_of_moves)}')
        print('###############################################')
        pos, dir = random.choice(list_of_moves)

        # Based on each of those moves, simulate games and see what moves gives the best future outcome 



        time_taken = time.time() - start_time
        
        print("My AI's turn took ", time_taken, "seconds.")

        return pos, dir
    
    def get_best_moves(self, chess_board, adv_pos, max_step, better_moves):
        list_of_offensive_moves = []
        free_space = 1000000

        # OFFENSIVE METHOD
        for move in better_moves:
            # Get position and wall of the current possible move
            my_pos, dir = move

            # Get all possible moves that my adversary can make if I were to perform "move" as my play this turn
            adv_better_moves, bad = self.get_adv_moves_given_wall_offense(chess_board, my_pos, adv_pos, max_step, dir)
            adv_good_moves, _ = self.get_possible_moves(chess_board, adv_pos, my_pos, max_step)

            # If we can corner our adversary exit the loop
            if not adv_better_moves:
                list_of_offensive_moves.clear()
                list_of_offensive_moves.append(move)
                break
            
            # If the potential new wall does not affect anything, skip offensive moves
            elif (adv_better_moves == adv_good_moves):
                continue

            # If we found a new better move, clear the list and update free space varaible
            elif len(adv_better_moves) < free_space:
                list_of_offensive_moves.clear()
                list_of_offensive_moves.append(move)
                free_space = len(adv_better_moves)

            # If we found a move with equivalent free space, add it to the list of moves
            elif len(adv_better_moves) == free_space:
                list_of_offensive_moves.append(move)

        # If some offensive moves were found, return the list of offensive moves
        if list_of_offensive_moves:
            return list_of_offensive_moves
        
        # If not, go to the defensive strategy
        list_of_defensive_moves = []
        free_space = -1

        # DEFENSIVE METHOD
        for move in better_moves:
            # Get position and wall of the current possible move
            my_pos, dir = move

            # Get all possible moves that I can make to have more options in the future
            my_better_moves, my_losing_moves = self.get_my_moves_given_wall_defensive(chess_board, my_pos, adv_pos, max_step, dir)

            # If we found a new better move, clear the list and update free space varaible
            if len(my_better_moves) > free_space:
                list_of_defensive_moves.clear()
                list_of_defensive_moves.append(move)
                free_space = len(my_better_moves)

            # If we found a move with equivalent free space, add it to the list of moves
            elif len(my_better_moves) == free_space:
                list_of_defensive_moves.append(move)

        return list_of_defensive_moves
    
    # Base on DEFENSIVE heuristic, our next move gives us more options in the future
    def get_my_moves_given_wall_defensive(self, chess_board, my_pos, adv_pos, max_step, new_wall):
        # Extract coordinate of my_pos and save inital positin
        my_pos_x, my_pos_y = my_pos 
        ini_pos = my_pos

        # Determine which move we cannot make if we place the new wall
        m = self.moves[new_wall]
        opposite_x, opposite_y = my_pos_x + m[0], my_pos_y + m[1]
        opposite_pos = (opposite_x, opposite_y)
        opposite_wall = self.opposites[new_wall]

        better_moves = []
        losing_moves = []
        state_queue = [(my_pos, 0)]
        visited = [(my_pos)]

        # BFS
        while state_queue:
            # Get next element
            curr_pos, curr_step = state_queue.pop(0)

            # First check if we are not outside of the step range
            if curr_step == max_step:
                break

            # get the position in x, y coordinate
            x, y = curr_pos

            # Check if we can advance in one of the four directions
            for dir, move in enumerate(self.moves):
                # If there is a wall we just skip because we can't move through it
                if chess_board[x, y, dir]:
                    continue

                # Check if we are not going through us or if we tried to go through the possible new placed wall form the start
                if (curr_pos == opposite_pos and dir == opposite_wall) or (curr_pos == ini_pos and dir == new_wall):
                    continue
                
                walls = 0
                for wall in range(4):
                    if chess_board[x, y, wall]:
                        walls += 1
                if walls < 3:
                    # Now we know we are safer, add the move to the list of possible moves
                    better_moves.append(((x,y), dir))
                else:
                    # In case we are stuck but have to play something anyway
                    losing_moves.append(((x,y), dir))

                # Update position
                new_x, new_y = x + move[0], y + move[1]
                new_pos = (new_x, new_y)

                # Verify that we don't go through ourself or that we did not already visited that position
                if new_pos == adv_pos or new_pos in visited:
                    continue

                # Add the new_pos to the visited list and to the state queue list with an updated step
                visited.append(new_pos)
                state_queue.append((new_pos, curr_step + 1))

        return better_moves, losing_moves
    
    # Base on OFFENSIVE heuristic, reducing the choice of play of our adversary
    def get_adv_moves_given_wall_offense(self, chess_board, my_pos, adv_pos, max_step, new_wall):
        # Extract coordinate of my_pos
        my_pos_x, my_pos_y = my_pos 

        # Determine which move our adv cannot make at a specific position 
        m = self.moves[new_wall]
        opposite_x, opposite_y = my_pos_x + m[0], my_pos_y + m[1]
        opposite_pos = (opposite_x, opposite_y)
        opposite_wall = self.opposites[new_wall]

        better_moves = []
        losing_moves = []
        state_queue = [(adv_pos, 0)]
        visited = [(adv_pos)]

        # BFS
        while state_queue:
            # Get next element
            curr_pos, curr_step = state_queue.pop(0)

            # First check if we are not outside of the step range
            if curr_step == max_step:
                break

            # get the position in x, y coordinate
            x, y = curr_pos

            # Check if we can advance in one of the four directions
            for dir, move in enumerate(self.moves):
                # If there is a wall we just skip because we can't move through it
                if chess_board[x, y, dir]:
                    continue

                # Check if we are not going through us or if we tried to go through the possible new placed wall
                if (curr_pos == opposite_pos and dir == opposite_wall):
                    continue
                
                walls = 0
                for wall in range(4):
                    if chess_board[x, y, wall]:
                        walls += 1
                if walls < 3:
                    # Now we know we are safe, add the move to the list of possible moves
                    better_moves.append(((x,y), dir))
                else:
                    # In case we are stuck but have to play something anyway
                    losing_moves.append(((x,y), dir))

                # Update position
                new_x, new_y = x + move[0], y + move[1]
                new_pos = (new_x, new_y)

                # Verify that we don't go through ourself or that we did not already visited that position
                if new_pos == my_pos or new_pos in visited:
                    continue

                # Add the new_pos to the visited list and to the state queue list with an updated step
                visited.append(new_pos)
                state_queue.append((new_pos, curr_step + 1))

        return better_moves, losing_moves
    
    def get_possible_moves(self, chess_board, my_pos, adv_pos, max_step):
        better_moves = []
        losing_moves = []
        state_queue = [(my_pos, 0)]
        visited = [(my_pos)]

        # BFS
        while state_queue:
            # Get next element
            curr_pos, curr_step = state_queue.pop(0)

            # First check if we are not outside of the step range
            if curr_step == max_step:
                break

            # get the position in x, y coordinate
            x, y = curr_pos

            # Check if we can advance in one of the four directions
            for dir, move in enumerate(self.moves):
                # If there is a wall we just kip because we can't move through it
                if chess_board[x, y, dir]:
                    continue

                # If there is no walls, make sure we are not surrounded by 3 other walls before adding a possible move
                walls = 0
                for wall in range(4):
                    if chess_board[x, y, wall]:
                        walls += 1
                if walls < 3:
                    # Now we know we are safe, add the move to the list of possible moves
                    better_moves.append(((x,y), dir))
                else:
                    # In case we are stuck but have to play something anyway
                    losing_moves.append(((x,y), dir))

                # Update position
                new_x, new_y = x + move[0], y + move[1]
                new_pos = (new_x, new_y)

                # Verify that we don't go through the adversary or that we did not already visited that position
                if new_pos == adv_pos or new_pos in visited:
                    continue

                # Add the new_pos to the visited list and to the state queue list with an updated step
                visited.append(new_pos)
                state_queue.append((new_pos, curr_step + 1))

        return better_moves, losing_moves