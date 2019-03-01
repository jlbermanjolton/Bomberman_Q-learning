# This is necessary to find the main code
import random
import sys

sys.path.insert(0, '../bomberman')
# Import necessary stuff
from entity import CharacterEntity
from colorama import Fore, Back
import math

from sensed_world import SensedWorld

entities = ['wall', 'hero', 'enemy', 'bomb', 'explosion', 'monster', 'exit']
actions = ['N', 'S', 'E', 'W', 'NE', 'NW', 'SE', 'SW', 'bomb', 'wait']

# TODO Save and Load Q_Table from file for persistence (optionally human readable ie JSON)
Q_Table = []
previous_state = None
previous_action = None
previous_reward = None

class WorldState:
    def __init__(self):
        self.my_x = 0
        self.my_y = 0
        self.exit_x = 0
        self.exit_y = 0
        # list of bombs
        self.bomb_x = []
        self.bomb_y = []
        # list of explosions
        self.exp_x = []
        self.exp_y = []
        # closest monster
        self.c_monst_x = math.inf
        self.c_monst_y = math.inf
        # list of monsters
        self.monst_x = []
        self.monst_y = []
    
    # returns a tuple consisting of the euclidean distance to the point, the x distance, and the y distance
    def dist(self, other_x, other_y):
        dx = other_x - self.my_x
        dy = other_y - self.my_y
        dist = math.sqrt(dx**2 + dy**2)
        return (dist, dx, dy)
    
    def bomb_danger_x(self):
        for i in range(len(self.bomb_x)):
            if self.my_x == self.bomb_x[i]:
                return 1
        return 0
        
    def bomb_danger_y(self):
        for i in range(len(self.bomb_x)):
            if self.my_y == self.bomb_y[i]:
                return 1
        return 0
        
    # returns a vector of whether each move is safe
    def all_move_danger(self):
        moves = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i != 0 or j != 0:
                    moves.append(self.move_danger(i, j))
        return moves
        
    # checks if a single move is safe
    def move_danger(self, dx, dy):
        for i in range(len(self.exp_x)):
            if self.my_x + dx == self.exp_x[i] and self.my_y + dy == self.exp_y[i]:
                return 1
        for i in range(len(self.monst_x)):
            if self.my_x + dx == self.monst_x[i] and self.my_y + dy == self.monst_y[i]:
                return 1
        return 0
    
    # compare two WorldState objects (note - if we end up involving inheritance this could get messed up)
    # https://stackoverflow.com/questions/390250/elegant-ways-to-support-equivalence-equality-in-python-classes
    def __eq__(self, other):
        if self.dist(self.exit_x, self.exit_y)[0] != other.dist(other.exit_x, other.exit_y)[0]:
            return False
        if self.dist(self.c_monst_x, self.c_monst_y)[0] != other.dist(other.c_monst_x, other.c_monst_y)[0]:
            return False
        if self.bomb_danger_x() != other.bomb_danger_x():
            return False
        if self.bomb_danger_y() != other.bomb_danger_y():
            return False
        return True
        
    def to_vector(self):
        vect_form = []
        vect_form.append(self.dist(self.exit_x, self.exit_y)[0])
        vect_form.append(self.dist(self.c_monst_x, self.c_monst_y)[0])
        vect_form.append(self.bomb_danger_x())
        vect_form.append(self.bomb_danger_y())
        vect_form += self.all_move_danger()
        return vect_form
        

class TestCharacter(CharacterEntity):

    def do(self, wrld):
        # TODO Create a Rewards Table and update the previous reward

        # TODO Update the Q_Table based on previous state-action-reward

        # Obtain the state of the current world
        world_cpy = SensedWorld.from_world(wrld)

        # Construct a grid of all possible states of the world
        # This consists of an X * Y * entities array
        # In the sample game this is 8x19x7 resulting in a state space of 1.86*10^137 (TOO LARGE)
        grid = []
        state = WorldState()
        for x in range(0, world_cpy.width()):
            column = []
            for y in range(0, world_cpy.height()):
                column.append([0] * len(entities))
                # Check if Wall in location, if so flip corresponding position
                if world_cpy.grid[x][y]:
                    column[y][entities.index("wall")] = 1
            grid.append(column)

        # Add Characters from world to grid
        for i, character_list in world_cpy.characters.items():
            for character in character_list:
                if character == world_cpy.me(self):  # Found current player
                    grid[character.x][character.y][entities.index("hero")] = 1
                    state.my_x = character.x
                    state.my_y = character.y
                else:
                    grid[character.x][character.y][entities.index("enemy")] = 1

        # Add bombs from world to grid
        for bomb in world_cpy.bombs.items():
            grid[bomb[1].x][bomb[1].y][entities.index("bomb")] = 1
            state.bomb_x.append(bomb[1].x)
            state.bomb_y.append(bomb[1].y)
            

        # Add explosions from world to grid
        for explosion in world_cpy.explosions.items():
            grid[explosion[1].x][explosion[1].y][entities.index("explosion")] = 1

        # Add monsters from world to grid
        for i, monster_list in world_cpy.monsters.items():
            for monster in monster_list:
                grid[monster.x][monster.y][entities.index("monster")] = 1
                closest_monst = state.dist(state.monst_x, state.monst_y)
                new_monst = state.dist(monster.x, monster.y)
                if new_monst[0] < closest_monst[0]:
                    state.monst_x = monster.x
                    state.monst_y = monster.y

        # Add exit cell from world to grid
        if world_cpy.exitcell is not None:
            grid[world_cpy.exitcell[0]][world_cpy.exitcell[1]][entities.index("exit")] = 1
            state.exit_x = world_cpy.exitcell[0]
            state.exit_y = world_cpy.exitcell[1]

        # Get the current action score pairs of the current world state
        action_score_pairs = None
        found = False
        for entry in Q_Table:
            if entry[0] == state:
                action_score_pairs = entry[1]
                found = True
                break

        # If the state is not currently in the QTable, add it with 0's as Q values for all actions
        # TODO is propagating new values as zero the best strategy?
        if not found:
            action_score_pairs = [[0] * len(actions)]
            Q_Table.append([state, action_score_pairs])

        # Using epsilon greedy formula, choose an action based on the obtained action score pairs
        # TODO implement epsilon greedy, using random currently
        action_selection = random.randint(0, len(actions) - 1)

        # Parse action selection and perform action
        action_selection_string = actions[action_selection]
        if action_selection_string == 'N':
            self.move(0, -1)
        elif action_selection_string == 'NE':
            self.move(1, -1)
        elif action_selection_string == 'NW':
            self.move(-1, -1)
        elif action_selection_string == 'S':
            self.move(0, 1)
        elif action_selection_string == 'SE':
            self.move(1, 1)
        elif action_selection_string == 'SW':
            self.move(-1, 1)
        elif action_selection_string == 'E':
            self.move(1, 0)
        elif action_selection_string == 'W':
            self.move(-1, 0)
        elif action_selection_string == 'wait':
            self.move(0, 0)
        elif action_selection_string == 'bomb':
            self.place_bomb()
        
        '''
        # testing moving based on exit_dist
        exit_dist = state.dist(state.exit_x, state.exit_y)
        self.move(d_exit[1]/abs(d_exit[1]), d_exit[2]/abs(d_exit[2]))
        '''
        # Update the persistent variables to hold information on this action selection
        # Reward is unknown until action is completed ie start of this function
        previous_state = grid
        previous_action = action_selection
        previous_reward = None

        # Profiling
        memory_usage = sys.getsizeof(Q_Table)

        pass
