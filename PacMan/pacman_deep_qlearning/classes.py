"""
        __________________________________

        Game's classes : Level, Pacman, Ghosts
        __________________________________
"""

"""
    Class for creating the levels
"""

from constants import *
import random
import tensorflow
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *


class Level:

    def __init__(self):
        self.file = LEVEL_MAP
        self.structure = []
        self.board = []

    def generate(self):
        """ method used to generate level depending on the file."""

        if self.structure == []:
            with open(self.file, "r") as file:

                level_structure = []

                for line in file:
                    lines = []
                    for sprite in line:
                        if sprite != '\n':
                            lines.append(sprite)
                    level_structure.append(lines)
                self.structure = level_structure

    def display(self, window):
        """Method for displaying the level depending on
        the structure of list returned by generate() method"""

        self.generate()
        wall = pygame.image.load(WALL_IMG).convert()
        way = pygame.image.load(WAY_IMG).convert()
        point = pygame.transform.scale(pygame.image.load(POINT_IMG), (5, 5))

        # browse throught the level lines
        line_num = 0

        for line in self.structure:
            square_num = 0
            board_line = []
            for sprite in line:
                x = square_num * SPRITE_SIZE
                y = line_num * SPRITE_SIZE
                if sprite == 'w':  # w = wall
                    window.blit(wall, (x, y))
                    board_line.append(0)
                else:
                    window.blit(way, (x, y))
                    board_line.append(1)
                
                square_num += 1
            self.board.append(board_line)
            line_num += 1


class PacMan:

    def __init__(self):
        self.right = pygame.image.load(PACMAN_RIGHT).convert_alpha()
        self.left = pygame.image.load(PACMAN_LEFT).convert_alpha()
        self.up = pygame.image.load(PACMAN_UP).convert_alpha()
        self.down = pygame.image.load(PACMAN_DOWN).convert_alpha()
        #  Pacman position
        self.case_x = 9
        self.case_y = 8
        self.x = self.case_x * SPRITE_SIZE
        self.y = self.case_y * SPRITE_SIZE

        # default direction
        self.direction = self.right
        # selected level
        level_class = Level()
        level_class.generate()
        self.map = level_class.structure
        self.x_change = 0
        self.y_change = 0
        self.states_n = 44 
        self.q_table = self.q_table = np.zeros((self.states_n, 4))


    def take_action(self, st, Q, eps=0.1):

        if eps is None:
            eps = self.eps
        random_choice = np.random.rand()
        # random action
        if random_choice < eps:
            action = random.randint(0,3)
        # greedy part
        else:
            action = np.argmax(Q[st])
        return action

    def change_direction(self, direction):

        self.x_change = 0
        self.y_change = 0

        if direction == "right":
            self.x_change = 1
            self.y_change = 0
            self.direction = self.right

        elif direction == "left":
            self.x_change = -1  # moving by one case
            self.y_change = 0  # moving by one case
            self.direction = self.left

        elif direction == "up":
            self.x_change = 0
            self.y_change = -1
            self.direction = self.up

        elif direction == "down":
            self.x_change = 0  # moving by one case
            self.y_change = 1  # moving by one case
            self.direction = self.down

    def move(self):

        if self.case_x < (SPRITE_WIDTH + self.x_change) and self.case_y < (SPRITE_HEIGHT + self.y_change):
            # check if there is not a wall
            if self.map[self.case_y + self.y_change][self.case_x + self.x_change] != 'w':
                self.case_x = self.case_x + self.x_change  # moving by one case
                self.x = self.case_x * SPRITE_SIZE
                self.case_y = self.case_y + self.y_change  # moving by one case
                self.y = self.case_y * SPRITE_SIZE

    def update(self):

    
    # l = E(s,a,r,s') [(r + gamma * max Q(s',a' teta) - Q(s,a,teta))square]
    def calcul_loss(self, model, state, action, reward, next_state, qtarget):
        v = states[st][] * (r + self.gamma * self.q_table[])
        return model.predict()

    def model(self, batch):
        model = Sequential()
        model.add(Dense(batch, activation='relu', use_bias=True))
        model.add(Dense(4, activation='linear', use_bias=True))

        optimizer = Adam(0.01)
