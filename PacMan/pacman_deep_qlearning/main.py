"""
    ____________________________________________________
                    <<  Pac-Man game  >>
    Files : main.py, classes.py, constants.py, resources
    ____________________________________________________
"""

import cv2
import os
import time
import datetime


from classes import *



    


ghost_move_count = 0
pygame.init()

random.seed()
direction = ""
score = 0
x_change, y_change = 0, 0
#  Pygame window
window = pygame.display.set_mode(WINDOW_SIZE)
icon = pygame.image.load(ICON)
pygame.display.set_icon(icon)
pygame.display.set_caption(WINDOW_TITLE)



pygame.time.Clock().tick(TICK_LIMIT)  # to limit the loop speed
                # Loading the background
                background = pygame.image.load(BACKGROUND_IMG).convert()

                #  generate level
                level_generator = Level()
                # level_generator.generate()
                level_generator.display(window)
                print(level_generator.board)
                pacman = PacMan()
                playing_loop = 1
                home_loop = 0

# Principal loop
game_loop = 1
epochs = 2
for epoch in range(epoch):

    pacman_last_pos = (pacman.case_y, pacman.case_x)

    current_player = np.random.choice([p1, p2])
    #env.get_state()
    while not env.game_over():
        action = current_player.take_action(env)
        #print("action", action)

        state_plus_1, reward = current_player.execute_action(env, action)
        #print("state_plus_1, reward", state_plus_1, reward)
        #print("r", r)

        # Update Q function
        action_plus_1 = current_player.take_action(env, 0.0)
        #print("action_plus_1", action_plus_1)
        current_player.q_table[state][action] = current_player.q_table[state][action] + current_player.learning_rate*(reward + current_player.gamma*current_player.q_table[state_plus_1][action_plus_1] - current_player.q_table[state][action])
        #print("Q[{}][{}] = {}".format(state, action, current_player.q_table[state][action]))
        #print("Q", current_player.q_table)
        #print("new state ", Q)



        # moving PACMAN
        pacman.change_direction("right")

        pacman.move()

        if not is_keydown:
            pacman.move()
            ghost_move_count += 1
            # ghost.move()
        #print("map :", level_generator.structure)
        # Display the new positions
        window.blit(background, (0, 0))

        print("pacman :", pacman.case_y, pacman.case_x)

        level_generator.display(window)
        window.blit(pacman.direction, (pacman.x + 5, pacman.y + 5))  # +5 refers to center pacman
        # window.fill((255, 0, 0), rect=(blinky.path[0][1] * 30, blinky.path[0][0] * 30, 30, 30))
        # window.fill((255,155, 0), rect=(clyde.path[0][1] * 30, clyde.path[0][0] * 30, 30, 30))
        pygame.display.flip()

        pygame.time.Clock().tick(1.5)

