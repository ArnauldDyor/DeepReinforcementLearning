import numpy as np
import time

LENGTH = 3


def get_state_hash_and_winner(env, i=0, j=0):
    results = []

    for v in (0, env.x, env.o):
        env.board[i, j] = v
        if j == 2:
            if i == 2:
                state = env.get_state()
                ended = env.game_over(force_recalculate=True)
                winner = env.winner
                results.append((state, winner, ended))
            else:
                results += get_state_hash_and_winner(env, i + 1, 0)
        else:
            results += get_state_hash_and_winner(env, i, j + 1)

    return results


def Initialize_value_x(env, state_winner_triple):
    # initialize states values as follows
    # if x wins, V(s) = 1
    # if x loses or draw, V(s) = 0
    # otherwise, V(s) = 0.5
    V = np.zeros(env.num_states)
    for state, winner, ended in state_winner_triple:
        if ended:
            if winner == env.x:
                v = 1
            else:
                v = 0
        else:
            v = 0.5
        V[state] = v
    return V


def Initialize_value_o(env, state_winner_triple):
    V = np.zeros(env.num_states)
    for state, winner, ended in state_winner_triple:
        if ended:
            if winner == env.o:
                v = 1
            else:
                v = 0
        else:
            v = 0.5
        V[state] = v
    return V

# number between (1..9)
def number_to_coordinates_human(case_number):
    i, j = (int(case_number) - 1) // 3, (int(case_number) - 1) % 3
    return i, j

# number between (1..9)
def number_to_coordinates(case_number):
    i, j = (int(case_number)) // 3, (int(case_number)) % 3
    return i, j
# number between (0..8)
def coordinates_to_number(row, column):
    case_number = int(row) * 3 + int(column)
    return case_number



class Environment(object):

    def __init__(self):
        self.board = np.zeros((LENGTH, LENGTH))
        self.x = -1 # player 1
        self.o = 1 # player
        self.winner = None
        self.ended = False
        self.num_states = 3**(LENGTH*LENGTH)

    def is_empty(self, i, j):
        return self.board[i, j] == 0

    # mapping state to a number
    #  D = 3**(N-1) *b(n-1) + ... + 3**(1) *b1 + 3**(0) * b0
    # b∈[0, 1, 2]
    # returns the current state
    def get_state(self):
        hash = 0
        k = 0
        for i in range(LENGTH):
            for j in range(LENGTH):
                if self.board[i, j] == 0:
                    v = 0
                elif self.board[i, j] == self.x:
                    v = 1
                elif self.board[i, j] == self.o:
                    v = 2
                hash += (3**k) * v
                k += 1
        return hash


    def reset(self):
        self.env = np.zeros((LENGTH, LENGTH))

    # check if there is a winner
    # across rows, columns, diagonals
    def game_over(self, force_recalculate=False):

        if not force_recalculate and self.ended:
            return self.ended
        # check rows
        for i in range(LENGTH):
            for player in (self.x, self.o):
                if self.board[i].sum() == player * LENGTH:
                    # print("GAME OVER (3 in a row)")
                    self.winner = player
                    self.ended = True
                    return True

        # check columns
        for j in range(LENGTH):
            for player in (self.x, self.o):
                if self.board[:, j].sum() == player * LENGTH:
                    # print("GAME OVER (3 in a column)")
                    self.winner = player
                    self.ended = True
                    return True

        # check diagonals
        for player in (self.x, self.o):
            #  top-left -> bottom-right diagonal
            if self.board.trace() == player * LENGTH:
                # print("GAME OVER (3 in diagonal top-left -> bottom-right)")
                self.winner = player
                self.ended = True
                return True
            #  top-right -> bottom-left diagonal
            if np.fliplr(self.board).trace() == player * LENGTH:
                # print("GAME OVER (3 in diagonal top-right -> bottom-left)")
                self.winner = player
                self.ended = True
                return True
        #  check if draw
        if np.all((self.board == 0) == False):
            #  winner stays None
            # print("DRAW MATCH")
            self.winner = None
            self.ended = True
            return True

        # game is not over
        self.winner = None
        return False


    def draw_board(self):
        for i in range(LENGTH):
            print("-------------")
            for j in range(LENGTH):
                print("  ", end="")
                if self.board[i, j] == self.x:
                    print("x ", end="")
                elif self.board[i, j] == self.o:
                    print("o ", end="")
                else:
                    print("  ", end="")
            print("")
        print("-------------")


    def reward(self, symbol):
        # Cannot get reward until game is over
        if not self.game_over():
            return 0
        #  symbol will be either self.x or self.o
        return 1 if self.winner == symbol else -1


class Agent(object):

    # epsilon-greedy strategy
    def __init__(self, eps=0.2, gamma=0.9, learning_rate = 0.1):
        self.eps = eps  # probability of choosing random action insted of greedy
        self.gamma = gamma # importance of the future rewards
        self.learning_rate = learning_rate
        self.verbose = False
        self.state_history = []
        self.actions_n = 3  # 0, x or o
        self.states_n = self.actions_n**(LENGTH * LENGTH)
        # rows are states, column are moves (0..8)
        self.q_table = np.zeros((self.states_n, 9))


    def set_symbol(self, symbol):
        self.sym = symbol

    def set_verbose(self, v):
        self.verbose = v

    def take_action(self, env, eps = None, display = True):
        # choose an action based on epsilon-greedy strategy
        if eps is None:
            eps = self.eps
        action = None
        random_choice = np.random.rand()
        state = env.get_state()
        if random_choice < eps:
            # random action
            if self.verbose:
                print("Taking a random action")
            # make a list of possible moves and choose one for the next one
            possible_moves = []
            for i in range(LENGTH):
                for j in range(LENGTH):
                    if env.is_empty(i, j):
                        possible_moves.append((i, j))
            idx = np.random.choice(len(possible_moves))
            action_coordinates = possible_moves[idx]
            action = coordinates_to_number(action_coordinates[0], action_coordinates[1])
        # greedy part
        else:
            next_move = None
            best_value = -999999
            if self.verbose and display:
                print("Taking a greedy action")
            for i in range(LENGTH):
                for j in range(LENGTH):
                    #print(i,j)
                    if env.is_empty(i, j):
                        case_number = coordinates_to_number(i, j)
                        value = self.q_table[i][j]
                        if value > best_value:
                            best_value = value
                            action = case_number
                            action_coordinates = (i, j)

            # if verbose, draw the board w/ the value
            if self.verbose:
                
                for i in range(LENGTH):
                    print("------------------")
                    for j in range(LENGTH):
                        if env.is_empty(i, j):
                            # print the value
                            case_number = coordinates_to_number(i, j)
                            print(" %.2f|" % self.q_table[state][case_number], end="")
                        else:
                            print("  ", end="")
                            if env.board[i, j] == env.x:
                                print("x  |", end="")
                            elif env.board[i, j] == env.o:
                                print("o  |", end="")
                            else:
                                print("   |", end="")
                    print("")
                print("------------------")


        return action


    def execute_action(self, env, action):
        action_coordinates = number_to_coordinates(action)
        env.board[action_coordinates[0], action_coordinates[1]] = self.sym
        next_state = env.get_state()
        reward = env.reward(self.sym)

        return next_state, reward


class Human:
    def __init__(self):
        self.state_history = []
        self.q_table = np.zeros((19683, LENGTH * LENGTH))

    def set_symbol(self, sym):
        self.sym = sym

    def take_action(self, env, eps=0.0, display = True):
        while True:
            # break if we make a legal move
            action = input("Please enter case's number for your next move (1..9): ")
            i, j = number_to_coordinates_human(action)
            if env.is_empty(i, j):
                return action

    def execute_action(self, env, action):
        action_coordinates = number_to_coordinates_human(action)
        env.board[action_coordinates[0], action_coordinates[1]] = self.sym
        next_state = env.get_state()
        reward = env.reward(self.sym)
        return next_state, reward


def play_game(p1, p2, env, draw=False):
    # loops until the game is over

    current_player = np.random.choice([p1, p2])

    while not env.game_over():
        if current_player == p1:
            current_player = p2
        else:
            current_player = p1

        state = env.get_state()
        action = current_player.take_action(env)

        state_plus_1, reward = current_player.execute_action(env, action)

        # Update Q function
        action_plus_1 = current_player.take_action(env, 0.0)
        if action_plus_1 is None:
            break
        current_player.q_table[state][action] = current_player.q_table[state][action] + current_player.learning_rate*(reward + current_player.gamma*current_player.q_table[state_plus_1][action_plus_1] - current_player.q_table[state][action])


def play_game_human(p1, human, env, draw=False):
    # loops until the game is over
    current_player = human

    while not env.game_over():
        if current_player == p1:
            current_player = human
        else:
            current_player = p1

        env.draw_board()
        state = env.get_state()
        action = current_player.take_action(env)

        state_plus_1, reward = current_player.execute_action(env, action)

        # Update Q function
        if not current_player == human:
            action_plus_1 = current_player.take_action(env, 0.0, display = False)
            if action_plus_1 is not None:
                current_player.q_table[state][action] = current_player.q_table[state][action] + current_player.learning_rate*(reward + current_player.gamma*current_player.q_table[state_plus_1][action_plus_1] - current_player.q_table[state][action])

        env.draw_board()



if __name__ == '__main__':
    #  train the agent
    p1 = Agent()
    p2 = Agent()

    #  set initial V for p1 and p2
    env = Environment()
    state_winner_triples = get_state_hash_and_winner(env)

    # give each player their symbol
    p1.set_symbol(env.x)
    p2.set_symbol(env.o)

    # p2.set_verbose(True)
    epochs = 250

    start_time = time.time()
    for epoch in range(epochs):
        if epoch % 20000 == 0:
            #print(epoch,"/",epochs)
            print("%i/%i (%.2f%%)" %(epoch, epochs, epoch/epochs *100))
            p1_path = "models/q_learning_p1_ep%s.npy" %str(epochs)
            p2_path = "models/q_learning_p2_ep%s.npy" %str(epochs)
            np.save(p1_path, p1.q_table)
            np.save(p2_path, p2.q_table)
        play_game(p1, p2, Environment(), draw=True)
    print("training time : %.0fs" %(time.time() - start_time))
    
    p1 = Agent()
    human = Human()
    p1.set_symbol(env.x)
    human.set_symbol(env.o)
    q_table_to_load =  "models/q_learning_p2_ep2500000.npy" #p1_path
    p1.q_table = np.load(q_table_to_load)
    p1.set_verbose(True)
    while True:
        #p1.set_verbose(True)
        play_game_human(p1, human, Environment(), draw=2)
        answer = input("play again? [Y/n]: ")
        if answer and answer.lower()[0] == 'n':
            break
