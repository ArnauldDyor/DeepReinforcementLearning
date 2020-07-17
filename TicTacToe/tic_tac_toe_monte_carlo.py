import numpy as np

LENGTH = 3

# returns for each state, if there is a winner and if it's over or not
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

# returns the value function for the first player
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

# returns the value function for the second player
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

    def reward(self, symbol):
        # Cannot get reward until game is over
        if not self.game_over():
            return 0
        #  symbol will be either self.x or self.o
        return 1 if self.winner == symbol else 0

    # mapping state to a number
    #  D = 3**(N-1) *b(n-1) + ... + 3**(1) *b1 + 3**(0) * b0
    # b∈[0, 1, 2]
    # empty state = 0
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


class Agent(object):

    # epsilon-greedy strategy
    def __init__(self, eps=0.1, gamma=0.5):
        self.eps = eps  # probability of choosing random action insted of greedy
        self.gamma = gamma
        self.verbose = False
        self.state_history = []

    def setV(self, V):
        self.V = V

    def set_symbol(self, symbol):
        self.sym = symbol

    def set_verbose(self, v):
        self.verbose = v

    def take_action(self, env):
        # choose an action based on epsilon-greedy strategy
        random_choice = np.random.rand()
        best_state = None
        if random_choice < self.eps:
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
            next_move = possible_moves[idx]
        # greedy part
        else:
            pos2value = {}  # for debugging
            next_move = None
            best_value = -1
            for i in range(LENGTH):
                for j in range(LENGTH):
                    if env.is_empty(i, j):
                        env.board[i, j] = self.sym
                        state = env.get_state()
                        env.board[i, j] = 0
                        pos2value[(i, j)] = self.V[state]
                        if self.V[state] > best_value:
                            best_value = self.V[state]
                            best_state = state
                            next_move = (i, j)

            # if verbose, draw the board w/ the value
            if self.verbose:
                print("Taking a greedy action")
                for i in range(LENGTH):
                    print("------------------")
                    for j in range(LENGTH):
                        if env.is_empty(i, j):
                            # print the value
                            print(" %.2f|" % pos2value[(i, j)], end="")
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

        env.board[next_move[0], next_move[1]] = self.sym

    def update(self, env):
        # we want to BAKCTRACK over the states, so that:
        #  V(prev_state) = V(prev_state) + gamma*(V(next_state) - V(prev_state))
        reward = env.reward(self.sym)
        target = reward
        for prev in reversed(self.state_history):
            value = self.V[prev] + self.gamma * (target - self.V[prev])
            self.V[prev] = value
            target = value
        self.state_history = []


class Human:
    def __init__(self):
        self.state_history = []

    def set_symbol(self, sym):
        self.sym = sym

    def take_action(self, env):
        while True:
            # break if we make a legal move
            n = input("Please enter case's number for your next move (1..9): ")
            i, j = (int(n) - 1) // 3, (int(n) - 1) % 3
            if env.is_empty(i, j):
                env.board[i, j] = self.sym
                break

    def update(self, env):
        pass

def play_game(p1, p2, env, draw=False):
    # loops until the game is over
    current_player = None
    #current_player = p1
    while not env.game_over():
        if current_player == p1:
            current_player = p2
        else:
            current_player = p1

        if draw:
            if draw == 1 and current_player == p1:
                env.draw_board()
            if draw == 2 and current_player == p2:
                env.draw_board()

        current_player.take_action(env)
        # update histories
        state = env.get_state()
        p1.state_history.append(state)
        p2.state_history.append(state)

    if draw:
        env.draw_board()
    # update the value function
    p1.update(env)
    p2.update(env)


if __name__ == '__main__':
    
    p1 = Agent()
    p2 = Agent()

    #  set initial V for p1 and p2
    env = Environment()
    state_winner_triples = get_state_hash_and_winner(env)

    Vx = Initialize_value_x(env, state_winner_triples)
    print(Vx)
    print(len(Vx))
    p1.setV(Vx)
    Vo = Initialize_value_o(env, state_winner_triples)
    p2.setV(Vo)

    # give each player their symbol
    p1.set_symbol(env.x)
    p2.set_symbol(env.o)

    #  train the agents
    epochs = 20000
    for epoch in range(epochs):
         if epoch % 2000 == 0:
             print("%i/%i (%.2f%%)" %(epoch, epochs, epoch/epochs *100))
         play_game(p1, p2, Environment())
    print("fnish")
    human = Human()
    human.set_symbol(env.o)
    while True:
    #for epoch in range(2):
        p1.set_verbose(True)
        play_game(p1, human, Environment(), draw=2)
        answer = input("play again? [Y/n]: ")
        if answer and answer.lower()[0] == 'n':
            break
