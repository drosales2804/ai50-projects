import math
import random
import time


class Nim:

    def __init__(self, initial=[1, 3, 5, 7]):
        """
        Sets up the Nim game. Basic stuff:
            - `piles`: List that says how many objs are in each pile.
            - `player`: Whose turn? 0 or 1.
            - `winner`: Nobody yet (None) or 0/1 if someone won.
        """
        self.piles = initial.copy()
        self.player = 0
        self.winner = None

    @classmethod
    def available_actions(cls, piles):
        """
        Returns all possible moves. Example: (i, j):
            - `i`: Which pile.
            - `j`: How many objs to take.
        """
        actions = set()
        for i, pile in enumerate(piles):
            for j in range(1, pile + 1):  # Can take 1 or more objs
                actions.add((i, j))
        return actions

    @classmethod
    def other_player(cls, player):
        """
        Switch player. If 0, make it 1. If 1, make it 0.
        """
        return 0 if player == 1 else 1

    def switch_player(self):
        """
        Change the turn to the other player.
        """
        self.player = Nim.other_player(self.player)

    def move(self, action):
        """
        Make a move `(i, j)`:
            - `i`: Pile number.
            - `j`: How many objs to remove.
        Checks if the move is valid or not, then updates piles.
        """
        pile, count = action

        # Check invalid stuff
        if self.winner is not None:
            raise Exception("Game already won")
        elif pile < 0 or pile >= len(self.piles):
            raise Exception("Invalid pile")
        elif count < 1 or count > self.piles[pile]:
            raise Exception("Invalid number of objs")

        # Make the move
        self.piles[pile] -= count
        self.switch_player()

        # Check if the game is done
        if all(pile == 0 for pile in self.piles):
            self.winner = self.player


class NimAI:

    def __init__(self, alpha=0.5, epsilon=0.1):
        """
        AI setup:
            - `q`: Dict for storing Q-values (state-action pairs).
            - `alpha`: Learning speed. Higher = learn faster.
            - `epsilon`: How often AI does random moves.
        """
        self.q = dict()
        self.alpha = alpha
        self.epsilon = epsilon

    def update(self, old_state, action, new_state, reward):
        """
        Update Q-values with this formula:
            Q(s, a) = old value + alpha * (reward + future rewards - old value)
        """
        old = self.get_q_value(old_state, action)
        best_future = self.best_future_reward(new_state)
        self.update_q_value(old_state, action, old, reward, best_future)

    def get_q_value(self, state, action):
        """
        Look up Q-value for `state` and `action`. Return 0 if not found.
        """
        return self.q.get((tuple(state), action), 0)

    def update_q_value(self, state, action, old_q, reward, future_rewards):
        """
        Use Q-learning formula to adjust the Q-value:
            Q(s, a) = old Q + alpha * [(reward + future) - old Q]
        """
        self.q[(tuple(state), action)] = old_q + self.alpha * (
            (reward + future_rewards) - old_q
        )

    def best_future_reward(self, state):
        """
        Find the max Q-value for all moves in this state.
        If no moves, just return 0.
        """
        actions = Nim.available_actions(state)
        if not actions:
            return 0
        return max(
            self.q.get((tuple(state), action), 0) for action in actions
        )

    def choose_action(self, state, epsilon=True):
        """
        Pick a move:
            - If `epsilon` is False: Pick the move with the best Q-value.
            - If `epsilon` is True: Sometimes pick random moves (for exploring).
        """
        actions = Nim.available_actions(state)
        if epsilon and random.random() < self.epsilon:
            return random.choice(list(actions))
        return max(
            actions, key=lambda action: self.q.get((tuple(state), action), 0)
        )


def train(n):
    """
    Train AI by making it play against itself `n` times.
    Returns the trained AI model.
    """
    player = NimAI()

    for i in range(n):
        print(f"Playing training game {i + 1}")
        game = Nim()
        last = {0: {"state": None, "action": None}, 1: {"state": None, "action": None}}

        while True:
            state = game.piles.copy()
            action = player.choose_action(state)
            last[game.player]["state"] = state
            last[game.player]["action"] = action

            game.move(action)
            new_state = game.piles.copy()

            if game.winner is not None:
                player.update(state, action, new_state, -1)
                player.update(
                    last[game.player]["state"],
                    last[game.player]["action"],
                    new_state,
                    1,
                )
                break
            elif last[game.player]["state"] is not None:
                player.update(
                    last[game.player]["state"],
                    last[game.player]["action"],
                    new_state,
                    0,
                )

    print("Done training")
    return player


def play(ai, human_player=None):
    """
    Play a game against the AI.
    You can go first (0) or second (1). Random if not set.
    """
    if human_player is None:
        human_player = random.randint(0, 1)

    game = Nim()

    while True:
        print("\nPiles:")
        for i, pile in enumerate(game.piles):
            print(f"Pile {i}: {pile}")

        available_actions = Nim.available_actions(game.piles)
        time.sleep(1)

        if game.player == human_player:
            print("Your Turn")
            while True:
                try:
                    pile = int(input("Choose Pile: "))
                    count = int(input("Choose Count: "))
                    if (pile, count) in available_actions:
                        break
                except ValueError:
                    pass
                print("Invalid move, try again.")
        else:
            print("AI's Turn")
            pile, count = ai.choose_action(game.piles, epsilon=False)
            print(f"AI chose to take {count} from pile {pile}.")

        game.move((pile, count))

        if game.winner is not None:
            print("\nGAME OVER")
            winner = "Human" if game.winner == human_player else "AI"
            print(f"Winner is {winner}")
            return
