import math
import random
import time


class Nim:

    def __init__(self, initial=[1, 3, 5, 7]):
        """
        Initialize the Nim game board with default or custom piles.
        The game keeps track of:
            - `piles`: List of integers, where each value represents the number of objects in each pile.
            - `player`: Integer, 0 or 1, indicating the current player's turn.
            - `winner`: None (if the game is ongoing), 0 (if player 0 wins), or 1 (if player 1 wins).
        """
        self.piles = initial.copy()
        self.player = 0
        self.winner = None

    @classmethod
    def available_actions(cls, piles):
        """
        Return all possible actions for the current state of the game.
        An action is represented as a tuple `(i, j)`:
            - `i`: The pile index (0-based).
            - `j`: The number of objects to remove from that pile.
        """
        actions = set()
        for i, pile in enumerate(piles):
            for j in range(1, pile + 1):  # Can take at least 1 and up to all objects from a pile
                actions.add((i, j))
        return actions

    @classmethod
    def other_player(cls, player):
        """
        Return the opponent of the given player (0 or 1).
        """
        return 0 if player == 1 else 1

    def switch_player(self):
        """
        Change the current turn to the other player.
        """
        self.player = Nim.other_player(self.player)

    def move(self, action):
        """
        Perform the action `(i, j)`:
            - `i`: Pile index.
            - `j`: Number of objects to remove.
        Raises exceptions for invalid actions or if the game has already ended.
        Updates the pile and switches the turn to the next player.
        """
        pile, count = action

        # Validate move
        if self.winner is not None:
            raise Exception("Game already won")
        elif pile < 0 or pile >= len(self.piles):
            raise Exception("Invalid pile")
        elif count < 1 or count > self.piles[pile]:
            raise Exception("Invalid number of objects")

        # Execute the move
        self.piles[pile] -= count
        self.switch_player()

        # Check if the game is over
        if all(pile == 0 for pile in self.piles):
            self.winner = self.player


class NimAI:

    def __init__(self, alpha=0.5, epsilon=0.1):
        """
        Initialize the AI with:
            - `q`: Dictionary storing Q-values for state-action pairs.
            - `alpha`: Learning rate, controls how much new knowledge overrides existing Q-values.
            - `epsilon`: Exploration rate, determines how often the AI explores random actions.
        """
        self.q = dict()
        self.alpha = alpha
        self.epsilon = epsilon

    def update(self, old_state, action, new_state, reward):
        """
        Update Q-values using the Q-learning formula:
            Q(s, a) <- Q(s, a) + alpha * [(reward + future rewards) - Q(s, a)]
        """
        old = self.get_q_value(old_state, action)
        best_future = self.best_future_reward(new_state)
        self.update_q_value(old_state, action, old, reward, best_future)

    def get_q_value(self, state, action):
        """
        Retrieve the Q-value for a given state-action pair.
        Returns 0 if the Q-value is not already stored.
        """
        return self.q.get((tuple(state), action), 0)

    def update_q_value(self, state, action, old_q, reward, future_rewards):
        """
        Update the Q-value for a state-action pair using:
            Q(s, a) <- old_q + alpha * [(reward + future_rewards) - old_q]
        """
        self.q[(tuple(state), action)] = old_q + self.alpha * (
            (reward + future_rewards) - old_q
        )

    def best_future_reward(self, state):
        """
        Calculate the maximum Q-value for any action in a given state.
        Returns 0 if no actions are available.
        """
        actions = Nim.available_actions(state)
        if not actions:
            return 0
        return max(
            self.q.get((tuple(state), action), 0) for action in actions
        )

    def choose_action(self, state, epsilon=True):
        """
        Select an action based on the current state:
            - If `epsilon` is False: Choose the action with the highest Q-value.
            - If `epsilon` is True: With probability `self.epsilon`, choose a random action; otherwise, choose the best action.
        """
        actions = Nim.available_actions(state)
        if epsilon and random.random() < self.epsilon:
            return random.choice(list(actions))
        return max(
            actions, key=lambda action: self.q.get((tuple(state), action), 0)
        )


def train(n):
    """
    Train the AI by simulating `n` games of Nim where the AI plays against itself.
    Returns the trained AI.
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
    Allow a human to play against the trained AI.
    Human player can be set to move first (0) or second (1). Defaults to random order.
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
