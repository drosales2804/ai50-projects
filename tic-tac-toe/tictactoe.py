"""
Tic Tac Toe Player
"""

import math
import copy
import random


X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    [EMPTY, EMPTY, EMPTY]
    ["X", "O", "X"]
    """
    return [[EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    x = 0
    o = 0

    for i in range(3):
        for j in range(3):
            if board[i][j] == "X":
                x += 1
            elif board[i][j] == "O":
                o += 1

    if (x == 0 and o == 0) or (x - o == 0):
        return "X"
    else:
        return "O"


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.

    """
    if terminal(board) is False:
        act = set()
        for i in range(3):
            for j in range(3):
                if board[i][j] == None:
                    act.add((i, j))
        return act


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    if (
        (board[action[0]][action[1]] != None)
        or (action[0] not in range(3))
        or (action[1] not in range(3))
    ):
        raise Exception("Invalid action")
    newboard = copy.deepcopy(board)
    newboard[action[0]][action[1]] = player(board)
    return newboard


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    for row in board:
        if row[0] == row[1] == row[2]:
            return row[0]

    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col]:
            return board[0][col]

    if board[0][0] == board[1][1] == board[2][2]:
        return board[0][0]

    if board[0][2] == board[1][1] == board[2][0]:
        return board[0][2]

    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """

    if (winner(board) == "X") or (winner(board) == "O"):
        return True

    for i in range(3):
        for j in range(3):
            if board[i][j] == None:
                return False

    return True


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    champion = winner(board)

    if champion == "X":
        return 1
    elif champion is None:
        return 0
    elif champion == "O":
        return -1


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    alpha = float("-inf")
    beta = float("inf")

    if terminal(board) is False:
        pl = player(board)

        if pl == "X":
            amax = {}
            for act in actions(board):
                amax[act] = minvalue(result(board, act), alpha, beta)
            maxrnd = max(amax.values())
            return random.choice([k for k in amax if amax[k] == maxrnd])
        else:
            amin = {}
            for act in actions(board):
                amin[act] = maxvalue(result(board, act), alpha, beta)
            minrnd = min(amin.values())
            return random.choice([k for k in amin if amin[k] == minrnd])
    else:
        return None


def minvalue(board, alpha, beta):
    v = float("inf")
    if terminal(board) is True:
        return utility(board)
    else:
        for act in actions(board):
            eval = maxvalue(result(board, act), alpha, beta)
            v = min(v, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return v


def maxvalue(board, alpha, beta):
    v = float("-inf")
    if terminal(board) is True:
        return utility(board)
    else:
        for act in actions(board):
            eval = minvalue(result(board, act), alpha, beta)
            v = max(v, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return v

