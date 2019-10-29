import numpy as np
import numba
import sys
import re
import os
from games import TicTacToe, Connect4, Draughts
from node import Node
import argparse
import tqdm


class InputError(Exception):
    pass


class MonteCarloTreeSearch(object):
    def __init__(self, game):
        self.game = game
        self.iterations = 100
        self.hash_table = {}
        self.print_path = False

    def calculate_UCB(self, node):
        '''
        Calculate the UCB score for a node
        '''
        if not node.expanded:
            return np.inf

        if node.visits == 0:
            return np.inf
        else:
            ln_N = np.log(node.parent.visits)

        return node.get_score() + (np.sqrt((2 * ln_N) / node.visits))

    def index_max(self, values):
        '''
        Return the index of the largest value in an array. Select at random if
        draw.
        '''
        max_val = max(values)
        max_indices = [ind for ind, val in enumerate(values) if val == max_val]
        return np.random.choice(max_indices)

    def get_highest_UCB(self, node):
        '''
        Return the child with the highest UCB from a node
        '''
        UCBs = []
        for child in node.children:
            child.UCB = self.calculate_UCB(child)
            UCBs.append(child.UCB)
        try:
            max_index = self.index_max(UCBs)
        except:
            print(UCBs)
            raise ValueError

        return node.children[max_index]

    def backpropogate(self, path, winner):
        '''
        Backpropogate the result of the simulation along the path taken to the leaf node
        '''
        for node in path:
            node.wins += winner * node.player * -1
            node.visits += 1

    def simulate(self, root_node):
        '''
        Traverse the tree, following the path of largest UCB until a leaf not is reached.
        Simulate a random game from that leaf node and backpropagate the result back up the tree
        '''
        node = root_node
        path = [node]  # record the nodes that are traversed

        #follow the nodes with the highest UCB until a node that is un-expanded is reached
        while node.expanded and not node.end_state:
            node = self.get_highest_UCB(node)
            path.append(node)

        self.expand(node)
        winner = self.play_out(node)
        self.backpropogate(path, winner)

    def expand(self, node):
        '''
        Generates a new node for each possible move from the current node
        '''
        moves = self.game.get_moves(node.board, node.player)
        next_player = self.game.get_next_player(node.player)
        for move in moves:
            node.children.append(Node(move, next_player, parent=node))

        node.expanded = True
        return node

    def get_next_move(self, board, player=1):
        '''
        Explore the possiblity tree to generate the next move for the machine
        '''

        # If a node is already in the tree, select it, otherwise create a new node
        if str(board) in self.hash_table:
            root_node = self.hash_table[str(board)]
        else:
            root_node = Node(board, player)
            self.hash_table[root_node.hash] = root_node

        # Follow the path of highest UCB to a leaf, expand it, and simulate
        # a random game from that point
        for iterations in tqdm.tqdm(range(self.iterations)):
            self.simulate(root_node)

        # Look through the scores for each possible next move and select
        # the one with highest score
        scores = [child.get_score() for child in root_node.children]
        best_index = self.index_max(scores)

        return root_node.children[best_index].board, root_node

    def game_not_finished(self, board):
        return np.isnan(self.game.get_winner(board))

    def play_out(self, node, print_path=False):
        '''
        Play out a given board state - selecting moves at random until the game ends
        '''
        if np.isnan(node.end_state):
            node.winner = self.game.get_winner(node.board)
            node.end_state = not np.isnan(node.winner)

        if node.end_state:
            return node.winner
        else:
            player = node.player
            board = node.board
            moves = [np.nan]

            # Choose moves at random until a game is won
            while self.game_not_finished(board) and len(moves) > 0:
                moves = self.game.get_moves(board, player)
                if not moves:
                    moves = [board]
                board_index = np.random.randint(len(moves))
                board = moves[board_index]
                player = self.game.get_next_player(player)
                # Allow printing of path for debug
                if self.print_path:
                    self.game.display_board(board)
                    print()
            return self.game.get_winner(board)


class Interface(object):
    """
    Allow a human to interact with the AI in the form of a game
    """

    def __init__(self, game):
        self.board = game.start_state()
        self.game = game
        self.searcher = MonteCarloTreeSearch(game)
        self.show_probabilities = True

    def print_path(self, print_path):
        ''''''
        self.searcher.print_path = print_path

    def set_iterations(self, iters):
        '''
        Set the number of simulations that the MCTS wiill run before sleecting the best move
        '''
        self.searcher.iterations = iters

    def human_go(self):
        '''
        Let a human enter a move
        '''
        self.board = self.game.human_go(self.board)

    def machine_go(self):
        '''
        Use MCTS to generate the next move for the machine
        '''
        print('Machine thinking...')
        self.board, node = self.searcher.get_next_move(self.board)

        for child in node.children:
            if self.show_probabilities:
                print('{}/{} = {}'.format(child.wins, child.visits,
                                          child.get_score()))
                print(child.board)
                print()

    def run_game(self):
        '''
        Allow the human and machine to play agianst each other until one of them wins
        '''
        player = 1
        while np.isnan(self.game.get_winner(self.board)):
            if player == 1:
                self.machine_go()
            elif player == -1:
                self.human_go()
            player *= -1
            self.game.display_board(self.board)
        if self.game.get_winner(self.board) == 0:
            print('Draw, Everyone lost')
        elif self.game.get_winner(self.board) == 1:
            print('Machine Won! All hail our new robot overlords')
        elif self.game.get_winner(self.board) == -1:
            print('You won! Humanity is safe, for the moment...')

        _ = input('Press Enter to play again or Ctrl+C to quit')
        os.system('cls')


if __name__ == '__main__':
    # Get game to play
    try:
        game = sys.argv[1]
        if game == 'tictactoe':
            game_rules = TicTacToe()
        elif game == 'connect4':
            game_rules = Connect4()
        elif game == 'draughts':
            game_rules = Draughts()
    except:
        game_rules = TicTacToe()

    # Decide whether to show probabilities from argument
    interface = Interface(game_rules)
    try:
        show_probs = sys.argv[2]
        if show_probs == 'show_probs':
            interface.show_probabilities = True
        else:
            interface.show_probabilities = False
    except:
        interface.show_probabilities = False

    print('Welcome to the Monte-Carlo')
    print('tree search AI')
    print('AI plays first')
    print()
    while True:
        interface.set_iterations(5000)
        interface.print_path(False)
        interface.run_game()
