import numpy as np
from node import Node
import sys
import numba
import re


class TicTacToe(object):
    """
    A class of functions that can be used to generate next moves from a node and test for a win.
    """

    def __init__(self):
        self.board_width = 3
        self.player_ids = [-1, 1]
        self.max_moves = 10

    def get_next_player(self, player):
        '''
        gets the Id of the next player
        '''
        return player * -1

    def get_winner(self, board):
        '''
        If a board has a winner, the index of the winner is returned.
        Returns -1 if loss, 1 if win, 0 if draw and nan if game is not finished
        '''
        for p_id in self.player_ids:
            win_array = np.array([p_id] * self.board_width, dtype=np.int8)
            for i in range(self.board_width):
                # check rows
                if np.array_equal(board[i], win_array):
                    return p_id
                #check columns
                elif np.array_equal(board[:, i], win_array):
                    return p_id
                # check leading diagonal
                elif np.array_equal(np.diagonal(board), win_array):
                    return p_id
                # check non-leading diagonal
                elif np.array_equal(np.diagonal(np.flipud(board)), win_array):
                    return p_id
        # return nan if no wins losses or draws
        for i in np.nditer(board):
            if i == 0:
                return np.nan
        # must be a draw so return 0
        return 0

    def get_moves(self, board, player):
        '''
        Return a list of all possible moves from a given board
        '''
        moves = []
        for x in range(self.board_width):
            for y in range(self.board_width):
                if board[x][y] == 0:
                    copy = board.copy()
                    copy[x][y] = player
                    moves.append(copy)
        return moves

    def display_board(self, board):
        '''
        Nicely display the current board
        '''
        print('  0 1 2')
        for x, row in enumerate(board):
            sys.stdout.write(str(x))
            for val in row:
                if val == 1:
                    sys.stdout.write('|X')
                elif val == -1:
                    sys.stdout.write('|O')
                else:
                    sys.stdout.write('| ')
            print('|')

    def human_go(self, board):
        '''
        Allow a human to take a turn
        '''
        coord_pattern = re.compile('[0-{}],[0-{}]'.format(
            board.shape[0], board.shape[1]))
        print('Enter Coordinates of your go then press enter.')
        input_str = input('(space seperated, 0-2 with origin in top left)\n')

        if not coord_pattern.match(input_str):
            print('That is not in the right format, please try again...')
            return self.human_go(board)
        else:
            y, x = [int(coord) for coord in input_str.split(',')]
        if board[x][y] != 0:
            print('That square is already taken, please try again')
            self.human_go()
        else:
            board[x][y] = -1
            return board

    def start_state(self):
        '''
        Returns the inital board state (an empty 3 * 3 grid)
        '''
        return np.zeros((3, 3), dtype=np.int8)


class Connect4(object):
    """
    A class of functions that can be used to generate next moves from a node and test for a win.
    """

    def __init__(self):
        self.board_size = np.array((6, 7))
        self.player_ids = np.array([-1, 1])
        self.max_moves = (6 * 7) + 1

    def get_next_player(self, player):
        '''
        gets the Id of the next player
        '''
        return player * -1

    def human_go(self, board):
        '''
        Allow a human to take a turn
        '''
        coord_pattern = re.compile('[0-{}]$'.format(board.shape[1]))
        print('Enter Column and press enter.')
        input_str = input('(from 0-6)\n')
        if not coord_pattern.match(input_str):
            print('That is not in the right format, please try again...')
            return self.human_go(board)
        else:
            col = int(input_str)
        if board[0][col] != 0:
            print('That column is already full, please try again')
            self.human_go()
        else:
            for row in board[::-1]:
                if row[col] == 0:
                    row[col] = -1
                    return board

    @staticmethod
    @numba.jit()
    def get_winner_c(board, player_ids, board_size):
        '''
        Uses numba to quickly get the winner
        '''
        for p_id in player_ids:
            width, height = board_size
            # Check vertical lines
            for y in range(height):
                for x in range(width - 3):
                    if (board[x][y] == p_id and board[x + 1][y] == p_id
                            and board[x + 2][y] == p_id
                            and board[x + 3][y] == p_id):
                        return p_id

            # Check horizontal lines
            for y in range(height - 3):
                for x in range(width):
                    if (board[x][y] == p_id and board[x][y + 1] == p_id
                            and board[x][y + 2] == p_id
                            and board[x][y + 3] == p_id):
                        return p_id

            # Check leading diagonals
            for y in range(height - 3):
                for x in range(width - 3):
                    if (board[x][y] == p_id and board[x + 1][y + 1] == p_id
                            and board[x + 2][y + 2] == p_id
                            and board[x + 3][y + 3] == p_id):
                        return p_id

            # Check non-leading diagonals
            for y in range(height - 3):
                for x in range(3, width):
                    if (board[x][y] == p_id and board[x - 1][y + 1] == p_id
                            and board[x - 2][y + 2] == p_id
                            and board[x - 3][y + 3] == p_id):
                        return p_id
        # return nan if no wins losses or draws and some cells contain 0
        for i in np.nditer(board):
            if i == 0:
                return np.nan
        # must be a draw so return 0
        return 0

    @staticmethod
    @numba.jit()
    def get_moves_c(board, player, height, width):
        '''
        Return a list of all possible moves from a given board
        '''
        moves = []
        for x in range(width):
            copy = board.copy()
            for y in range(height):
                if board[height - y - 1][x] == 0:
                    copy[height - y - 1][x] = player
                    moves.append(copy)
                    break
        return moves

    def get_winner(self, board):
        '''
        If a board has a winner, the index of the winner is returned.
        Returns -1 if loss, 1 if win, 0 if draw and nan if game is not finished
        '''
        ids = self.player_ids
        board_size = self.board_size
        return self.get_winner_c(board, ids, board_size)

    def get_moves(self, board, player):
        '''
        Return a list of all possible moves from a given board
        '''
        width, height = self.board_size
        return self.get_moves_c(board, player, width, height)

    def display_board(self, board):
        '''
        Nicely print the current board in a form that humans can understand
        '''
        width = self.board_size[1]
        top_row_index = ' '.join([str(i) for i in range(width)])
        print('  {}'.format(top_row_index))
        for x, row in enumerate(board):
            sys.stdout.write(str(x))
            for val in row:
                if val == 1:
                    sys.stdout.write('|X')
                elif val == -1:
                    sys.stdout.write('|O')
                else:
                    sys.stdout.write('| ')
            print('|')

    def start_state(self):
        '''
        Returns the inital board state (an empty 3 * 3 grid)
        '''
        return np.zeros(self.board_size, dtype=np.int8)


class Draughts(object):
    """
    A class of functions that can be used to generate next moves from a node
    and test for a win. The rules of this game ore the rules of draughts.
    -1 is human, -2 is human king, 1 is ai, 2 is ai king
    """

    def __init__(self):
        self.board_size = 8, 8
        self.player_ids = [-1, 1]
        self.max_moves = 256

    def get_next_player(self, player):
        '''
        gets the Id of the next player
        '''
        return player * -1

    def get_winner(self, board):
        '''
        If a board has a winner, the index of the winner is returned.
        Returns -1 if loss, 1 if win, 0 if draw and nan if game is not finished
        '''
        human_pieces_left = False
        ai_pieces_left = False

        # Set draw criteria
        if board.moves_since_king > 80:
            return 0
        if board.moves_since_capture > 80:
            return 0
        if board.moves_since_start > 512:
            return 0

        for val in np.nditer(board):
            if val > 0:
                human_pieces_left = True
            elif val < 0:
                ai_pieces_left = True
            if ai_pieces_left and human_pieces_left:
                return np.nan
        # if human won
        if not ai_pieces_left:
            return 1
        # If ai won
        if not human_pieces_left:
            return -1

    @staticmethod
    @numba.jit()
    def check_bounds(x, y):
        '''
        Return the allowed search area around a piece.
        Ensures that areas outside the board arent searched
        '''
        x_range = []
        y_range = []
        width = 7
        if x > 0:
            x_range.append(x - 1)

        if y > 0:
            y_range.append(y - 1)

        if x < width:
            x_range.append(x + 1)

        if y < width:
            y_range.append(y + 1)

        return x_range, y_range

    def search_local_area(self, board, x, y):
        '''
        search the local area around a piece to generate
        legal moves for that piece
        '''
        # Ensure we aren't searching outside the board
        x_range, y_range = self.check_bounds(x, y)
        moves = []
        capture_moves = []
        current_piece = board[y][x]

        for x_index in x_range:
            for y_index in y_range:
                # if empty square found
                if board[y_index][x_index] == 0:
                    # Let kings move in any direction
                    if current_piece in [2, -2]:
                        move = self.move_piece(board, x, y, x_index, y_index)
                        moves.append(move)
                    #let normal pieces only move 'forward'
                    elif current_piece in [
                            1, -1
                    ] and current_piece == (y - y_index):
                        move = self.move_piece(board, x, y, x_index, y_index)
                        moves.append(move)
                # If opposing piece in range see if it can be captured
                elif self.are_different_teams(board[y_index][x_index],
                                              current_piece):
                    capture = self.capture_piece(board, x, y, x_index, y_index)
                    if capture:
                        capture_moves.extend(capture)
        # If a piece can be captured, force the capture move
        if capture_moves:
            return capture_moves
        else:
            return moves

    def move_piece(self, board, start_x, start_y, end_x, end_y):
        '''
        A helper function to move a piece from start_x, start_y to end_x, end_y.
        Also converts pieces into kings if they hit the opposing boundaries
        '''
        copy = board.copy()
        # Convert a piece to king if it reaches the other side
        if copy[start_y][start_x] == 1 and end_y == 0:
            copy[end_y][end_x] = 2
            copy.moves_since_king = 0
        elif copy[start_y][start_x] == -1 and end_y == 7:
            copy[end_y][end_x] = -2
            copy.moves_since_king = 0
        # Otherwise move the piece
        else:
            copy[end_y][end_x] = copy[start_y][start_x]
            copy[start_y][start_x] = 0
        return copy

    @staticmethod
    @numba.jit()
    def capture_is_legal(board, after_x, after_y):
        '''
        Detect whether a given capture is legal
        '''
        # Check whether the attack was legal, if it was, return True
        return 0 <= after_x <= 7 and 0 <= after_y <= 7 and board[after_y][
            after_x] == 0

    def inside_board(self, val):
        '''
        Check whether a given index is inside the range of the baord
        '''
        return 0 <= val <= 7

    def are_different_teams(self, piece_1, piece_2):
        '''
        Test two pieces to find whether they are on different teams
        '''
        return piece_1 * piece_2 < 0

    def possible_captures(self, board, x, y):
        '''
        Search the area around a location to see if any captures are possible
        '''
        piece = board[x][y]
        possible_captures = []

        x_range, y_range = self.check_bounds(x, y)

        if piece in [-1, 1]:
            for x in x_range:
                if self.inside_board(piece + y) and self.are_different_teams(
                        board[y + piece][x] * piece):
                    possible_captures.append([x, y + piece])
        else:
            for x in x_range:
                for y in y_range:
                    if self.are_different_teams(board[y][x], piece):
                        possible_captures.append([x, y])
        return possible_captures

    def capture_piece(self,
                      board,
                      attack_x,
                      attack_y,
                      defend_x,
                      defend_y,
                      moves=[]):
        '''
        Return the board state after the piece at attack_x, attack_y
        takes the piece at defend_x, defend_y. Return False if the
        move is blocked
        '''
        # Calculate the position of the piece after the attack
        # Formula comes from rearranging aft = atk + 2(def-atk)
        after_x = defend_x + defend_x - attack_x
        after_y = defend_y + defend_y - attack_y

        if self.capture_is_legal(board, after_x, after_y):
            move = self.move_piece(board, attack_x, attack_y, after_x, after_y)
            move[defend_y][defend_x] = 0
            move.moves_since_capture = 0
            moves.append(move)

            # See if a multiple capture is possible
            for capture in self.possible_captures(move, after_x, after_y):
                self.capture_piece(move, after_x, after_y, capture[0],
                                   capture[1], moves)
            return moves
        else:
            return False

    def get_moves(self, board, player):
        '''
        Return a list of possible moves from the current position.
        If the b
        '''
        moves = []
        for x in range(8):
            for y in range(8):
                # if selected square contains one of the current
                # player's pieces
                if not self.are_different_teams(board[y][x], player):
                    moves.extend(self.search_local_area(board, x, y))
        # Increment move counter
        for move in moves:
            move.moves_since_start += 1
        return moves

    def display_board(self, board):
        '''
        Nicely display the current board
        '''
        # print x axis labels
        sys.stdout.write('  ')
        for y in range(board.shape[0]):
            sys.stdout.write('{} '.format(y))
        print()

        # print the actual board
        for x, row in enumerate(board):
            sys.stdout.write(str(x))
            for val in row:
                if val == 1:
                    sys.stdout.write('|w')
                elif val == 2:
                    sys.stdout.write('|W')  # king white
                elif val == -1:
                    sys.stdout.write('|b')
                elif val == -2:
                    sys.stdout.write('|B')  # king black
                else:
                    sys.stdout.write('| ')
            print('|')

    def human_go(self, board):
        '''
        Allow a human to take a turn
        '''
        coord_pattern = re.compile('[0-{}],[0-{}]-[0-{}],[0-{}]'.format(
            board.shape[0], board.shape[1], board.shape[0], board.shape[1]))
        print('Enter Coordinates of your go then press enter.')
        input_str = input(
            '(space seperated, 0-{} with origin in top left)\n'.format(
                board.shape[0]))

        if not coord_pattern.match(input_str):
            print('That is not in the right format, please try again...')
            return self.human_go(board)
        else:
            y, x = [int(coord) for coord in input_str.split(',')]
        if board[x][y] != 0:
            print('That square is already taken, please try again')
            self.human_go()
        else:
            board[x][y] = -1
            return board

    def start_state(self):
        '''
        Returns the inital board state
        '''
        board = np.array(
            [[0, -1, 0, -1, 0, -1, 0, -1], [-1, 0, -1, 0, -1, 0, -1, 0],
             [0, -1, 0, -1, 0, -1, 0, -1], [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 1, 0, 1, 0, 1, 0],
             [0, 1, 0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0, 1, 0]],
            dtype=np.int8)
        # Add game state attributes to array
        board = DraughtsBoard(board)
        return board


class DraughtsBoard(np.ndarray):
    """
    Modify the standard numpy array to contain some new, draughts
    specific, sattributes:
     - moves since start
     - moves since capture
     - moves since king
    """

    def __new__(cls, input_array, info=None):
        # Convert input_array into a DraughtsBoard instance
        obj = np.asarray(input_array).view(cls)
        # add the new attributes to the created instance
        obj.moves_since_start = 0
        obj.moves_since_capture = 0
        obj.moves_since_king = 0
        # Finally, return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        # Add the extra attributes to the object
        self.moves_since_start = getattr(obj, 'moves_since_start', None)
        self.moves_since_capture = getattr(obj, 'moves_since_capture', None)
        self.moves_since_king = getattr(obj, 'moves_since_king', None)
