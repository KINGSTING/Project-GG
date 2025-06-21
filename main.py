import random
import math


class Game:
    def __init__(self):
        # Define slots and pieces
        self.slot_code = [f"{row}{col}" for row in "ABCDEFGH" for col in range(1, 10)]
        self.squares = ['[---]' for _ in range(72)]
        self.record_squares = ['[---]' for _ in range(72)]
        self.pieces = (
                ['5SG', '4SG', '3SG', '2SG', '1SG', 'COL', 'LTC', 'MAJ', 'CPT',
                 '1LT', '2LT', 'SRG'] +
                ['PVT'] * 6 +
                ['SPY'] * 2 + ['FLG']
        )

        # Define ranks for pieces
        self.rank_value = {
            'PVT': 2,  # Private
            'SRG': 3,  # Sergeant
            '1LT': 4,  # First Lieutenant
            '2LT': 5,  # Second Lieutenant
            'CPT': 6,  # Captain
            'MAJ': 7,  # Major
            'COL': 8,  # Colonel
            'LTC': 9,  # Lieutenant Colonel
            '1SG': 10,  # 1 Star General
            '2SG': 11,  # 2 Star General
            '3SG': 12,  # 3 Star General
            '4SG': 13,  # 4 Star General
            '5SG': 14,  # 5 Star General
            'SPY': 15,  # Spy
            'FLG': 1  # Flag
        }

        self.player2_setup()
        self.player1_setup()

    def player1_setup(self):
        available_pieces = self.pieces.copy()
        random.shuffle(available_pieces)

        # Available slots for player 1 (A1-C9)
        player1_slots = [f"{row}{col}" for row in "ABC" for col in range(1, 10)]
        random.shuffle(player1_slots)

        for piece in available_pieces:
            if player1_slots:
                slot = player1_slots.pop()  # Get a random available slot
                conv_slot = self.slot_code.index(slot)
                self.squares[conv_slot] = f"[1_{piece}]"  # Prefix player 1 to each piece
            else:
                break

        # Update the board view
        self.update()
        self.player_move()

    def player2_setup(self):
        available_pieces = self.pieces.copy()
        random.shuffle(available_pieces)

        # Available slots for player 2 (F1-H9)
        player2_slots = [f"{row}{col}" for row in "FGH" for col in range(1, 10)]
        random.shuffle(player2_slots)

        for piece in available_pieces:
            if player2_slots:
                slot = player2_slots.pop()
                conv_slot = self.slot_code.index(slot)
                self.squares[conv_slot] = f"[2_{piece}]"  # Prefix player 2 to each piece

    def update(self):
        print("\nCurrent Board:")
        # Iterate over rows (A to H)
        for i, row in enumerate("ABCDEFGH"):
            # Extract the corresponding row of squares
            print(f"{row} " + " ".join(self.squares[i * 9:(i + 1) * 9]))

        # Print column numbers (1 to 9)
        print("    " + "   ".join(map(str, range(1, 10))) + "\n")

        # Check if either flag exists on the board
        if '[1_FLG]' not in self.squares:
            print("Computer Wins.")
            exit()  # Stop the program
        elif '[2_FLG]' not in self.squares:
            print("Player Wins")
            exit()  # Stop the program
        else:
            print("***************************************************************************************************")

    def capture_piece(self, piece, target, piece_value, target_value):

        # Special rules: Flag captures the opposing flag by moving into its square
        if piece == '[FLG]' and target == '[FLG]':
            print(f"Flag captures the opposing flag by moving into its square!")
            return True  # Flag captures the enemy flag

        # Elimination rules: Flag cannot capture any piece except the opponent's flag
        if piece == '[FLG]':
            print(f"Flag cannot capture any piece except the opponent's flag.")
            return False

        # Elimination rules: Private and Flag can be eliminated by any piece
        if target == '[PVT]' or target == '[FLG]':
            print(f"{piece} eliminates {target}!")
            return True  # Any piece can eliminate a private or flag

        # Elimination based on rank (Piece eliminates target if it has a higher rank)
        if piece_value < target_value:
            print(f"{piece} is eliminated by {target}!")
            return False  # Piece is eliminated

        elif piece_value > target_value:
            if target_value == 0:
                return True
            else:
                print(f"{piece} eliminates {target}")
            return True  # Piece eliminates the target

        # The Spy: Only the Private can eliminate the Spy
        elif target == '[SPY]' and piece == '[PVT]':
            print(f"{piece} eliminates the spy!")
            return True  # Only the Private can eliminate the Spy

        # If none of the above conditions are met, the piece cannot capture the target
        else:
            print(f"{target} cannot be captured by {piece}.")
            return False

    def get_rank(self, piece):
        # Define a dictionary that maps the piece string to the corresponding rank name
        piece_map = {
            "[1_PVT]": 'PVT', "[1_SRG]": 'SRG', "[1_1LT]": '1LT', "[1_2LT]": '2LT',
            "[1_CPT]": 'CPT', "[1_MAJ]": 'MAJ', "[1_COL]": 'COL', "[1_LTC]": 'LTC',
            "[1_1SG]": '1SG', "[1_2SG]": '2SG', "[1_3SG]": '3SG', "[1_4SG]": '4SG',
            "[1_5SG]": '5SG', "[1_SPY]": 'SPY', "[1_FLG]": 'FLG',
            "[2_PVT]": 'PVT', "[2_SRG]": 'SRG', "[2_1LT]": '1LT', "[2_2LT]": '2LT',
            "[2_CPT]": 'CPT', "[2_MAJ]": 'MAJ', "[2_COL]": 'COL', "[2_LTC]": 'LTC',
            "[2_1SG]": '1SG', "[2_2SG]": '2SG', "[2_3SG]": '3SG', "[2_4SG]": '4SG',
            "[2_5SG]": '5SG', "[2_SPY]": 'SPY', "[2_FLG]": 'FLG'
        }

        # Look up the rank in the dictionary based on the piece
        rank = piece_map.get(piece, None)

        if rank is not None:
            return self.rank_value.get(rank, 0)  # Fetch the rank value, default to 0 if not found
        else:
            return 0  # Return 0 if the piece is not found in the dictionary

    def player_move(self):
        while True:
            # Prompt the player to choose a piece to move
            choose = input("Pick the slot with the piece you want to move (e.g., F5): ").strip().upper()

            if choose not in self.slot_code:
                print("Invalid slot. Try again.")
                continue

            # Convert the chosen slot to its index
            from_index = self.slot_code.index(choose)

            # Check if the chosen square contains the player's piece
            if self.squares[from_index] in ['[---]', '[XXX]'] or not self.squares[from_index].startswith("[1_"):
                print("Invalid choice: Empty or opponent's piece. Try again.")
                continue

            # Get the piece type (Flag or other piece)
            piece_rank = self.squares[from_index]

            # Directions for movement
            directions = [9, -9, 1, -1]

            # Remove invalid directions (out of bounds)
            if from_index % 9 == 0:  # Left edge
                directions.remove(-1)
            if (from_index + 1) % 9 == 0:  # Right edge
                directions.remove(1)

            # Get possible move positions
            moves = [from_index + d for d in directions if 0 <= from_index + d < 72]

            # If the piece is a Flag, only allow moves to empty squares or the opponent's flag
            if piece_rank == '[1_FLG]':  # Assuming player's flag is '[1_FLG]'
                available_moves = [self.slot_code[m] for m in moves if self.squares[m] == '[---]' or self.squares[
                    m] == '[2_FLG]']  # Only opponent's flag and empty squares
            else:
                # Filter valid moves: empty squares or opponent's pieces
                available_moves = [self.slot_code[m] for m in moves if self.squares[m] in ['[---]', '[XXX]']]

                # Add opponent pieces if they can be captured
                opponent_pieces = [
                    self.slot_code[m]
                    for m in moves
                    if self.squares[m] not in ['[---]', '[XXX]'] and not self.squares[m].startswith("[1_")
                ]
                available_moves.extend(opponent_pieces)

            print(f"Available moves: {available_moves}")

            # Prompt the player to choose where to move
            move_to = input("Pick a slot to move the piece: ").strip().upper()

            if move_to not in self.slot_code or move_to not in available_moves:
                print("Invalid move. Try again.")
                continue

            # Convert the move-to slot to its index
            to_index = self.slot_code.index(move_to)

            # Get the piece and target ranks
            piece_value = self.get_rank(self.squares[from_index])  # Rank of the piece at from_index
            target_value = self.get_rank(self.squares[to_index])  # Rank of the piece at to_index
            piece_rank = self.squares[from_index]
            target_rank = self.squares[to_index]

            # Print the information
            print(f"Chosen slot: {choose} -> Piece: {self.squares[from_index]}")
            print(f"Target slot: {move_to} -> Piece: {self.squares[to_index]}")
            print(f"Piece rank: {piece_value}")
            print(f"Target rank: {target_value}")

            # Move the piece or capture an opponent's piece
            if self.squares[to_index] == '[---]':  # Move to an empty square
                self.squares[to_index] = self.squares[from_index]
                self.squares[from_index] = '[---]'
            else:  # Capture an opponent's piece
                captured_piece = self.squares[to_index]

                # Move the piece or capture an opponent's piece
                if piece_value == target_value:
                    print(f"{piece_rank} and {target_rank} eliminate each other!")
                    self.squares[from_index] = '[---]'
                    self.squares[to_index] = '[---]'

                # Flag piece special rule: Flag cannot capture any piece except the opponent's flag
                if piece_rank == '[FLG]':
                    if not (target_rank == '[1_FLG]'):  # The Flag can only capture the opponent's Flag
                        print(f"Flag cannot capture any piece except the opponent's flag.")
                        return False  # Do not proceed with the capture or move
                    else:
                        print(f"Flag captures the opponent's flag!")
                        self.squares[to_index] = self.squares[from_index]
                        self.squares[from_index] = '[---]'
                        captured_piece = self.squares[to_index]
                        print(f"Captured: {captured_piece}")
                        return True  # Proceed with flag capturing the opponent's flag

                # If the piece can't capture, both pieces are eliminated
                if not self.capture_piece(piece_rank, target_rank, piece_value, target_value):
                    # Eliminate the lower rank piece and set the square to [---]
                    if piece_value < target_value:
                        self.squares[from_index] = '[---]'  # Lower rank piece vanishes
                    else:
                        self.squares[to_index] = '[---]'  # Lower rank piece on target vanishes
                else:
                    # Successful capture: Move the attacking piece to the target square
                    captured_piece = self.squares[to_index]
                    print(f"Captured: {captured_piece}")
                    self.squares[to_index] = self.squares[from_index]
                    self.squares[from_index] = '[---]'

            self.comp_move()  # Pass the turn to the computer
            break

    def comp_move(self):
        def heuristic_evaluation(board, is_computer_turn):
            """
            Advanced heuristic evaluation for the game.
            Considers piece value, board control, mobility, and prioritizes flag capture.
            """
            piece_weights = {
                "[2_FLG]": 100, "[1_FLG]": -100,
                "[2_COL]": 10, "[1_COL]": -10,
                "[2_MAJ]": 8, "[1_MAJ]": -8,
                "[2_CAP]": 6, "[1_CAP]": -6,
                "[2_1LT]": 5, "[1_1LT]": -5,
                "[2_2LT]": 4, "[1_2LT]": -4,
                "[2_1SG]": 15, "[1_1SG]": -15,
                "[2_2SG]": 14, "[1_2SG]": -14,
                "[2_3SG]": 13, "[1_3SG]": -13,
                "[2_4SG]": 12, "[1_4SG]": -12,
                "[2_5SG]": 11, "[1_5SG]": -11,
                "[2_PVT]": 1, "[1_PVT]": -1,
                "[2_SPY]": 50, "[1_SPY]": -50,
            }

            # Central squares for strategic control
            center_squares = ["D4", "D5", "E4", "E5"]  # Modify according to board setup
            center_bonus = 2

            # Flag capture priority
            flag_capture_bonus = 50

            # Mobility bonus for aggressive positioning
            mobility_bonus = 1

            # Defensive penalty for leaving the flag exposed
            flag_defense_penalty = 30

            score = 0
            opponent_flag_pos = None
            ai_flag_pos = None

            # Find the positions of the flags
            for i, square in enumerate(board):
                if square == "[1_FLG]":
                    opponent_flag_pos = i
                elif square == "[2_FLG]":
                    ai_flag_pos = i

            # Iterate over the board to calculate the score
            for i, square in enumerate(board):
                if square in piece_weights:
                    # Add piece value
                    score += piece_weights[square]

                    # Add bonus for controlling central squares
                    slot_code = f"{chr(65 + i // 9)}{i % 9 + 1}"
                    if slot_code in center_squares:
                        score += center_bonus if square.startswith("[2_") else -center_bonus

                    # Add mobility bonus for aggressive positioning
                    if i % 9 in [0, 8] or i // 9 in [0, 7]:
                        score += mobility_bonus if square.startswith("[2_") else -mobility_bonus

                    # Prioritize attacking the enemy flag
                    if square.startswith("[2_") and opponent_flag_pos is not None:
                        adj_positions = get_adjacent_positions(i, len(board))
                        if opponent_flag_pos in adj_positions:
                            score += flag_capture_bonus

            # Add a penalty if the AI flag is exposed
            if ai_flag_pos is not None:
                adj_positions = get_adjacent_positions(ai_flag_pos, len(board))
                for pos in adj_positions:
                    if board[pos].startswith("[1_"):
                        score -= flag_defense_penalty

            return score

        def get_adjacent_positions(index, board_size):
            """
            Helper function to get adjacent positions of a square on the board.
            """
            board_width = 9  # Assuming a 9x9 board
            board_height = board_size // board_width
            row, col = index // board_width, index % board_width

            adjacent_positions = []

            # Check above, below, left, right
            if row > 0:  # Above
                adjacent_positions.append((row - 1) * board_width + col)
            if row < board_height - 1:  # Below
                adjacent_positions.append((row + 1) * board_width + col)
            if col > 0:  # Left
                adjacent_positions.append(row * board_width + (col - 1))
            if col < board_width - 1:  # Right
                adjacent_positions.append(row * board_width + (col + 1))

            return adjacent_positions

        def generate_moves(board, player_prefix):
            directions = [9, -9, 1, -1]  # Directions for movement: down, up, right, left
            moves = []

            for index, piece in enumerate(board):
                if piece.startswith(f"[{player_prefix}_"):  # Only process pieces belonging to the current player
                    # Special handling for the flag piece
                    if piece == f"[{player_prefix}_FLG]":
                        valid_directions = directions[:]

                        # Remove invalid directions for edges
                        if index % 9 == 0:  # Left edge (no left movement)
                            if -1 in valid_directions:
                                valid_directions.remove(-1)
                        if (index + 1) % 9 == 0:  # Right edge (no right movement)
                            if 1 in valid_directions:
                                valid_directions.remove(1)

                        # Loop through all possible directions for the flag
                        for d in valid_directions:
                            target_index = index + d
                            if 0 <= target_index < len(board):  # Ensure target is within bounds
                                target_piece = board[target_index]

                                # Check if the target square is empty or an opponent's flag
                                if target_piece == '[---]' or target_piece == "[1_FLG]":
                                    moves.append((index, target_index))  # Add valid move to the list
                    else:
                        # For other pieces, generate standard valid moves
                        valid_directions = directions[:]

                        # Remove invalid directions for edges
                        if index % 9 == 0:  # Left edge (no left movement)
                            if -1 in valid_directions:
                                valid_directions.remove(-1)
                        if (index + 1) % 9 == 0:  # Right edge (no right movement)
                            if 1 in valid_directions:
                                valid_directions.remove(1)

                        # Loop through all possible directions for regular pieces
                        for d in valid_directions:
                            target_index = index + d
                            if 0 <= target_index < len(board):  # Ensure target is within bounds
                                target_piece = board[target_index]

                                # Check if the target square is empty or occupied by an opponent's piece
                                if target_piece == '[---]' or not target_piece.startswith(f"[{player_prefix}_"):
                                    moves.append((index, target_index))  # Add valid move to the list

            return moves

        def alpha_beta_pruning(board, depth, alpha, beta, is_computer_turn):
            """
            Alpha-Beta Pruning algorithm to determine the best move.
            """
            if depth == 0 or not generate_moves(board, "2" if is_computer_turn else "1"):
                return heuristic_evaluation(board, is_computer_turn), None

            if is_computer_turn:
                max_eval = -math.inf
                best_move = None
                for move in generate_moves(board, "2"):
                    # Simulate the move
                    new_board = board[:]
                    from_index, to_index = move
                    captured_piece = new_board[to_index]
                    new_board[to_index] = new_board[from_index]  # Move piece to the new position
                    new_board[from_index] = '[---]'  # Empty the original position

                    eval, _ = alpha_beta_pruning(new_board, depth - 1, alpha, beta, False)
                    if eval > max_eval:
                        max_eval = eval
                        best_move = move
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break
                return max_eval, best_move
            else:
                min_eval = math.inf
                best_move = None
                for move in generate_moves(board, "1"):
                    # Simulate the move
                    new_board = board[:]
                    from_index, to_index = move
                    captured_piece = new_board[to_index]
                    new_board[to_index] = new_board[from_index]  # Move piece to the new position
                    new_board[from_index] = '[---]'  # Empty the original position

                    eval, _ = alpha_beta_pruning(new_board, depth - 1, alpha, beta, True)
                    if eval < min_eval:
                        min_eval = eval
                        best_move = move
                    beta = min(beta, eval)
                    if beta <= alpha:
                        break
                return min_eval, best_move

            # Call Alpha-Beta Pruning to find the best move for the computer

        _, best_move = alpha_beta_pruning(self.squares, depth=3, alpha=-math.inf, beta=math.inf, is_computer_turn=True)

        if best_move:
            from_index, to_index = best_move

            print("Computer is making a move...")
            # Print the chosen slot and target slot
            choose = self.slot_code[from_index]
            move_to = self.slot_code[to_index]
            print(f"Chosen slot: {choose} -> Piece: {self.squares[from_index]}")
            print(f"Target slot: {move_to} -> Piece: {self.squares[to_index]}")

            piece_value = self.get_rank(self.squares[from_index])
            target_value = self.get_rank(self.squares[to_index])
            piece_rank = self.squares[from_index]
            target_rank = self.squares[to_index]

            print(f"Piece rank: {piece_value}")
            print(f"Target rank: {target_value}")

            # Move the piece or capture an opponent's piece
            if piece_value == target_value:
                print(f"{piece_rank} and {target_rank} eliminate each other!")
                self.squares[from_index] = '[---]'
                self.squares[to_index] = '[---]'

            # Flag piece special rule: Flag cannot capture any piece except the opponent's flag
            if piece_rank == '[2_FLG]':
                if target_rank != '[1_FLG]':  # The AI's flag can only capture the opponent's flag
                    print(f"AI Flag cannot capture any piece except the opponent's flag.")
                    # Skip this move, return control to pick a valid move
                    return "ITS NOT WORKING"  # Recurse to pick another move if the flag can't capture

                else:
                    print(f"AI Flag captures the opponent's flag!")
                    self.squares[to_index] = self.squares[from_index]
                    self.squares[from_index] = '[---]'
                    captured_piece = self.squares[to_index]
                    print(f"Captured: {captured_piece}")
                    return True  # Proceed with flag capturing the opponent's flag

            # If the piece can't capture, both pieces are eliminated
            if not self.capture_piece(piece_rank, target_rank, piece_value, target_value):
                if piece_value < target_value:
                    self.squares[from_index] = '[---]'  # Lower rank piece vanishes
                else:
                    self.squares[to_index] = '[---]'  # Lower rank piece on target vanishes
            else:
                # Successful capture: Move the attacking piece to the target square
                captured_piece = self.squares[to_index]
                if target_value == 0:
                    print(" ")
                else:
                    print(f"Captured: {captured_piece}")
                self.squares[to_index] = self.squares[from_index]
                self.squares[from_index] = '[---]'
        else:
            print("Computer has no available moves left.")

        self.update()
        self.player_move()


if __name__ == "__main__":
    Game()
