#!/usr/bin/env python
import numpy as np
from itertools import product, combinations, cycle
from scipy.ndimage import label
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import datetime
import argparse # Import the argparse module

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Solve Caesar's Calendar puzzle.")
parser.add_argument("--month", dest="month_abbr", type=str, help="Three-letter month abbreviation (e.g., 'JUN')")
parser.add_argument("--day", dest="day_number", type=str, help="Day of the month as a number (e.g., '26')")
parser.add_argument("--weekday", dest="weekday_abbr", type=str, help="Three-letter weekday abbreviation (e.g., 'THU')")
parser.add_argument("--solutions", dest="desired_solutions", type=int, default=3,
                    help="Number of desired solutions to find (default: 3)")

args = parser.parse_args()

# Get the current date and time
today = datetime.date.today()

# Get the three-letter month abbreviation (e.g., "JUN")
# Use the command-line argument if provided, otherwise use today's date
month_abbr = args.month_abbr if args.month_abbr else today.strftime("%b").upper()

# Get the day of the month as a number (e.g., "26")
# Use the command-line argument if provided, otherwise use today's date
day_number = args.day_number if args.day_number else str(today.day)

# Get the three-letter weekday abbreviation (e.g., "THU")
# Use the command-line argument if provided, otherwise use today's date
weekday_abbr = args.weekday_abbr if args.weekday_abbr else today.strftime("%a").upper()

# Use the command-line argument for desired_solutions
desired_solutions = args.desired_solutions

# Tetrominoes (3 pieces)
tetrominoes = [
    np.array([[1, 1, 1, 1]]),  # I
    np.array([[0, 1, 1],
              [1, 1, 0]]),     # S4
    np.array([[1, 0, 0],
              [1, 1, 1]])      # L4
]

# Pentominoes (7 pieces)
pentominoes = [
    np.array([[1, 0, 0, 0],
              [1, 1, 1, 1]]),  # L5
    np.array([[1, 1, 0],
              [0, 1, 0],
              [0, 1, 1]]),     # Z
    np.array([[1, 1, 1],
              [0, 1, 0],
              [0, 1, 0]]),     # T5
    np.array([[1, 0, 1],
              [1, 1, 1]]),     # U
    np.array([[0, 1, 1, 1],
              [1, 1, 0, 0]]),  # S5
    np.array([[1, 0, 0],
              [1, 0, 0],
              [1, 1, 1]]),     # L-square
    np.array([[1, 1, 0],
              [1, 1, 1]])      # b
]

pieces = tetrominoes + pentominoes
piece_colors = np.array([i for i in range(0, len(pieces))])

# Constants
BOARD_HEIGHT = 8
BOARD_WIDTH = 7

# Create the board
board = np.ones((BOARD_HEIGHT, BOARD_WIDTH), dtype=int)
blocked_edges = [(0, 6), (1, 6), (7, 0), (7, 1), (7, 2), (7, 3)]


# Define the lists for associations
month_names = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
day_numbers = [str(num) for num in list(range(1, 32))] # Numbers from 1 to 31
weekday_names = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]

# Create a combined list of all labels
cell_labels = month_names + day_numbers + weekday_names

# Create cyclical iterators for labels
labels_cycler = cycle(cell_labels)

# Dictionary to store the forward associations (coordinate -> label)
# The keys will be (row, column) tuples, and values will be a label
# containing a 'month', 'day' or 'weekday' for that coordinate.
board_labels = {}

# Iterate through the board from left to right, top to bottom
for r in range(BOARD_HEIGHT):
    for c in range(BOARD_WIDTH):
        # Check if the current coordinate is in the blocked edges list
        if (r, c) not in blocked_edges:
            # If not blocked, associate it with the next available label
            current_label = next(labels_cycler)
            board_labels[(r, c)] = current_label

# --- Create the reversed association dictionary (label -> list of coordinates) ---
# This dictionary will map each label to a list of all coordinates that it is associated with.
board_coords = {}
for coord, label_text in board_labels.items():
    # If the label is not yet a key in the reversed dictionary, initialize it with an empty list
    if label_text not in board_coords:
        board_coords[label_text] = []
    # Add the current coordinate to the list associated with this label
    board_coords[label_text]= coord


holes = [board_coords[label_text] for label_text in [month_abbr, day_number, weekday_abbr]]  # e.g. JUN, 24, TUE
for r, c in holes + blocked_edges:
    board[r, c] = 0


# Orientation generator
def generate_orientations(piece):
    orientations = set()
    for k in range(4):
        rotated = np.rot90(piece, k)
        orientations.add(tuple(map(tuple, rotated)))
        orientations.add(tuple(map(tuple, np.fliplr(rotated))))
    return [np.array(o) for o in orientations]

# Valid placement
def can_place(board, piece, top, left):
    h, w = piece.shape
    if top + h > BOARD_HEIGHT or left + w > BOARD_WIDTH:
        return False
    subgrid = board[top:top + h, left:left + w]
    return np.all((piece == 0) | (subgrid == 1))

def place_piece(board, piece, top, left):
    h, w = piece.shape
    board[top:top + h, left:left + w] += piece

def remove_piece(board, piece, top, left):
    h, w = piece.shape
    board[top:top + h, left:left + w] -= piece

# Heuristic: region fillability
def board_has_unfillable_regions(board, remaining_pieces):
    temp_board = (board == 1).astype(int)
    labeled, num_features = label(temp_board)
    piece_sizes = [np.sum(p) for p in remaining_pieces]

    for region_label in range(1, num_features + 1):
        region_size = np.sum(labeled == region_label)

        if region_size < 4:
            return True
        if not is_region_fillable(region_size, piece_sizes):
            return True
    return False

def is_region_fillable(region_size, piece_sizes):
    for r in range(1, len(piece_sizes) + 1):
        for combo in combinations(piece_sizes, r):
            if sum(combo) == region_size:
                return True
    return False

def solve(board, pieces_left, placements):
    global iteration_count
    depth = len(placements)

    if not pieces_left:
        solutions.append(list(placements))
        return True

    if board_has_unfillable_regions(board, pieces_left):
        return False

    piece = pieces_left[0]
    orientations = generate_orientations(piece)

    for orient in orientations:
        h, w = orient.shape
        for r, c in product(range(BOARD_HEIGHT - h + 1), range(BOARD_WIDTH - w + 1)):
            iteration_count += 1
            if iteration_count % 1000 == 0:
                print(f"\rIteration {iteration_count}, depth {depth}: Trying piece at ({r},{c})", end="", flush=True)

            if can_place(board, orient, r, c):
                place_piece(board, orient, r, c)
                placements.append((orient, r, c))
                if solve(board, pieces_left[1:], placements):
                    return True
                placements.pop()
                remove_piece(board, orient, r, c)
    return False

# Solver with pruning and iteration display

solutions = []
solution_colors = []

while (len(solutions) < desired_solutions):
    # get a different solution every time?
    pieces_shuffled, colors_shuffled = shuffle(pieces, piece_colors)

    # Run solver
    board_copy = board.copy()
    iteration_count = 0
    solve(board_copy, pieces_shuffled, [])
    solution_colors.append(colors_shuffled)
    print(f"\n✅ Found solution {len(solutions)} in {iteration_count} iterations.\n")

piece_ids = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=int)

if 'solutions' in globals() and solutions:
    for solution_no, solution in enumerate(solutions):
        for idx, (shape, r0, c0) in enumerate(solution, 1):
            h, w = shape.shape
            for i in range(h):
                for j in range(w):
                    if shape[i, j] == 1:
                        piece_ids[r0 + i, c0 + j] = idx

        # Plotting the actual solution
        fig, ax = plt.subplots(figsize=(4.66, 5.33))
        ax.set_xlim(0, BOARD_WIDTH)
        ax.set_ylim(0, BOARD_HEIGHT)
        ax.set_xticks(np.arange(0, BOARD_WIDTH + 1))
        ax.set_yticks(np.arange(0, BOARD_HEIGHT + 1))
        ax.grid(True)

        for r in range(BOARD_HEIGHT):
            for c in range(BOARD_WIDTH):
                y = BOARD_HEIGHT - r - 1
                x = c
                piece_val = piece_ids[r, c]

                if (r, c) in holes:
                    facecolor = 'white'
                elif (r, c) in blocked_edges:
                    facecolor = 'black'
                elif board[r, c] == 1:
                    facecolor = f"C{(solution_colors[solution_no][piece_val - 1])}" if piece_val else "tan"
                else:
                    facecolor = 'white'

                rect = patches.Rectangle((x, y), 1, 1, linewidth=1, edgecolor='black', facecolor=facecolor)
                ax.add_patch(rect)

        # Draw thick outlines between pieces
        for r in range(BOARD_HEIGHT):
            for c in range(BOARD_WIDTH):
                id = piece_ids[r, c]
                if id == 0:
                    continue

                y = BOARD_HEIGHT - r - 1
                x = c

                for dr, dc, side in [(-1, 0, 'top'), (1, 0, 'bottom'), (0, -1, 'left'), (0, 1, 'right')]:
                    nr, nc = r + dr, c + dc
                    neighbor_id = piece_ids[nr, nc] if 0 <= nr < BOARD_HEIGHT and 0 <= nc < BOARD_WIDTH else -1
                    if neighbor_id != id:
                        if side == 'top':
                            ax.plot([x, x + 1], [y + 1, y + 1], color='black', linewidth=2)
                        elif side == 'bottom':
                            ax.plot([x, x + 1], [y, y], color='black', linewidth=2)
                        elif side == 'left':
                            ax.plot([x, x], [y, y + 1], color='black', linewidth=2)
                        elif side == 'right':
                            ax.plot([x + 1, x + 1], [y, y + 1], color='black', linewidth=2)

        # Add labels in hole positions for today

        for label_text in [month_abbr, day_number, weekday_abbr]:
            (r, c) = board_coords[label_text]
            ax.text(c + 0.5, BOARD_HEIGHT - r - 0.5, label_text, ha='center', va='center', fontsize=9, color='black')

        ax.set_aspect('equal')
        my_title = f"Caesar's Calendar — {month_abbr} {day_number}, {weekday_abbr}"
        ax.set_title(my_title)
        plt.tight_layout()

# After the loop, call plt.show() once to display all figures
plt.show()
