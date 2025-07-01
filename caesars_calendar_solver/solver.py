import numpy as np
from itertools import product, combinations, cycle
from scipy.ndimage import label
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import datetime
import argparse

# Constants
BOARD_HEIGHT = 8
BOARD_WIDTH = 7

tetrominoes = [
    np.array([[1, 1, 1, 1]]),
    np.array([[0, 1, 1],
              [1, 1, 0]]),
    np.array([[1, 0, 0],
              [1, 1, 1]])
]

pentominoes = [
    np.array([[1, 0, 0, 0],
              [1, 1, 1, 1]]),
    np.array([[1, 1, 0],
              [0, 1, 0],
              [0, 1, 1]]),
    np.array([[1, 1, 1],
              [0, 1, 0],
              [0, 1, 0]]),
    np.array([[1, 0, 1],
              [1, 1, 1]]),
    np.array([[0, 1, 1, 1],
              [1, 1, 0, 0]]),
    np.array([[1, 0, 0],
              [1, 0, 0],
              [1, 1, 1]]),
    np.array([[1, 1, 0],
              [1, 1, 1]])
]

pieces = tetrominoes + pentominoes
piece_colors = np.array(range(len(pieces)))
blocked_edges = [(0, 6), (1, 6), (7, 0), (7, 1), (7, 2), (7, 3)]

month_names = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
day_numbers = [str(num) for num in range(1, 32)]
weekday_names = ["SUN", "MON", "TUE", "WED", "THU", "FRI", "SAT"]
cell_labels = month_names + day_numbers + weekday_names


def generate_orientations(piece):
    orientations = set()
    for k in range(4):
        rot = np.rot90(piece, k)
        orientations.add(tuple(map(tuple, rot)))
        orientations.add(tuple(map(tuple, np.fliplr(rot))))
    return [np.array(o) for o in orientations]


def can_place(board, piece, top, left):
    h, w = piece.shape
    if top + h > BOARD_HEIGHT or left + w > BOARD_WIDTH:
        return False
    return np.all((piece == 0) | (board[top:top + h, left:left + w] == 1))


def place_piece(board, piece, top, left):
    board[top:top + piece.shape[0], left:left + piece.shape[1]] += piece


def remove_piece(board, piece, top, left):
    board[top:top + piece.shape[0], left:left + piece.shape[1]] -= piece


def is_region_fillable(region_size, piece_sizes):
    for r in range(1, len(piece_sizes) + 1):
        for combo in combinations(piece_sizes, r):
            if sum(combo) == region_size:
                return True
    return False


def board_has_unfillable_regions(board, remaining_pieces):
    temp = (board == 1).astype(int)
    labeled, n = label(temp)
    sizes = [np.sum(p) for p in remaining_pieces]

    for i in range(1, n + 1):
        if not is_region_fillable(np.sum(labeled == i), sizes):
            return True
    return False


def solve(board, pieces_left, placements, solutions, iteration_count):
    if not pieces_left:
        solutions.append(list(placements))
        return True

    if board_has_unfillable_regions(board, pieces_left):
        return False

    piece = pieces_left[0]
    for orient in generate_orientations(piece):
        for r, c in product(range(BOARD_HEIGHT - orient.shape[0] + 1),
                            range(BOARD_WIDTH - orient.shape[1] + 1)):
            iteration_count[0] += 1
            if can_place(board, orient, r, c):
                place_piece(board, orient, r, c)
                placements.append((orient, r, c))
                if solve(board, pieces_left[1:], placements, solutions, iteration_count):
                    return True
                placements.pop()
                remove_piece(board, orient, r, c)
    return False


def solve_puzzle(month_abbr, day_number, weekday_abbr, desired_solutions=3):
    board = np.ones((BOARD_HEIGHT, BOARD_WIDTH), dtype=int)
    labels_cycler = cycle(cell_labels)
    board_labels, board_coords = {}, {}

    for r in range(BOARD_HEIGHT):
        for c in range(BOARD_WIDTH):
            if (r, c) not in blocked_edges:
                label = next(labels_cycler)
                board_labels[(r, c)] = label
                board_coords[label] = (r, c)

    holes = [board_coords[label] for label in (month_abbr, day_number, weekday_abbr)]
    for r, c in holes + blocked_edges:
        board[r, c] = 0

    solutions, colors = [], []

    while len(solutions) < desired_solutions:
        shuffled_pieces, shuffled_colors = shuffle(pieces, piece_colors)
        iteration_count = [0]
        solve(board.copy(), shuffled_pieces, [], solutions, iteration_count)
        colors.append(shuffled_colors)
        print(f"\n✅ Found solution {len(solutions)} in {iteration_count[0]} iterations.")

    return solutions, colors, board, holes, board_coords


def display_solution(solutions, colors, board, holes, board_coords, month, day, weekday):
    for idx, solution in enumerate(solutions):
        piece_ids = np.zeros_like(board)
        for piece_idx, (shape, r0, c0) in enumerate(solution, 1):
            for i in range(shape.shape[0]):
                for j in range(shape.shape[1]):
                    if shape[i, j]:
                        piece_ids[r0 + i, c0 + j] = piece_idx

        fig, ax = plt.subplots(figsize=(4.66, 5.33))
        ax.set_xlim(0, BOARD_WIDTH)
        ax.set_ylim(0, BOARD_HEIGHT)
        ax.set_xticks(np.arange(BOARD_WIDTH + 1))
        ax.set_yticks(np.arange(BOARD_HEIGHT + 1))
        ax.grid(True)

        for r in range(BOARD_HEIGHT):
            for c in range(BOARD_WIDTH):
                y, x = BOARD_HEIGHT - r - 1, c
                pid = piece_ids[r, c]
                if (r, c) in holes:
                    color = 'white'
                elif (r, c) in blocked_edges:
                    color = 'black'
                elif board[r, c] == 1:
                    color = f"C{colors[idx][pid - 1]}" if pid else "tan"
                else:
                    color = 'white'
                ax.add_patch(patches.Rectangle((x, y), 1, 1, edgecolor='black', facecolor=color))

        for r in range(BOARD_HEIGHT):
            for c in range(BOARD_WIDTH):
                id = piece_ids[r, c]
                if id == 0:
                    continue
                y, x = BOARD_HEIGHT - r - 1, c
                for dr, dc, side in [(-1, 0, 'top'), (1, 0, 'bottom'), (0, -1, 'left'), (0, 1, 'right')]:
                    nr, nc = r + dr, c + dc
                    nid = piece_ids[nr, nc] if 0 <= nr < BOARD_HEIGHT and 0 <= nc < BOARD_WIDTH else -1
                    if nid != id:
                        if side == 'top':
                            ax.plot([x, x + 1], [y + 1, y + 1], color='black', lw=2)
                        elif side == 'bottom':
                            ax.plot([x, x + 1], [y, y], color='black', lw=2)
                        elif side == 'left':
                            ax.plot([x, x], [y, y + 1], color='black', lw=2)
                        elif side == 'right':
                            ax.plot([x + 1, x + 1], [y, y + 1], color='black', lw=2)

        for label in (month, day, weekday):
            r, c = board_coords[label]
            ax.text(c + 0.5, BOARD_HEIGHT - r - 0.5, label, ha='center', va='center', fontsize=9, color='black')

        ax.set_aspect('equal')
        ax.set_title(f"Caesar's Calendar — {month} {day}, {weekday}")
        plt.tight_layout()

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Solve Caesar's Calendar puzzle.")
    parser.add_argument("--month", type=str, help="Month abbreviation (e.g., JUN)")
    parser.add_argument("--day", type=str, help="Day number (e.g., 26)")
    parser.add_argument("--weekday", type=str, help="Weekday abbreviation (e.g., THU)")
    parser.add_argument("--solutions", type=int, default=3, help="Number of solutions to find (default: 3)")
    args = parser.parse_args()

    today = datetime.date.today()
    month = (args.month or today.strftime("%b")).upper()
    day = args.day or str(today.day)
    weekday = (args.weekday or today.strftime("%a")).upper()
    
    print(f"Looking for {args.solutions} solution(s) for {weekday}, {month} {day}...")

    sols, colors, board, holes, coords = solve_puzzle(month, day, weekday, args.solutions)
    display_solution(sols, colors, board, holes, coords, month, day, weekday)


if __name__ == "__main__":
    main()
