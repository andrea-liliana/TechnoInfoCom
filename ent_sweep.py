import random
import numpy as np

# Need both
random.seed(10)
np.random.seed(16)


def nb_pos(px,py):
    """Enumerate legal coordinates adjacent to a given position.
    """

    good_x = [px]
    if px > 0:
        good_x += [px-1]
    if px < 10-1:
        good_x += [px+1]

    good_y = [py]
    if py > 0:
        good_y += [py-1]
    if py < 10-1:
        good_y += [py+1]

    pos = []
    for x in good_x:
        for y in good_y:
            if x == px and y == py:
                continue
            else:
                pos.append((x,y))
    return pos


def clue(px,py,display=False):
    """ Compute information of a clue located in px,py :
    - the number of adjacent mines
    - the number of unrevealed adjacent cells
    - the entropy of the clue
    """

    mines = 0
    unrevealed_neighbours = 0
    #print(nb_pos(px, py))
    for x, y in nb_pos(px, py):
        if REVEAL[y,x] == 1:
            continue

        unrevealed_neighbours += 1
        if MINES[y, x] == 1:
            mines += 1

    if mines == unrevealed_neighbours or unrevealed_neighbours == 0 or mines == 0:
        clue_entropy = 0
    else:
        p = float(mines)/unrevealed_neighbours

        # H = -sum_{p_i} p_i log2 p_i

        clue_entropy = - unrevealed_neighbours * p * np.log2(p)

    if display:
        print(f"({px},{py}) : mines:{mines}, unrevealed_neighbours:{unrevealed_neighbours}, clue_entropy:{clue_entropy}")
    return mines, unrevealed_neighbours, clue_entropy



def draw_board(entropies):
    for y in range(10):
        s = ""
        for x in range(10):
            if REVEAL[y,x] == 0:
                if MINES[y,x] == 1:
                    s += "   m   "
                else:
                    s += "   .   "

            elif REVEAL[y,x] == 1:
                if MINES[y,x] == 1:
                    s += "   M   "
                else:
                    mines, uneighbours, clue_entropy = clue(x, y)
                    s += f"{mines}/{entropies[y,x]:.2f} "
            else:
                s += "   .   "
        print(f"{s}")


def compute_board_entropies():
    entropies = np.ones((10,10)) * 99
    for y in range(10):
        for x in range(10):
            if REVEAL[y, x] == 1: # and MINES[y,x] == 0:
                # Clues are computed only on revealed cells

                mines, uneighbours, clue_entropy = clue(x, y)
                entropies[y, x] = clue_entropy

    #print(entropies)
    return entropies


all_turns = 0
all_mines = 0
for game in range(800):

    # Fill the minefield in a sparse way
    MINES = np.random.randint(2, size=(10,10))
    for i in range(100):
        x = random.randint(0,9)
        y = random.randint(0,9)
        MINES[y,x] = 0
    all_mines += np.sum(MINES)

    # Note revealed position (1 == revealed, 0 == unrevealed)
    REVEAL = np.zeros((10,10),dtype=int)

    turns = 0
    while True:
        entropies = compute_board_entropies()

        best = None

        for px in range(10):
            for py in range(10):
                if REVEAL[py, px] == 1:
                    continue

                neighbours_pos = nb_pos(px, py)
                neighbouring_clues = 0
                entropy = 0
                for x, y in neighbours_pos:
                    if REVEAL[y, x] == 1:
                        neighbouring_clues += 1
                        entropy += entropies[y, x]

                if neighbouring_clues > 0:
                    if best is None or entropy < best[0]:
                        best = entropy, None, px, py

        if best:
            e,nn,x,y = best
        else:
            e,nn,x,y = -1, 0, 1, 1

        if MINES[y,x] == 1:
            # Boom ! dead.
            break

        turns += 1
        REVEAL[y,x] = 1

    all_turns += turns
    print(f"Game {game}, turns {turns}, avg={all_turns / (game+1):.3f}, avg mines={all_mines/(game+1):.1f}")

#print(entropies)
