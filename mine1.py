import random
import numpy as np
np.set_printoptions(precision=2)

random.seed(14)

"""

MAde yet another thing : let's consider a clue and say I have its
entropy. Now I change one of the neighbours cell : if it was a mine, I
set it to "not a mine" (or vice versa). Then I look how the clue
entropy has changed. If the entropy goes up, it means that its
prediction is not in line with what I've put in the cell, so what i


We make the following obsevartions :

- the entropy of clue gives a measure of certainty of the prediction made by the clue
  it doesn't indicate *what* the clue predicts (so the clue is certain or not uncertain,
  but we don't know if it is certain/uncertain regarding the fact that there is a mine).

- combining several clue won't help : we still don't know what they collectively predict.

- a clue close to the fringe makes two predictions : it predicts that there may or may not
  be a mine in a cell outside the firnge. But also, since the fringe is the fringe, it
  predicts that clues in the fringe are not mines (else the game would already be finished).




"""


def nb_pos(px,py):
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
    #print(REVEAL)
    # print("")
    for y in range(10):
        s = ""
        for x in range(10):
            if REVEAL[y,x] == 1:
                if MINES[y,x] == 0:
                    mines, uneighbours, clue_entropy = clue(x, y)
                    entropies[y,x] = clue_entropy
                    s += f"{mines}/{entropies[y,x]:.2f} "
    return entropies

all_turns = 0
for game in range(1):
    MINES = np.random.randint(2, size=(10,10))
    for i in range(100):
        x = random.randint(0,9)
        y = random.randint(0,9)
        MINES[y,x] = 0

    #print(MINES)
    REVEAL = np.zeros((10,10),dtype=int)
    REVEAL[0:1,:] = 1
    #REVEAL[1,3:7] = 1

    turns = 0
    while True:
        entropies = compute_board_entropies()
        draw_board(entropies)

        print("\nNo mine")
        clue(0,0,True)
        REVEAL[1,0] = 1
        clue(0,0,True)
        REVEAL[1,0] = 0
        entropies = compute_board_entropies()
        print("\nMine")
        clue(3,0,True)
        REVEAL[1,3] = 1
        entropies = compute_board_entropies()
        clue(3,0,True)
        REVEAL[1,3] = 0
        entropies = compute_board_entropies()
        break


        for x in range(10):
            m_old = MINES[1, x]
            _, _, c_before = clue(x, 0)
            MINES[1, x] = 1
            _, _, c_after = clue(x, 0)
            MINES[1, x] = m_old

            # putting a mine leads to worse prediction, so previous
            # prediction was not a mine...
            choose = c_after - c_before > 0
            print(f"x={x}: mine_old={m_old}, diff = {c_after - c_before}, {choose}")





        best_entropy = None

        for px in range(10):
            for py in range(10):
                if REVEAL[py,px] == 1:
                    continue

                neighbours_pos = nb_pos(px, py)

                # entropy = 0
                # for x, y in neighbours_pos:
                #     entropy += entropies[y, x]
                # entropy /= len(neighbours_pos)

                entropy = 0
                for x, y in neighbours_pos:
                    entropy = max(entropy, entropies[y, x])

                # Take minimal entropy
                # IF there's a tie, then take the cell that will give
                # the more clues
                if best_entropy is None or entropy > best_entropy[0]  or (entropy == best_entropy[0]  and len(neighbours_pos) > best_entropy[1]) :
                    best_entropy = entropy, len(neighbours_pos), px, py

        e,nn,x,y = best_entropy
        #print(f"x={x}, y={y}, ent={e}, mine? {MINES[y,x] == 1}")

        # Simulate random play
        # x = random.randint(0,9)
        # y = 1

        if MINES[y,x] == 1:
            break
        turns += 1
        REVEAL[y,x] = 1

    all_turns += turns
    print(f"Game {game}, turns {turns}, avg={all_turns / (game+1)}")

#print(entropies)
