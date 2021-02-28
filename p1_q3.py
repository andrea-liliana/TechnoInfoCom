import csv
import numpy as np


def entropy(probabilities):
    """
    Computes H(X)

    X is given as a numpy array, all elements of the array
    are assumed to represent the distribution (to the shape
    of the array is not meaningful)

    """

    # Avoid situations where log can't be computed
    logs = np.log2(probabilities[probabilities != 0])
    return - np.sum(probabilities * logs)


def joint_entropy(X_and_Y):
    """
    Compute H(X ∩ Y)

    X_and_Y : joint probabilities, a 2D numpy array such that
              X_and_Y[i,j] == P[X_i ∩ Y_j]

    We don't assume X is conditionally independent of Y.
    """
    return entropy(X_and_Y)


def conditional_entropy(X_given_Y, Y):
    """
    Computes H(X|Y)

    X_given_Y : conditional probability P(X|Y)
                X_i | Y_i : row(i), column(j)
    Y: marginal probability P(Y)

    Slide 3, course 2 : H(X |Y) = − Σ Σ P(Xi ∩ Yj) log P(Xi | Yj)
    Recall that P(X ∩ Y) = P(X | Y) P(Y)
    => P(X_i ∩ Y_j) = P(X_i|Y_j) * P(Y_j)
    => H(X |Y) = − Σ Σ P(X_i|Y_j) * P(Y_j) log P(Xi | Yj)

    => if we know P(Y) and P(X | Y), we can compute the
    conditional entropy.

    But to do that we must be able to match P(X|Y_i) and P(Y_i),
    so we need to have the values of these random variables.

    """

    # Compare number of columns in X_given_Y (that's number of Y's)
    # to length of the Y vector.
    assert X_given_Y.shape[1] == Y.shape[1], f"Shape don't match : {X_given_Y.shape[1]} != {Y.shape[1]}"

    # z helps to compute combinations of i and j indices.
    z = np.tile(Y, (X_given_Y.shape[0], 1))

    return - np.sum(X_given_Y * z * np.log2(X_given_Y))


def test_pandas():
    import pandas
    x_given_y = pandas.DataFrame(
        [[True, True, 0.25],
         [True, False, 0.25],
         [False, True, 0.2],
         [False, False, 0.3]],
        columns=["X", "Y", "P(X|Y)"])

    y = pandas.DataFrame([[True, 0.3],
                          [False, 0.7]],
                         columns=["Y", "P(Y)"])

    # Compute the entropy

    # Firs, relate P(X_i|Y_j) to P(Y_j)
    r = pandas.merge(x_given_y, y)
    cond_entropy = - np.sum(r["P(X|Y)"] * r["P(Y)"] * np.log2(r["P(X|Y)"]))

    # Compute mutual information


    # Remember, we don't know if X and Y are
    # conditionally independent

    x_and_y = pandas.DataFrame(
        [[True, True, 0.25],
         [True, False, 0.25],
         [False, True, 0.2],
         [False, False, 0.3]],
        columns=["X", "Y", "P(X^Y)"])

    # compute p_x by marginalization
    p_x = x_and_y.groupby("X").agg(p_x=('P(X^Y)', 'sum')).reset_index()['p_x']

    # idem for p_y
    p_y = x_and_y.groupby("Y").agg(p_y=('P(X^Y)', 'sum')).reset_index()['p_y']

    # Apply : I(X;Y) = H(X) + H(Y) - H(X,Y) (See wikipedia)
    def h(p):
        return - np.sum(p * np.log2(p))

    mutual_information = h(p_x) + h(p_y) - h(x_and_y['P(X^Y)'])



def mutual_information(X, Y, X_and_Y):
    """
    I(X;Y) = H(X) + H(Y) - H(X,Y) (See wikipedia)
    """

    print(X.shape)
    print(Y.shape)
    print(X_and_Y.shape)

    # Warning! Vector's shape are only one integer long
    assert X.shape[0] == X_and_Y.shape[0]
    assert Y.shape[0] == X_and_Y.shape[1]

    return entropy(X) + entropy(Y) - joint_entropy(X_and_Y)


def joint_entropy2(X,Y,Z):
    """
    FIXME
    I'm afraid this suggests P(X,Y,Z) = P(X)P(Y)P(Z) which is true
    only if X ⊥ Y ⊥ Z. No ?
    """

    probability = []
    for x1 in set(X):
        for x2 in set(Y):
            for x3 in set(Z):

                # FIXME The doc : https://numpy.org/doc/stable/reference/generated/numpy.logical_and.html
                # says something different...

                # FIXME I don't understand the np.mean.
                probability.append(np.mean(np.logical_and(X == x1, Y == x2, Z==x3)))

    return np.sum(-p * np.log2(p) for p in probability)

def cond_mutual_information(X_and_Z, Y_and_Z, X, Y, Z):
    """
    I(X,Y|Z) = H(X,Z)+ H(Y,Z) - H(Z) - H(X,Y,Z)

    H(X,Y,Z) = H(X,Y|Z)+H(Z) => Stef: H(X,Y,Z) = H(X | Y,Z) + H(Y | Z) + H(Z)  (chain rule)
                                               = H(Y,X|Z) - H(Y|Z) + H(Y | Z) + H(Z) (see rule below)
                                               = H(Y,X|Z) + H(Z)

    H(X,Y|Z) = H(X|Z) + H(Y|X,Z) => from the course (Slide 9, point 2)

    which we prove like this :

    H(X,Y|Z) = - Σ_X Σ_Y Σ_Z P(X,Y ∩ Z) log(P(X,Y|Z))
             = - Σ_X Σ_Y Σ_Z P(X,Y,Z) log(P(Y|Z,X) P(X|Z))  # Because P(X,Y|Z) = P(X,Y,Z)/P(Z)
                                                                               = (P(X,Y,Z)/P(X,Z)))*(P(X,Z)/P(Z)
                                                                               = P(Y|X,Z) * P(X|Z)
             = - Σ_X Σ_Y Σ_Z P(X,Y,Z) log P(Y|Z,X)  # because log of product
               - Σ_X Σ_Y Σ_Z P(X,Y,Z) log P(X|Z)
             = H(Y|Z,X)  # because feinition of H
               - Σ_Y P(Y|Z,X) Σ_X Σ_Z P(Z,X) log P(X|Z)
             = H(Y|Z,X)
               + 1 * H(Z,X)  # Because Y probability distribution => Σ_i Y_i = 1
             = H(Y|Z,X) + H(X,Z)
    """
    return joint_entropy(X_and_Z) + joint_entropy(Y_and_Z) - entropy(Z) - joint_entropy2(X,Y,Z)

def cond_joint_entropy(X,Y,Z):
    """
    Computes H(X,Y|Z)

    H(X,Y|Z) this term is obtained here: H(X,Y,Z) = H(X,Y|Z)+H(Z)
    H(X,Y|Z) = H(X,Y,Z) - H(Z) (see above)

    """

    return joint_entropy2(X,Y,Z) - entropy(Z)


def cond_joint_entropy2(X_Y_Z, Z):
    """
    Computes H(X,Y|Z) base on the distribution table for P(X=x_i ∩ Y=y_i ∩ Z=z_i) (X_Y_Z)
    and P(Z=z_i)

    We have :

    H(X,Y,Z) = H(X,Y|Z)+H(Z) => Stef: H(X,Y,Z) = H(X | Y,Z) + H(Y | Z) + H(Z)  (chain rule)
                                               = H(Y,X|Z) - H(Y|Z) + H(Y | Z) + H(Z) (see rule below)
                                               = H(Y,X|Z) + H(Z)

    So :

    H(X,Y|Z) =  H(X,Y,Z) - H(Z)

    note that P(Z) can be obtained from P(X,Y,Z) by marginalilzation, but that would
    require another structure for X_Y_Z.

    """

    return entropy(X_Y_Z) - entropy(Z)



if __name__ == "__main__":

    test_pandas()
    exit()

    """
    Question 12
    ===========


    a RxC matrix
    there are M mines M < RxC
    sart of the game : no field revealed.

    So, each field has a probability of p_1_i_j = M/(RxC) of having a mine in it.
    So entropy of a field = - log2 p_1_i_j = - log2 M/(RxC)


    Question 13
    ===========

    FIXME unclear

    The entropy of on unrevealed cell is one of a binary experiment.
    H(U) = p log p + (1 - p) log(1 - p) where p is the probability of
    having a bomb in U (1-p the probability of not having a bomb)

    The probability of having a bomb is deduced from the clues in the
    immediate neighourhood By definition of a clue, an unrevealed cell
    adjacent to a clue has a 1/N probabiliyt (n = number of unrevealed
    cells around the clue) of being a mine. We gather all the clues
    surrounding the unrevealed by suming their probabilities. We could
    reason also on entropies by just summing entropies of each
    clue-cells around u-cell.

    Question 14
    ===========

    I have two needs :

    - I want to avoid the mine

      I look at the probabilities and choose cell where the probability
      of having a mine is smaller.

      I could also look at the entropy of the cell. If the entropy is
      small it means that the probability above is more certain.

    - I want to reveal a cell that gives me the most useful clue.

      A clue is useful if it decreases the entropies of its neighbours
      because less entropy, the more certain we are to have a mine or not
      a mine around (ie probability of having close to 1 or to zero).


    H(cell) = P_mine log p_mine + p_cell log p_cell with p_cell = 1 - p_mine

    if I minimize p_mine, H(cell) gets smaller as well (I'm more certain of
    revealing a clue)


    H(clue cell) = 0 = it's 100% certain we have clue

    let p = # undiscovered mines / # unrevealed cells
    H(unrevealed cell) = p log(p)

    if there's a clue adjacent, I have two probabilities.
    p above and p_clue given by the clue.

    How do we interpret p and p2 together, knowing both give
    the probability of having a mine ?


    Submarine : whenever I reveal a cell, I got
    H(cell is not a mine) = 1 - # undiscovered mines / # unrevealed cells

    """


    R, C, M = 8, 8, 1

    h_tot = 0
    for i in range(R*C - M): # The mines are the last two

        # p mine was not in the revealed places
        p = 1 - M / (R*C - i)

        h = np.log2( 1/p)
        h_tot += h
        print(f"p={p:.3f} h={h:.3f}, h_tot={h_tot:.3f}")


    exit()

    A = [[0, 1,    2,    3,    2],
         [1, 2,    None, None, None],
         [1, None, None, None, None]]

    P = [[None] * C,
         [None] * C,
         [None] * C]

    def count_unrevealed_neighbours(A,r,c):
        # Compute number of unrevealed cells around celle (r,c)
        # We do a lot of clipping
        nb_unrevealed, nb_clues = 0, 0
        for nr in range(max(0,r-1),min(r+2,R)):
            for nc in range(max(0,c-1),min(c+2,C)):
                if nr == nc == 0:
                    continue
                elif A[nr][nc] is None:
                    # The cell is unrevealed
                    nb_unrevealed += 1
                else:
                    nb_clues += 1

        return nb_unrevealed, nb_clues

    for r in range(R):
        for c in range(C):
            if A[r][c] is None:

                # Check the neighbourhoods of A(r,c)
                nb_unrevealed, nb_clues = count_unrevealed_neighbours(A, r, c)
                if nb_clues > 0:
                    # By question 13 statement, we only look in cells
                    # which are unrevealed and adjacent to clue-cells.

                    # We collect clues around A(r,c)

                    prob_cell_is_mine = 0
                    for nr in range(max(0,r-1),min(r+2,R)):
                        for nc in range(max(0,c-1),min(c+2,C)):
                            if nr == nc == 0:
                                continue
                            elif A[nr][nc] is not None:
                                # The cell is a clue

                                _, nb_c = count_unrevealed_neighbours(A, nr, nc)
                                p = (A[nr][nc]/nb_c)
                                # prob_cell_is_mine += - p * np.log2(p)
                                prob_cell_is_mine +=  p

                    P[r][c] = prob_cell_is_mine

    for row in A:
        print( " ".join([f"  {x} " if x is not None else " -- " for x in row]))

    print()
    for row in P:
        print( " ".join([f"{x:.2f}" if x is not None else " -- " for x in row]))
