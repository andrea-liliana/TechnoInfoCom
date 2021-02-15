import numpy as np


def entropy(probabilities):
    return - np.sum(probabilities * np.log2(probabilities))


def joint_entropy(jp):
    """
    jp : joint probabilities, a 2D numpy array such that
    jp[i,j] == P[X_i ^ Y_j]
    """
    return entropy(jp)


def conditional_entropy(X_given_Y, Y):
    """ cp : conditional probability P(X|Y)
    mp: marginal probability P(Y)

    Recall P(X,Y) = P(X|Y) P(Y)

    => P(X_i, Y_j) = P(X_i|Y_j) * P(Y_j)
    """

    # FIXME not tested yet !!!

    z = np.tile(Y, (X_given_Y.shape[0], 1))

    return - X_given_Y * z * np.log2(X_given_Y)



if __name__ == "__main__":
    p = np.array([0.5, 0.5])
    assert entropy(p) == 1, "2 outcomes needs 1 bit"

    p = np.array([0.25, 0.25, 0.25, 0.25])
    assert entropy(p) == 2, "4 outcomes needs 2 bits"

    jp = np.array([[0.25, 0.25], [0.25, 0.25]])
    print(jp)
    assert joint_entropy(jp.flatten()) == 2, f"{joint_entropy(jp)}"
