import csv
import numpy as np


def entropy(probabilities):
    """
    Computes H(X)
    """

    # Avoid situations where log can't be computed
    logs = np.nan_to_num(np.log2(probabilities))
    return - np.sum(probabilities * logs)


def joint_entropy(X_and_Y):
    """
    Compute H(X ^ Y)

    X_and_Y : joint probabilities, a 2D numpy array such that
    X_and_Y[i,j] == P[X_i ^ Y_j]
    """
    return entropy(X_and_Y)


def conditional_entropy(X_given_Y, Y):
    """
    Computes H(X|Y)

    X_given_Y : conditional probability P(X|Y)
                X_i | Y_i : row(i), column(j)
    Y: marginal probability P(Y)

    Recall P(X,Y) = P(X|Y) P(Y)

    => P(X_i, Y_j) = P(X_i|Y_j) * P(Y_j)
    """

    # Compare nulber of columns in X_given_Y (that's number of Y's)
    # to length of the Y vector.
    assert X_given_Y.shape[1] == Y.shape[1], f"Shape don't match : {X_given_Y.shape[1]} != {Y.shape[1]}"

    z = np.tile(Y, (X_given_Y.shape[0], 1))

    return - np.sum(X_given_Y * z * np.log2(X_given_Y))



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
    probability = []
    for x1 in set(X):
        for x2 in set(Y):
            for x3 in set(Z):
                probability.append(np.mean(np.logical_and(X == x1, Y == x2, Z==x3)))

    return np.sum(-p * np.log2(p) for p in probability)

def cond_mutual_information(X_and_Z, Y_and_Z, X, Y, Z):
    """
    I(X,Y|Z) = H(X,Z)+ H(Y,Z) - H(Z) - H(X,Y,Z)
    H(X,Y,Z) = H(X,Y|Z)+H(Z)
    H(X,Y|Z) = H(X|Z) + H(Y|X,Z)
    """
    return joint_entropy(X_and_Z) + joint_entropy(Y_and_Z) - entropy(Z) - joint_entropy2(X,Y,Z)



if __name__ == "__main__":
    p = np.array([0.5, 0.5])
    assert entropy(p) == 1, "2 outcomes needs 1 bit"

    p = np.array([0, 1])
    assert entropy(p) == 0, f"Only one possible  outcome needs 0 bit not {entropy(p)}"

    p = np.array([0.25, 0.25, 0.25, 0.25])
    assert entropy(p) == 2, "4 outcomes needs 2 bits"

    X_and_Y = np.array([[0.25, 0.25], [0.25, 0.25]])
    assert joint_entropy(X_and_Y.flatten()) == 2, f"{joint_entropy(jp)}"

    X_given_Y = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]])
    Y = np.array([[0.1, 0.2, 0.7]])
    print(conditional_entropy(X_given_Y, Y))

    X = np.array([0.5, 0.5])
    Y = np.array([0.5, 0.5])
    X_and_Y = np.array([[0.25, 0.25], [0.25, 0.25]])
    print(mutual_information(X, Y, X_and_Y))

    PARAMS = {
        "Age" : ["<40", ">40"],
        "Sex" : ["man", "woman"],
        "Obesity" : ["thin","regular","overweight"],
        "Alcoholic antecedents" : ["yes", "no"],
        "Iron" : ["low", "normal", "high", "very high"],
        "Disease" : ["healthy", "PBC", "steatosis"],
        "Fatigue" : ["yes", "no"],
        "Triglycerides" : ["abnormal", "normal"],
        "Alanine transaminase" : ["abnormal", "normal"],
        "Aspartate transaminase" : ["abnormal", "normal"],
        "Gamma-glutamyl transpeptidase" : ["abnormal", "normal"],
        "CholesterolCHL" : ["low", "normal", "high"],
        "Anti-mitochondrial antibody" : ["yes", "no"],
        "Muscular ache" : ["yes", "no"],
        "Bilirubin" : ["abnormal", "normal"],
        "Itching" : ["yes", "no"],
        "Jaundice" : ["yes", "no"] }

    with open('P1_medicalDB.csv') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            print(row)

    for k,v in PARAMS.items():
        ent = entropy( np.ones((1,len(v))) / len(v))
        print(f"{k} : {ent}")

    v = PARAMS["Disease"]
    X = np.ones((1,len(v))) / len(v)

    for k,v in PARAMS.items():
        if k != "Disease":
            # X = disease; Y=other variable
            Y = np.ones((1,len(v))) / len(v)
            conditional_entropy(X_given_Y, Y)
            ent = entropy( np.ones((1,len(v))) / len(v))
            print(f"{k} : {ent}")

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



    """
