import csv
import numpy as np


def entropy(probabilities):
    """
    Computes H(X)

    X is given as a numpy array, all elements of the array
    are assumed to represent the distribution (the shape
    of the array is not meaningful)

    """

    # Avoid situations where log can't be computed
    logs = np.log2(probabilities[probabilities != 0])
    return - np.sum(probabilities * logs)


def joint_entropy(X_and_Y):
    """
    Compute H(X ∩ Y)

    X_and_Y : joint probabilitiesof X and Y, a 2D numpy array such that
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
    Recall that P(X ∩ Y) = P(X | Y) P(Y) => We can use the marginal
    => P(X_i ∩ Y_j) = P(X_i|Y_j) * P(Y_j)
    """

    # Compare number of columns in X_given_Y (that's number of Y's)
    # to length of the Y vector.
    assert X_given_Y.shape[1] == Y.shape[1], f"Shape don't match : {X_given_Y.shape[1]} != {Y.shape[1]}"

    # z helps to compute combinations of i and j indices.
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


def joint_entropy2(P) #X_and_Y_and_Z):
    """
    FIXME
    I'm afraid this suggests P(X,Y,Z) = P(X)P(Y)P(Z) which is true
    only if X ⊥ Y ⊥ Z. No ?


    P(X,Y,Z) =

      X     Y      Z       P(X and Y and Z)
    -----------------------------------------
      true  false  true    0.1
      false false  true   0.05
      ...


    P(Z) = sum_X sum_Y P(...)


    H(a and b) = - sum P(A and B) log P(A and B)

    """

    return entropy(X_and_Y_and_Z)


def cond_mutual_information(X_and_Z, Y_and_Z, X, Y, Z):
    """
    I(X,Y|Z) = H(X,Z)+ H(Y,Z) - H(Z) - H(X,Y,Z)

    H(X,Y,Z) = H(X,Y|Z)+H(Z) => Stef: H(X,Y,Z) = H(X | Y,Z) + H(Y | Z) + H(Z)  (chain rule)

                                               = H(Y,X|Z) - H(Y|Z) + H(Y | Z) + H(Z) (see rule below)
                                               = H(Y,X|Z) + H(Z) ( - H(Y|Z) + H(Y | Z) == 0 !)

    H(X,Y|Z) = H(X|Z) + H(Y|X,Z) => from the course (Slide 9, point 2)
    => H(Y|X,Z) = H(X,Y|Z) - H(X|Z)
    ... renaming X <-> Y
    => H(X|Y,Z) = H(Y,X|Z) - H(Y|Z)


    which we prove like this :

    H(X,Y|Z) = - Σ_X Σ_Y Σ_Z P(X,Y ∩ Z) log(P(X,Y|Z))
             = - Σ_X Σ_Y Σ_Z P(X,Y,Z) log(P(Y|Z,X) P(X|Z))  # Because P(X,Y|Z) = P(X,Y,Z)/P(Z)
                                                                               = (P(X,Y,Z)/P(X,Z)))*(P(X,Z)/P(Z)
                                                                               = P(Y|X,Z) * P(X|Z)
             = - Σ_X Σ_Y Σ_Z P(X,Y,Z) log P(Y|Z,X)  # because log of product
               - Σ_X Σ_Y Σ_Z P(X,Y,Z) log P(X|Z)
             = H(Y|Z,X)  # because definition of H
               - Σ_Y P(Y|Z,X) Σ_X Σ_Z P(Z,X) log P(X|Z)
             = H(Y|Z,X)
               + 1 * H(Z,X)  # Because Y probability distribution => Σ_i Y_i = 1
             = H(Y|Z,X) + H(X,Z)


    X_and_Z = found by marginalizing Y out of X_Y_Z
    BUT !!! margnialization can only be done if we know the values of the random varaibles X,Y and Z
    => we can't just have a probability table, we must have a distribution table P(X=x_i AND Y=y_i AND Z=Z_i) == 0.19... for all x_i, y_i, z_i.

    => only  X_Y_Z is needed, other P(...) can be deduced by appropriate margnilaization.

    """

    return joint_entropy(X_and_Z) + joint_entropy(Y_and_Z) - entropy(Z) - joint_entropy2(X_Y_Z)

def cond_joint_entropy(X_Y_Z):
    """
    Computes H(X,Y|Z)

    H(X,Y|Z) this term is obtained here: H(X,Y,Z) = H(X,Y|Z)+H(Z)
    H(X,Y|Z) = H(X,Y,Z) - H(Z) (see above)

    """

    return joint_entropy2(X_Y_Z) - entropy(Z + marginalzation of Y and X)


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
    p = np.array([0.5, 0.5])
    assert entropy(p) == 1, "2 outcomes needs 1 bit"

    p = np.array([0, 1])
    assert entropy(p) == 0, f"Only one possible  outcome needs 0 bit not {entropy(p)}"

    p = np.array([0.25, 0.25, 0.25, 0.25])
    assert entropy(p) == 2, "4 outcomes needs 2 bits"

    X_and_Y = np.array([[0.25, 0.25], [0.25, 0.25]])
    h = joint_entropy(X_and_Y.flatten())
    assert h == 2, f"{h}"

    X_given_Y = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]])
    Y = np.array([[0.1, 0.2, 0.7]])
    print(conditional_entropy(X_given_Y, Y))

    # FIXME fails
    # X = np.array([0.5, 0.5])
    # Y = np.array([0.5, 0.5])
    # X_and_Y = np.array([[0.25, 0.25], [0.25, 0.25]])
    # print(mutual_information(X, Y, X_and_Y))

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
