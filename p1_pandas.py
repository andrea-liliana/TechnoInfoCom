import pandas as pd
import numpy as np

def marginalize(dist_table: pd.DataFrame, variables, leave_values=False):
    """
    Marginalize variables out of probability distribution table.
    The probability distribution table is a DataFrame of which
    the first columns are the random variables values and the
    last column is the probability of having this combination
    of values. For example : X, Y, P(X ^ Y)

    Each column of variables are labelled with capital letter
    denoting the name of the variable.
    """

    p_column = dist_table.columns[-1]
    r = dist_table.groupby(variables).agg(summed=(p_column, 'sum')).reset_index()
    r['summed'] /= r['summed'].sum()

    if not leave_values:
        return r['summed']
    else:
        if type(variables) == str:
            p_name = f"P({variables})"
        else:
            p_name = f"P({'^'.join(variables)})"
        r.rename(columns={'summed': p_name}, inplace=True)
        return r

def entropy(probabilities):
    """
    Computes H(X)

    X is given as a numpy array, all elements of the array
    are assumed to represent the distribution (to the shape
    of the array is not meaningful)
    """

    # Avoid situations where log can't be computed
    non_zero = probabilities[probabilities != 0]
    return - np.sum(non_zero * np.log2(non_zero))


def joint_entropy2(x_and_y):
    # Compute the joint entropy
    return entropy(x_and_y["P(X^Y)"])


def cond_entropy2(x_given_y, y):
    # Compute the conditional entropy

    # First, relate P(X_i|Y_j) to P(Y_j)
    r = pd.merge(x_given_y, y)
    return - np.sum(r["P(X|Y)"] * r["P(Y)"] * np.log2(r["P(X|Y)"]))


def mutual_information(x_and_y, var_x="X", var_y="Y"):
    # I(X;Y) = H(X) + H(Y) - H(X,Y) (See wikipedia)

    x = marginalize(x_and_y, var_x)
    y = marginalize(x_and_y, var_y)
    p_x_and_y = x_and_y[x_and_y.columns[-1]]  # last column is the probabilities

    return entropy(x) + entropy(y) - entropy(p_x_and_y)


def joint_entropy3(x_and_y_and_z):
    # Compute joint entropy on three variables
    return entropy(x_and_y_and_z["P(X^Y^Z)"])


def cond_entropy3(x_and_y_and_z):
    # H(X,Y|Z) = H(X,Y,Z) - H(Z) (see above)
    z = marginalize(x_and_y_and_z, "Z")
    return entropy(x_and_y_and_z['P(X^Y^Z)']) - entropy(z)


def cond_information3(x_and_y_and_z):
    # I(X;Y|Z) = H(X,Z)+ H(Y,Z) - H(Z) - H(X,Y,Z)

    x_and_z = marginalize(x_and_y_and_z, ["X", "Z"])
    y_and_z = marginalize(x_and_y_and_z, ["Y", "Z"])
    z = marginalize(x_and_y_and_z, "Z")

    return entropy(x_and_z) + entropy(y_and_z) - entropy(z) - entropy(x_and_y_and_z['P(X^Y^Z)'])


def cond_entropy2b(x_and_y, variable):
    """ Compute the conditional entropy based on *joint*
    probability table.

    Slide 3, course 2 : H(X|Y) = − Σ Σ P(Xi ∩ Yj) log P(Xi | Yj)
    Applying P(a|b) = P(a,b) / P(b), we get :
    H(X|Y) = − Σ Σ P(Xi ∩ Yj) log P(Xi ∩ Yj) / P(Yj)
    """

    join_p_column = x_and_y.columns[-1]

    y = marginalize(x_and_y, variable)
    r = pd.merge(x_and_y, y)

    return - np.sum(x_and_y[join_p_column] * np.log2(r[join_p_column] / y))


def implementation():

    # Some test data sets

    x_given_y = pd.DataFrame(
        [[True, True, 0.25],
         [True, False, 0.25],
         [False, True, 0.2],
         [False, False, 0.3]],
        columns=["X", "Y", "P(X|Y)"])

    x_and_y = pd.DataFrame(
        [[True, True, 0.25],
         [True, False, 0.25],
         [False, True, 0.2],
         [False, False, 0.3]],
        columns=["X", "Y", "P(X^Y)"])

    y = pd.DataFrame([[True, 0.3],
                          [False, 0.7]],
                         columns=["Y", "P(Y)"])

    x_and_y_and_z = pd.DataFrame(
        [[True, True, True, 0.25],
         [True, True, False, 0.25],
         [True, False, True, 0.2],
         [True, False, False, 0.3],
         [False, True, True, 0.25],
         [False, True, False, 0.25],
         [False, False, True, 0.2],
         [False, False, False, 0.3]],
        columns=["X", "Y", "Z", "P(X^Y^Z)"])

    # Q1
    entropy(x_and_y)

    # Q2
    joint_entropy2(x_and_y)

    # Q3
    cond_entropy2(x_given_y, y)

    # Q4
    mutual_information(x_and_y)

    # Q5
    joint_entropy3(x_and_y_and_z)
    cond_entropy3(x_and_y_and_z)
    cond_information3(x_and_y_and_z)



def medical_diagnosis():
    df = pd.read_csv('P1_medicalDB.csv')
    print(df)

    # Compute joint proabilities and add them to the table
    jpd = df.groupby(list(df.columns)).size().reset_index()
    jpd[0] /= jpd[0].sum()  # r[0] is the new counts columns

    # Question 6
    for var_name in jpd.columns[0:-1]:
        print(f"{var_name}\tcard:{len(jpd[var_name].unique())} {entropy(marginalize(jpd, var_name)):.3f}")

    # Question 7

    # All variables names, without the disease
    vnames = set(list(df.columns)) - set(['DIS'])

    for vname in vnames:
        """ Compute the conditional entropy based on joint probability table.

        Slide 3, course 2 : H(X|Y) = − Σ Σ P(Xi ∩ Yj) log P(Xi | Yj)
        Applying P(a|b) = P(a,b) / P(b), we get :
        H(X|Y) = − Σ Σ P(Xi ∩ Yj) log P(Xi ∩ Yj) / P(Yj)

        Renaming
        H(D|S) = − Σ Σ P(Di ∩ Sj) log P(Di ∩ Sj) / P(Sj)
        """

        dis_symptom = marginalize(jpd, ["DIS", vname], True)
        symptom = marginalize(dis_symptom, vname, True)
        m = pd.merge(dis_symptom, symptom)  # relate (x_i^y_j)'s to y_j's
        p_di_sj = m[m.columns[-2]]  # P(x_i^y_j)
        p_sj = m[m.columns[-1]]  # P(y_j)
        e = - np.sum(p_di_sj * np.log2(p_di_sj / p_sj))

        if vname in ('JAU','BIL'):
            mark = "***"
        else:
            mark = ""

        print(f"H(DIS|{vname})\t{e:.3f} {mark}")

    # Question 8

    obesity_age = marginalize(jpd, ["obesity","age"], True)

    print(mutual_information(obesity_age, "obesity","age"))
    print(mutual_information(obesity_age, "age","obesity"))

    # Question 9

    for vname in vnames:
        dis_symptom = marginalize(jpd, ["DIS", vname], True)
        mi = mutual_information(dis_symptom, "DIS", vname)
        print(f"I(DIS;{vname})\t{mi:.3f}")

    # Question 10

    # Compute joint proabilities and add them to the table

    jpd = df[df.DIS.isin(['steatosis', 'healthy'])].groupby(list(df.columns)).size().reset_index()
    jpd[0] /= jpd[0].sum()  # r[0] is the counts columns

    for vname in vnames:
        dis_symptom = marginalize(jpd, ["DIS", vname], True)
        symptom = marginalize(dis_symptom, vname, True)
        m = pd.merge(dis_symptom, symptom)  # relate (x_i^y_j)'s to y_j's
        p_di_sj = m[m.columns[-2]]  # P(x_i^y_j)
        p_sj = m[m.columns[-1]]  # P(y_j)

        e = - np.sum(p_di_sj * np.log2(p_di_sj / p_sj))
        mi = mutual_information(dis_symptom, "DIS", vname)

        print(f"DIS|{vname:10s}:\tH={e:.3f} I={mi:.3f}")

    # import code
    # code.interact(local=dict(globals(), **locals()))




if __name__ == "__main__":

    implementation()
    medical_diagnosis()
