import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def marginalize(dist_table: pd.DataFrame, variables, leave_values=False):
    """
    Marginalize variables out of probability distribution table.

    dist_table: The probability distribution table is a DataFrame of which
    the first columns are the random variables values and the
    last column is the probability of having this combination
    of values (joint probability). For example : X, Y, P(X ^ Y)
    Each column are labelled with capital letter
    denoting the name of the variable (for the columns containing
    variables).

    variables: teh marginalization target. So If one has P(X^Y) and
    want to get P(X) out of it, then X is such a target.
    """

    p_column = dist_table.columns[-1]
    r = dist_table.groupby(variables).agg(
        summed=(p_column, 'sum')).reset_index()
    r['summed'] /= r['summed'].sum()

    if not leave_values:
        # Only return probabilities, not a whole
        # contingency table
        return r['summed']
    else:
        # Return a fulle contingency table, with
        # nice colums headers.
        if type(variables) == str:
            p_name = f"P({variables})"
        else:
            p_name = f"P({'^'.join(variables)})"
        r.rename(columns={'summed': p_name}, inplace=True)
        return r


def entropy(probabilities: np.array):
    """
    Computes H(X)

    X is given as a numpy array, all elements of the array
    are assumed to represent the distribution (to the shape
    of the array is not meaningful)
    """

    # Avoid situations where log can't be computed
    non_zero = probabilities[probabilities != 0]
    return - np.sum(non_zero * np.log2(non_zero))


def joint_entropy(x_and_y: pd.DataFrame):
    """
    Computes the joint entropy H(X,Y)

    Expects a dataframe with three columns :
    - values of X
    - values of Y
    - P(X=x, Y=y) : probability distribution; must sum to one.
    """

    return entropy(x_and_y["P(X^Y)"])


def cond_entropy(x_given_y: pd.DataFrame, y: pd.DataFrame):
    """
    Compute the conditional entropy

    Expects a dataframe with three columns :
    - x_given_y: values of X|Y as table of rows (x,y,X=x|Y=y)
    - y: values of P(Y=y) as one column table
    """

    # First, relate P(X_i|Y_j) to P(Y_j)
    r = pd.merge(x_given_y, y)
    return - np.sum(r["P(X|Y)"] * r["P(Y)"] * np.log2(r["P(X|Y)"]))


def mutual_information(x_and_y: pd.DataFrame,
                       var_x: str = "X",
                       var_y: str = "Y"):
    """ Computes :

    I(X;Y) = H(X) + H(Y) - H(X,Y)

    Expects parameters :

    - x_and_y : a table (DataFrame) giving P(one row of the table)
    - var_x : name of the variable X, must be in the columns of x_and_y
    - var_y : name of the variable Y, must be in the columns of x_and_y
    """

    # The code here is a bit more dynamic so we can use this function
    # in part 2 of the problem statement.

    # Compute probabilities for all values of random variable X.
    x = marginalize(x_and_y, var_x)
    y = marginalize(x_and_y, var_y)
    # last column is the probabilities
    p_x_and_y = x_and_y[x_and_y.columns[-1]]

    # FIXME What happen sif the table has these
    # columns : X,Y,a,b,c,P(row)

    return entropy(x) + entropy(y) - entropy(p_x_and_y)


def joint_entropy3(x_and_y_and_z):
    # Compute joint entropy on three variables
    return entropy(x_and_y_and_z["P(X^Y^Z)"])


def cond_mutual_information(x_and_y_and_z):
    # I(X;Y|Z) = H(X,Z)+ H(Y,Z) - H(Z) - H(X,Y,Z)

    x_and_z = marginalize(x_and_y_and_z, ["X", "Z"])
    y_and_z = marginalize(x_and_y_and_z, ["Y", "Z"])
    z = marginalize(x_and_y_and_z, "Z")

    return entropy(x_and_z) - entropy(z) - entropy(x_and_y_and_z['P(X^Y^Z)']) + entropy(y_and_z)


def cond_joint_entropy(x_and_y_and_z):
    # H(X,Y|Z) = H(X,Y,Z) - H(Z) (see above)
    z = marginalize(x_and_y_and_z, "Z")
    return entropy(x_and_y_and_z['P(X^Y^Z)']) - entropy(z)


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
    joint_entropy(x_and_y)

    # Q3
    cond_entropy(x_given_y, y)

    # Q4
    mutual_information(x_and_y)

    # Q5
    joint_entropy3(x_and_y_and_z)
    cond_joint_entropy(x_and_y_and_z)
    cond_mutual_information(x_and_y_and_z)


def medical_diagnosis():
    df = pd.read_csv('P1_medicalDB.csv')
    print(df)

    # Compute joint proabilities and add them to the table

    """

    a,i,w
    a,i,w
    a,i,z

    =>

    a,i,w 2
    a,i,z 1
    """

    jpd = df.groupby(list(df.columns)).size().reset_index()

    """
    a,i,w 2 / sum(1+2) => P(a,i,w)
    a,i,z 1 / sum(1+2) => P(a,i,z)
    """

    jpd[0] /= jpd[0].sum()  # r[0] is the new counts columns; it's the last column

    # Question 6

    entropies = []
    cardinalities = []
    names = []

    """
    age, obseity, BIL, ...
    ----------------------------
    >40, y,       10
    <40, n,       11
    ...
    """

    for var_name in jpd.columns[0:-1]: # every column but the last one
        card = len(jpd[var_name].unique())
        # => >40, <40. => cardinality

        # H(age) = P(>40)*log(P(>40)) + P(<40)*log(P(<40))
        e = entropy(marginalize(jpd, var_name))

        names.append(var_name)
        cardinalities.append(card)
        entropies.append(e)

    with open("question6.inc","w") as fout:
        for vname, ent in sorted( zip(names, entropies), key=lambda t:t[1]):
            fout.write(f"{vname} & {ent:.3f} \\\\\n")

    plt.figure()
    plt.scatter(cardinalities, entropies)
    for i, name in enumerate(names):
        plt.annotate(name, (cardinalities[i], entropies[i]))
    plt.xticks([2,3,4])
    plt.xlabel("Cardinalities")
    plt.ylabel("Entropies")
    plt.savefig("entropiescardinalities.pdf")
    plt.show()


    # Question 7

    # All variables names excluding the disease (DIS)
    # Make a list(...) to keep order
    vnames = list(set(list(df.columns)) - set(['DIS']))

    entropies = []
    for vname in vnames:
        """ Compute the conditional entropy H(DIS|variable) based on joint
        probability table.

        age, obseity, BIL, ...
        ----------------------------
        >40, y,       10
        <40, n,       11
        ... for missing combinations of variables we have :
        ?,   ?,       ? => P(?,?,?) = 0

        X Y Z P(X,y,Z)
        --------------   --> FULL P(x,y,z) table.
        n n n 0.1
        n y n
        n n y
        n y y
        y n n
        y y n
        y n y
        y y y

        We can't reuse our funtion cond_entropy() because it expects
        P(X|Y) but the most direct thing we have is P(X^Y).

        Slide 3, course 2 : H(X|Y) = − Σ Σ P(Xi ∩ Yj) log P(Xi | Yj)
        Applying P(a|b) = P(a ^ b) / P(b), we get :
        H(X|Y) = − Σ Σ P(Xi ∩ Yj) log P(Xi ∩ Yj) / P(Yj)

        Renaming X -> D, T -> S(ymptom)
        H(D|S) = − Σ Σ P(Di ∩ Sj) log P(Di ∩ Sj) / P(Sj)
        """

        dis_symptom = marginalize(jpd, ["DIS", vname], True)  # P(D ∩ S)
        symptom = marginalize(dis_symptom, vname, True)  # P(S)

        m = pd.merge(dis_symptom, symptom)  # relate (d_i ^ s_j)'s to s_j's
        p_di_sj = m[m.columns[-2]]  # P(d_i^s_j)
        p_sj = m[m.columns[-1]]  # P(s_j)
        e = - np.sum(p_di_sj * np.log2(p_di_sj / p_sj)) # compute H

        entropies.append( (vname,e) )

    for vname, e in sorted(entropies, key=lambda p:p[1]):
        e = f"{e:.3f}"
        if vname in ('JAU', 'BIL'):
            vname = f"\\textbf{{{vname}}}"
            e = f"\\textbf{{{e}}}"

        print(f"{vname} & {e} \\\\")

    # Question 8

    obesity_age = marginalize(jpd, ["obesity", "age"], True)

    print(mutual_information(obesity_age, "obesity", "age"))
    print(mutual_information(obesity_age, "age", "obesity")) # swap variabkes to checking for bugs

    # Question 9

    print("-"*80)
    mutual_info = []
    for vname in vnames:
        dis_symptom = marginalize(jpd, ["DIS", vname], True)
        mi = mutual_information(dis_symptom, "DIS", vname)

        mutual_info.append((vname,mi))


    for vname, i in sorted(mutual_info, key=lambda p:p[1]):
        i = f"{i:.3f}"
        print(f"{vname} & {i} \\\\")


    # Question 10

    # Recompute joint proabilities excluding steatosis/healthy

    jpd = df[df.DIS.isin(['steatosis', 'healthy'])].groupby(
        list(df.columns)).size().reset_index()
    jpd[0] /= jpd[0].sum()  # r[0] is the counts columns

    mutual_info = []
    for vname in vnames:
        dis_symptom = marginalize(jpd, ["DIS", vname], True)
        symptom = marginalize(dis_symptom, vname, True)

        m = pd.merge(dis_symptom, symptom)  # relate (x_i^y_j)'s to y_j's
        p_di_sj = m[m.columns[-2]]  # P(x_i^y_j)
        p_sj = m[m.columns[-1]]  # P(y_j)

        e = - np.sum(p_di_sj * np.log2(p_di_sj / p_sj))
        mi = mutual_information(dis_symptom, "DIS", vname)

        mutual_info.append((vname,mi))
        # print(f"DIS|{vname:10s}:\tH={e:.3f} I={mi:.3f}")


    with open("question10.inc","w") as fout:
        for vname, mi in sorted(mutual_info, key=lambda p:p[1]):
            fout.write(f"{vname} & {mi:.3f} \\\\\n")



    # import code
    # code.interact(local=dict(globals(), **locals()))


if __name__ == "__main__":

    implementation()
    medical_diagnosis()
