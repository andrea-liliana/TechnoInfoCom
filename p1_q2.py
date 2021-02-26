import numpy as np
import pandas as pd

if __name__ == "__main__":

    df = pd.read_csv('P1_medicalDB.csv')
    print(df)

    # I build a table with same structure as the original
    # data frame but where each value is replaced by the probability
    # of that value occuring.

    proba = df.copy()

    # One column at a time
    for cname in df:

        # Count how many times each value appears in the column and
        # divide to get its proportion, that is, its probability

        a = df[cname].value_counts(normalize=True)

        # Replace values by probabilities
        proba[cname] = proba[cname].map(
            a.to_dict())

    # Compute the joint probability on each *row*
    # by multiplying all columns together.
    proba['joint'] = proba.product(axis=1)

    # Clean values so we have a probability distribution
    # (i.e. all probabilities sum to one)
    proba['joint'] = proba['joint'] / proba['joint'].sum()

    print(proba['joint'].sum())

    # Question 6 -----------------------------------------------------

    """ Entropy of a variable X = - sum p_i log p_i where p_i is the
    probability of the variable to have value X_i

    In general, the more possibilities for a given variable, the more
    uncertainty, that is, entropy.

    Why is it true ?

    """

    for cname in sorted(df.columns):
        vc = df[cname].value_counts(normalize=True)
        a = np.array(vc.values)
        entropy = -np.sum(a * np.log2(a))
        print(f"{cname}\t{len(vc.values)}\tH={entropy:.3f}")

    # Question 7 -----------------------------------------------------

    """ H( D | X) = sum p(d,x) log p(d,x) / p(d)
    where D and X are r.v.

    Should we compute H(D == healthy | X)
    OR should we compute H(D | X) ?
    answer = H(D | X)


    """

    # All variables names, without the disease
    vnames = set(list(df.columns)) - set(['DIS'])

    # Compute probabilities of each value of the disease r.v. DIS
    prob_dis = df['DIS'].value_counts(normalize=True).to_dict()

    for vname in sorted(vnames):  # list(sorted(vnames))[5:6]:

        h = 0

        # Find all combinations of values of the r.v. DIS and X For
        # each of the combination, count how many times it appears in
        # the dataset.  Dividing by the size of the data set, we
        # obtain the probability for that combination, that is a joint
        # probability : P(D=d_i, X=x_j).  For example : P(D=steatose,
        # X=age>40)

        for gname, size in df.groupby(['DIS', vname]).size().items():

            # Compute P(DIS,X)
            p_dis_x = size / len(df)

            # Compute P(DIS = ...), for example P(DIS == steatosis)

            # P(DIS) = # persons with steatosis OR PBC / # persons

            # FIXME most prbably a bug !
            p_dis = prob_dis[gname[0]]

            # Compute H(DIS | X)
            h += - p_dis_x * np.log2(p_dis_x / p_dis)

            print(f"P({gname[0]},{gname[1]}) = {p_dis_x}")

        print(f"H(D|{vname}) = {h:.3f}\n")

    """
    H(DIS)=0.897 (see question 6)


    *** For JAU (jaundice; yes/no),   H(DIS|Jaundice) = 0.334

    If we test for Jaundice(yes or no), then the result of a test for the
    disase (steatosis or pbc or healthy) will be more certain.

    NOT THE SAME AS : If someone HAS jaundice, then it's more probable
    that he has a disease.


    *** For BIL (bilirunine; yes/no), H(DIS|Bilirubine) = 0.239

    If we test for Bilirubine(yes or no), then the result of a test
    for the disase (steatosis or pbc or healthy) will be more certain.


    If one knows a person has jaundice, then one will be less suprised
    when one will discover its disease.




    If one knows a person has bilirubine, the we will be even less
    suprised when we will discover its disease.

    So testing for bilirubine tells more about the disease than
    testing jaundice.  If I know the BIL of someone, then I'll be able
    to better determine its DIS (better than if I know about its
    jaundice).

    FIXME : In the disease values, thre's the "healthy" part.
            How do we analyse that ?

    H(D|JAU) = 0.334, from probabilites :

    P(PBC,no) = 0.067
    P(PBC,yes) = 0.066
    P(healthy,no) = 0.7755 => healthy, without jaundice are most frequent
    P(healthy,yes) = 0.027
    P(steatosis,no) = 0.058
    P(steatosis,yes) = 0.0065


    H(D|BIL) = 0.239, from probabilites :

    P(PBC,normal) = 0.054
    P(PBC,abnormal) = 0.079
    P(healthy,normal) = 0.7915
    P(healthy,abnormal) = 0.011
    P(steatosis,normal) = 0.0595
    P(steatosis,abnormal) = 0.005

    """
