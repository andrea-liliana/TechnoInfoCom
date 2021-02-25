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

    """ Entropy of a variable X is - sum p_i log p_i where p_i is the
    probability of the variable to have value X_i """

    for cname in sorted(df.columns):
        vc = df[cname].value_counts(normalize=True)
        a = np.array(vc.values)
        entropy = -np.sum(a * np.log2(a))
        print(f"{cname}\t{len(vc.values)}\tH={entropy:.3f}")

    # Question 7 -----------------------------------------------------

    """ H( D | X) = sum p(d,x) log p(d,x) / p(d)
    where D and X are r.v.
    """

    # All variables, not the disease
    vnames = set(list(df.columns)) - set(['DIS'])

    # Compute probabilities of each value of the disease r.v.
    prob_dis = df['DIS'].value_counts(normalize=True).to_dict()

    for vname in sorted(vnames):  # list(sorted(vnames))[5:6]:

        h = 0

        # Compute for all combination of values of the r.v. D and X
        # For each of the combinatin, count how many times it
        # appears in the dataset. Dividing by the size of the data
        # set, we obtain the probability for that combination, that
        # is a joint probability : P(D=d_i, X=x_j).
        # For example : P(D=steatose, X=age>40)

        for gname, size in df.groupby(['DIS', vname]).size().items():
            #print(f"{gname[0]}:{dc[gname[0]]:.3f} * {gname[1]}:{vc[gname[1]]:.3f}")
            #print(f"{gname} {size}")

            p_dis_x = size / len(df)
            p_dis = prob_dis[gname[0]]
            h += - p_dis_x * np.log2(p_dis_x / p_dis)

        print(f"H(D|{vname}) = {h:.3f}")

    """
    For JAU (jaundice; yes/no), H(DIS|Jaundice) = 0.334
    For BIL (bilirunine; yes/no), H(DIS|Bilirubine) = 0.239

    """
