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

    for cname in df:
        vc = df[cname].value_counts(normalize=True)
        a = np.array(vc.values)
        entropy = -np.sum(a * np.log2(a))
        print(f"{cname}\t{vc.values}\t{entropy:.3f}")
