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
        # Count each value appear in the column and
        # divide to get its proportion, that is,
        # its probabiliyt

        a = df[cname].value_counts() / len(df)

        # Replace values by probabilities
        proba[cname] = proba[cname].map(
            a.to_dict())

    # Compute the joint probability on each *row*
    # by multiplyin all rows together.
    proba['joint'] = proba.product(axis=1)

    # Clean values so we have a proabability distribution
    # (ie all probabilities sum to one)
    proba['joint'] = proba['joint'] / proba['joint'].sum()

    print(proba['joint'].sum())
