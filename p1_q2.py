import pandas as pd

if __name__ == "__main__":

    df = pd.read_csv('P1_medicalDB.csv')
    print(df)

    proba = df.copy()

    for cname in df:

        #print(df[cname].value_counts())
        a = df[cname].value_counts() / len(df)
        #print(a.to_dict())
        proba[cname] = proba[cname].map(
            a.to_dict())

    proba['joint'] = proba.product(1)
    print(proba)
