import numpy as np
import openml
from sklearn.preprocessing import MinMaxScaler

from helpers.preprocessing import get_categorical_attributes, encode_categorical_attributes


def initial_dataset_preprocess():
    ml_ds = openml.datasets.get_dataset(43141)
    df, *_ = ml_ds.get_data()
    #print(ml_ds.description[:500])
    df.rename(columns={'MAR': 'marital status', 'RAC1P': 'race', 'AGEP': 'age', 'PINCP': 'target', 'SEX': 'sex'},
              inplace=True)

    df["age"] = df["age"].apply(categorize_age)
    df['race'] = np.where(df['race'] == 1, 1, 2)
    df['marital status'] = df['marital status'].replace(3, 2)
    df['marital status'] = df['marital status'].replace(4, 2)
    df['marital status'] = df['marital status'].replace(5, 3)

    return df


def categorize_age(age):
    if age < 30:
        return '0'
    if (age >= 30) and (age < 50):
        return '1'
    if age >= 50:
        return '2'



