from sklearn.preprocessing import LabelEncoder


def get_categorical_attributes(df):
    categorical_attributes = []
    for col in df.columns:
        if df[col].dtype == "object" or df[col].dtype == "category":
            categorical_attributes.append(col)
    print(categorical_attributes)
    return categorical_attributes


def encode_categorical_attributes(df_num, categorical_attributes):
    # encode categorical columns to numerical
    for col in categorical_attributes:
        le = LabelEncoder()
        df_num[col] = le.fit_transform(df_num[col])
        le_map = dict(zip(le.classes_, le.transform(le.classes_)))
        print('Attribute: ' + col)
        print(le_map)
