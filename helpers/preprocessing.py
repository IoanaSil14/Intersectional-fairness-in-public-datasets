from sklearn.preprocessing import LabelEncoder, MinMaxScaler


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


# normalize data, convert to numerical
def encode_and_scale(data, target):
    categorical_attributes = get_categorical_attributes(data)
    df_numerical = data.copy()
    encode_categorical_attributes(df_numerical, categorical_attributes)
    scaler = MinMaxScaler()
    for col in df_numerical.columns:
        if col not in categorical_attributes and col != target:
            df_numerical[col] = scaler.fit_transform(df_numerical[[col]])
    return df_numerical
