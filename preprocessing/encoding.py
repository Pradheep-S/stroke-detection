from sklearn.preprocessing import LabelEncoder, StandardScaler

def encode_and_scale(df):
    categorical_cols = [
        'gender', 'ever_married', 'work_type',
        'Residence_type', 'smoking_status'
    ]

    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    X = df.drop('stroke', axis=1)
    y = df['stroke']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y
