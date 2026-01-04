import pandas as pd

def clean_data(df):
    # Drop ID column
    df = df.drop(columns=['id'])

    # Fill missing BMI with median
    df['bmi'] = df['bmi'].fillna(df['bmi'].median())

    # Replace 'Unknown' smoking status
    df['smoking_status'] = df['smoking_status'].replace('Unknown', 'never smoked')

    return df
