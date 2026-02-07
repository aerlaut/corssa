import pandas as pd
from sklearn.model_selection import train_test_split

def map_dssp_to_3state(df: pd.DataFrame, outcolname='dssp3', as_categorical=True) -> pd.DataFrame:
    df[outcolname] = df['dssp'].map({
        'H': 'H',
        'G': 'H',
        'I': 'H',
        'P': 'H',
        'B': 'E',
        'E': 'E',
        '.': '.',
        'T': '.',
        'S': '.'
    })

    if as_categorical:
        df[outcolname] = df[outcolname].astype('category')

    return df

def split_data(df: pd.DataFrame, test_size=0.2, random_state=42, feature_col='dssp3'):

    dist_cols = [col for col in df.columns if col.startswith('dist_')]
    angle_cols = [col for col in df.columns if col.startswith('angle_') or col.startswith('dihedral_')]
    neighbor_cols = [col for col in df.columns if col.startswith('neighbor_')]

    X = df[angle_cols + dist_cols + neighbor_cols +["aa"]]
    y = df[feature_col]

    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)