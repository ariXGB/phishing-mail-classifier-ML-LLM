import pandas as pd
from sklearn.model_selection import train_test_split

from Components.project_paths import PROJECT_ROOT


def load_data(filename):
    df = pd.read_csv(PROJECT_ROOT / "Data" / filename)
    return df

def load_and_split_data(filename: str, test_size: float, random_state: int) -> tuple:

    df = load_data(filename)
    train, test = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df['label'])

    return train, test
