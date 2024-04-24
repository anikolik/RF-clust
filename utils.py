import numpy as np
import pandas as pd

# for calculating perfromance 
from sklearn.metrics import mean_absolute_error, median_absolute_error

def read_fold_data(directory: str, fold_number: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Read X_train, y_test, X_test, y_test from csv file.
    """
    X_train = pd.read_csv(f"{directory}/X_train_fold={fold_number}.csv"
                         , index_col=["f_id", "i_id"], dtype={"f_id": int, "i_id": int})

    X_test = pd.read_csv(f"{directory}/X_test_fold={fold_number}.csv"
                         , index_col=["f_id", "i_id"], dtype={"f_id": int, "i_id": int})

    y_train = pd.read_csv(f"{directory}/y_train_fold={fold_number}.csv"
                          , index_col=["f_id", "i_id"], dtype={"f_id": int, "i_id": int})

    y_test = pd.read_csv(f"{directory}/y_test_fold={fold_number}.csv"
                        , index_col=["f_id", "i_id"], dtype={"f_id": int, "i_id": int})

    return X_train, y_train, X_test, y_test

def mae(df, col_name):
    """
    Function to calculate MAE.
    """
    mae = mean_absolute_error(df['true'].values, df['predicted'].values)

    return pd.Series({col_name: mae})


def mdae(df, col_name):
    """
    Function to calculate MAE.
    """
    mdae = median_absolute_error(df['true'].values, df['predicted'].values)

    return pd.Series({col_name: mdae})
