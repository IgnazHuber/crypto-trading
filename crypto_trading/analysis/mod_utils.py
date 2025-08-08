# analysis/mod_utils.py

import pandas as pd

def add_columns_from_result(df, result, colname=None):
    if isinstance(result, pd.DataFrame):
        for col in result.columns:
            df[col] = result[col]
    elif isinstance(result, pd.Series):
        if colname:
            df[colname] = result
        else:
            df[result.name if result.name else "value"] = result
    elif isinstance(result, dict):
        for key, val in result.items():
            df[key] = val
    else:
        if colname:
            df[colname] = result
        else:
            df["value"] = result
