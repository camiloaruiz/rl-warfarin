import pandas as pd

def display_full_df(df, max_rows = None, max_columns = None, rows_from_top = False):
    if (rows_from_top):
        with pd.option_context('display.max_columns', max_columns):
            display(df.iloc[:max_rows, :])
    else:
        with pd.option_context('display.max_rows', max_rows, 'display.max_columns', max_columns):
            display(df)