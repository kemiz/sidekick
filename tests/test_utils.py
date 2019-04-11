import pandas as pd

from sidekick import utils


def test_balance_dataframe():
    target_column = 'target'
    df = pd.DataFrame({'feat': [1, 2, 3], 'target': ['a', 'b', 'b']})
    balanced_df = utils.balance_dataset(df, target_column)
    counts = balanced_df[target_column].value_counts().to_dict()
    assert(counts['a'] == counts['b'])
