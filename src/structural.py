import pandas as pd


def get_null_df(pd_df: pd.DataFrame, drop_zero_cnt: bool = True) -> pd.DataFrame:
    """ """
    null_df = pd_df.isna().sum() / pd_df.shape[0]
    if drop_zero_cnt:
        null_df = null_df[null_df > 0]
    return null_df.sort_values(ascending=False)


def impute_mode(pd_df: pd.DataFrame, columns: list):
    """ """
    print("Starting Impute Mode to Categorical Columns")
    null_columns = [c for c in get_null_df(pd_df).index.tolist() if c in columns]
    print(f"Columns with Null Values => {null_columns}")
    for c in null_columns:
        print(f"\t Imputing mode for column {c}")
        mode = pd_df.groupby(c)[c].count().sort_values(ascending=False).index[0]
        pd_df[c] = pd_df[c].fillna(mode)
    print("Process Finished")
    return pd_df


def balance_data(
    pd_df: pd.DataFrame, target_column_name: str, labels: list = [0, 1]
) -> pd.DataFrame:
    """
    For datasets whose distribution of a binary target labels is unbalanced:
    - return a dataset balanced.
    - retain the maximum number of rows for the majority class.
    """
    print("Creating Balanced Dataset")
    assert not set(pd_df[target_column_name].unique()).difference(set(labels))

    # Separate Dataframes into Positive & Negative Labels
    pos_df = pd_df[pd_df[target_column_name] == 1]
    neg_df = pd_df[pd_df[target_column_name] == 0]
    pos_cnt = pos_df.shape[0]
    neg_cnt = neg_df.shape[0]
    print(
        f"Positive Label Record Count => {pos_cnt}, Negative Label Record Count => {neg_cnt}"
    )
    assert pos_cnt != neg_cnt, "Dataset is already balanced."

    # Define Majority & Minority CLasses
    maj_df = pos_df if pos_cnt > neg_cnt else neg_df
    min_df = pos_df if pos_cnt < neg_cnt else neg_df

    # Randomly Sample Majority Class == cnt minotiry class
    maj_df = maj_df.sample(min_df.shape[0])
    print(f"Downsampling majority label.  New Record Count => {maj_df.shape[0]}")

    pd_df = pd.concat([maj_df, min_df])
    print(f"Returning dataset with record count => {pd_df.shape[0]}")
    return pd_df


def dim_redux(
    pd_df: pd.DataFrame, target_column: str, threshold: float
) -> pd.DataFrame:
    """ """
    # Subset DataFrame
    col_df = pd_df[[target_column]]

    # Get Number of Levels
    num_levels = len(col_df[target_column].unique())
    print(f"Pre-Transformation Number of Levels => {num_levels}")

    # Group DataFrame & Get Count By Level
    group_df = col_df.groupby(target_column)[target_column].count()

    # Construct Levels DataFrame
    levels_df = pd.DataFrame(
        {
            "FEATURES": group_df.index,
            "LEVELS": group_df.values,
            "PCT": group_df.values / group_df.sum(),
        }
    )
    levels_df.sort_values(by="PCT", ascending=False, inplace=True)
    levels_df["CUMSUM"] = levels_df.PCT.cumsum()
    levels_df["THRESHOLD"] = list(
        map(lambda x: 1 if x <= threshold else 0, levels_df.CUMSUM.values)
    )

    # Get Feature Set Whose Cumulative Sum Rows <= Threshold
    feature_set = levels_df[levels_df.THRESHOLD == 1].FEATURES.values.tolist()

    # Assign New Values to Column (> Threshold Assign OTHER)
    pd_df[target_column] = list(
        map(lambda x: x if x in feature_set else "OTHER", pd_df[target_column].values)
    )

    # Log
    num_levels = len(pd_df[target_column].unique())
    print(f"Post-Transformation Number of Levels => {num_levels}")

    return pd_df


def impute_mean(pd_df: pd.DataFrame, columns: list):
    """ """
    print(f"Replacing null values with mean for columns {columns}")

    for c in columns:
        # Calculate Null Pct
        null_pct = round(pd_df[c].isna().sum() / pd_df.shape[0], 2)

        if null_pct > 0:
            # Calculate Feature Mean
            mean = pd_df[c].mean()
            # Impute Mean
            print(f"\tFeature {c} has null pct {null_pct}.  Imputing mean of {mean}")
            pd_df[c] = pd_df[c].fillna(mean)
    print("Process Completed Successfully.  Returning DataFrame")
    return pd_df


class Encoder:
    # TODO: Add Decoder method.
    def __init__(self, dataframe: pd.DataFrame, columns: list):
        self.dataframe = dataframe
        self.columns = columns
        self.encoder = {}
        self.decoder = {}
        print(f"{__class__} Instantiated Successfully")

    def _build_encoder(self):
        print("Building Encoder for Columns => {columns}")
        for c in self.columns:
            levels = self.dataframe[c].unique()
            self.encoder[c] = {x: y for x, y in zip(levels, range(len(levels)))}
        return self

    def _build_decoder(self):
        print("Building Decoder")
        for c in self.encoder:
            self.decoder[c] = {x: y for y, x in self.encoder[c].items()}
        return self

    def _encode_columns(self):
        for c in self.encoder:
            self.dataframe[c] = list(
                map(lambda x: self.encoder[c].get(x, None), self.dataframe[c].values)
            )
            print(f"\tColumn Encoded => {c}")
        return self

    def _decode_columns(self):
        for c in self.decoder:
            self.dataframe[c] = list(
                map(lambda x: self.decoder[c].get(x, None), self.dataframe[c].values)
            )
            print(f"\tColumn Decoded => {c}")
        return self

    def encode(self):
        self._build_encoder()
        self._build_decoder()
        self._encode_columns()
        print("Encoding Completed")
        return self.dataframe

    def decode(self):
        assert self.encoder, "Features must be encoded first"
        assert self.decoder, "Features must be encoded first"
        self._decode_columns()
        print("Decoding Completed")
        return self.dataframe


class OutlierDector:
    def __init__(
        self, pd_df, target_columns, threshold: int = 3, drop_outliers: bool = True
    ):
        self.pd_df = pd_df
        self.target_columns = target_columns
        self.threshold = threshold
        self.drop_outliers = drop_outliers
        self.suffix = "OutlierDetection"
        print(f"Class {__class__} instantiated successfully")

    def _get_mean(self, column: str):
        return self.pd_df[column].dropna().mean()

    def _get_std(self, column: str):
        return self.pd_df[column].dropna().std()

    def _get_outliers(self, column):
        print(f"Identifying outliers for column => {column}")
        mu = self._get_mean(column)
        std = self._get_std(column)
        col_new = f"{column}-{self.suffix}"
        self.pd_df[col_new] = list(
            map(
                lambda x: 1 if ((x - mu) / std) > self.threshold else 0,
                self.pd_df[column],
            )
        )
        print(f"\t{self.pd_df[col_new].sum()} outliers were detected")
        return self

    def _drop_outliers(self, column):
        if self.drop_outliers:
            for c in [c for c in self.pd_df.columns if self.suffix in c]:
                print(f"\tDropping outliers for column => {c}")
                self.pd_df = self.pd_df[self.pd_df[c] != 1]
                print(f"\tDropping column {c}")
                self.pd_df = self.pd_df.drop(f"{c}", axis=1)
        return self

    def detect(self):
        print(f"Starting detection w/ data dimensions => {self.pd_df.shape}")
        for c in self.target_columns:
            self._get_outliers(c)
            self._drop_outliers(c)

        print(f"Process completed. Final data dimensions => {self.pd_df.shape}")
        return self
