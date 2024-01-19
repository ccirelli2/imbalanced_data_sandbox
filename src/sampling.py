from src.preprocessing import structural
import pandas as pd
from collections import Counter
from imblearn.over_sampling import (
    SMOTE,
    RandomOverSampler,
)
from imblearn.combine import SMOTETomek

# from imblearn.combine import SMOTETomek, SMOTEENN


class Sampling:
    """
    TODO: Add undersampling of majority class.
    """

    def __init__(
        self,
        pd_df: pd.DataFrame,
        target_column: str,
        sampling_strategy: dict = {},
        technique: str = "oversample",
    ):
        self.pd_df = pd_df
        self.target_column = target_column
        self.sampling_strategy = sampling_strategy
        self.technique = technique
        self.class_balance = {}
        self.y = pd.DataFrame()
        self.X = pd.DataFrame()
        self.acceptable_strategies = ["smote", "oversample", "smote_tomek"]
        msg = "Sampling technique not recognized.  Acceptable options => {self.acceptable_strategies}"
        assert self.technique in (self.acceptable_strategies), msg
        print(f"Class {__class__} instantiated successfully")
        if not self.sampling_strategy:
            print(
                "Warning: No sampling strategy provided.  Utilizing balanced strategy"
            )

    def _get_class_balance(self):
        print("Calculating class balance")
        self.class_balance = Counter(self.pd_df[self.target_column])
        print(
            f"Class Balance => {self.class_balance}, Minority Pct. => {round(self.class_balance[1] / self.class_balance[0], 2)}"
        )
        return self

    def _split_xy(self):
        print("Spliting X & y")
        self.y = self.pd_df[self.target_column]
        self.X = self.pd_df.drop(self.target_column, axis=1)
        print(f"X dimensions => {self.X.shape}, y dimensions => {self.y.shape}")
        return self

    def _create_balanced_sampling_strategy(self):
        print("Creating balanced sampling strategy")
        assert self.class_balance
        self.sampling_strategy = {0: self.class_balance[0], 1: self.class_balance[0]}
        print(f"Sampling Strategy => {self.sampling_strategy}")
        return self

    def _oversample(self):
        print(f"Sampling using => {self.sampling_strategy}")
        ros = RandomOverSampler(
            random_state=42, sampling_strategy=self.sampling_strategy
        )
        X, y = ros.fit_resample(self.X, self.y)
        X[self.target_column] = y
        self.pd_df = X
        print(
            "Samling Completed", f"X dimensions => {X.shape}, y-dimensions => {y.shape}"
        )
        return self

    def _smote(self):
        print(f"Sampling using => {self.sampling_strategy}")
        assert not structural.get_null_df(
            self.pd_df
        ).sum(), "Smote requires zero null values"
        print(f"Smote sampling strategy => {self.sampling_strategy}")
        sampler = SMOTE(random_state=42, sampling_strategy=self.sampling_strategy)
        X, y = sampler.fit_resample(self.X, self.y)
        X[self.target_column] = y
        print(type(X))
        self.pd_df = X
        return self

    def _smote_tomek(self):
        print(f"Sampling using => {self.sampling_strategy}")
        assert not structural.get_null_df(
            self.pd_df
        ).sum(), "SmoteTomek requires zero null values"
        print(f"Smote sampling strategy => {self.sampling_strategy}")
        sampler = SMOTETomek(random_state=42, sampling_strategy=self.sampling_strategy)
        X, y = sampler.fit_resample(self.X, self.y)
        X[self.target_column] = y
        print(type(X))
        self.pd_df = X
        return self

    def sample(self):
        self._get_class_balance()
        self._split_xy()
        if not self.sampling_strategy:
            self._create_balanced_sampling_strategy()
        if self.technique == "oversample":
            self._oversample()
            self._get_class_balance()
        elif self.technique == "smote":
            self._smote()
            self._get_class_balance()
        elif self.technique == "smote_tomek":
            self._smote_tomek()
            self._get_class_balance()
        else:
            raise Exception("Sampling technique not recognized")
        return self
