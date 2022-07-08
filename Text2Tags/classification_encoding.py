from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class Classifications():
    """Class for managing the classifications."""
    df: pd.DataFrame
    mapping: str = "onehot"
    multiple_classifications_handling: str | callable = "last"
    weighting: str | callable = "uniform"
    agreement: float = 0.

    def __post_init__(self):
        self.df = self.df.dropna()
        self.encoding = self.encode()
        self.decoding = {v: k for k, v in self.encoding.items()}
        if self.multiple_classifications_handling is not None:
            self.duplicates()

    def duplicates(self) -> None:
        """Finds and handles situations where a single user classified the same
        target multiple times."""
        agg = {
            "annotations": self.multiple_classifications_handling,
            "classification_id": self.multiple_classifications_handling,
            self.mapping: self.multiple_classifications_handling
        }
        if self.multiple_classifications_handling not in ["first", "last"]:
            agg["annotations"] = lambda x: ",".join(x)

        self.df = self.df.groupby(["user_id", "subject_ids"]).agg(agg).reset_index()
        return

    def create_target_vectors(self) -> pd.DataFrame:
        """Returns the classifications for target sources, using the weighting
        and agreement parameters."""
        df = self.df.copy()
        df = df.explode(self.mapping, ignore_index=True)
        df = df.groupby(["classification_id"]).agg({
            "subject_ids": "first",
            self.mapping: lambda x: np.sum(x)
        })
        df["count"] = 1
        df = df.groupby("subject_ids").apply(lambda x: x.sum()).reset_index(drop=True)
        if self.weighting == "uniform":
            df[self.mapping] = df[self.mapping]/df['count']
        elif callable(self.weighting):
            df[self.mapping] = df[self.mapping].apply(self.weighting)
        else:
            raise NotImplementedError
        df[self.mapping] = df[self.mapping].apply(lambda x: np.where(x<=self.agreement, 0, 1))
        return df[["subject_ids", self.mapping, "count"]]

    def num_to_str(self, number: int) -> str:
        string = self.decoding(number)
        return string

    def str_to_num(self, string: str) -> int:
        number = self.encoding(string)
        return number

    def encode(self) -> dict:
        """Generates integer encoding for the classifications."""        
        classifications = self.df.annotations.dropna().str.split(',').explode().values
        classifications = list(set(classifications))
        classifications.sort()
        encoding = {}
        for idx, clf in enumerate(classifications):
            encoding[clf] = idx
        if self.mapping == "onehot":
            vectors = np.identity(len(classifications), dtype=np.int32)
        else:
            raise NotImplementedError
        self.df[self.mapping] = self.df.annotations.str.split(',').apply(lambda x: [vectors[encoding[i]] for i in x])
        return encoding

def main():
    df = pd.read_csv("../data/expert_classifications.csv", index_col=0)
    enc = Classifications(df)

    print(f">>> ENCODED DATA:")
    print(f"{enc.df}")
    print(f">>> TRAINING VECTORS:")
    print(f"{enc.create_target_vectors()}")

if __name__=="__main__":
    main()
