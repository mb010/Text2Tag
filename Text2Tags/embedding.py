from __future__ import annotations
from dataclasses import dataclass, replace
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spacy
from spacy.tokens import Doc


@dataclass
class Annotations():
    df: pd.DataFrame
    similarity_measure: str = "cosine"
    save: bool = False
    lemmatize: bool = False
    nlp = spacy.load("en_core_web_lg")

    def __post_init__(self):
        self.df = self.embed()
        self.df = self.similarities()

    def embed(self) -> pd.DataFrame:
        """Embed cleaned annotations to a documents."""
        df = self.df.copy()
        df['doc'] = [self.nlp(ann) for ann in df.annotations.values]
        return df

    def similarities(self) -> pd.DataFrame:
        """Use similarity measure to calculate similarities."""
        df = self.df.copy()
        if self.similarity_measure == "cosine":
            df['similarities'] = df['doc'].apply(
                lambda x: [doc.similarity(x) for doc in df['doc']]
            )
        else:
            raise NotImplementedError
        return df

    def similarities_histogram(self, save: str | None = None) -> None:
        """Plot a paper ready histogram of the similarity measures
        that are available."""
        values = self.df['similarities'].values
        values = np.vstack(values)
        values -= np.identity(values.shape[0])
        values = values.flatten()

        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.hist(values, bins=50)
        ax.set_xlabel(f"Similarity Measure (cosine similarity)")
        ax.set_ylabel(f"Count")
        if save is not None:
            fig.savefig(save)
        else:
            plt.show()

    def lemmatize_doc(self, annotation) -> str:
        """Return a lemmatized version of the input documentation."""
        tmp_string = " ".join([token.lemma_ for token in annotation])
        return self.nlp(tmp_string)

    def mean_vector_(self, row: pd.DataFrame, threshold: float) -> np.ndarray:
        """Take the mean vector across the selected similarity threshold."""
        similarities = np.asarray(row["similarities"])
        if self.lemmatize:
            doc_series = self.df[similarities>=threshold]["doc"].to_list()
            mean_vector = self.lemmatize_doc(Doc.from_docs(doc_series)).vector
        else:
            doc_series = self.df[similarities>=threshold]["doc"].to_list()
            mean_vector = Doc.from_docs(doc_series).vector
        return mean_vector

    def most_similar_(self, row: pd.DataFrame) -> str:
        """Retrieve embedded string which is most similar to mean vectors."""
        most_similar, _, distances = self.nlp.vocab.vectors.most_similar(row.mean_vector[np.newaxis], n=1)
        return self.nlp.vocab.strings[most_similar[0][0]].lower()

    def extract_most_similar(self, threshold: float) -> pd.DataFrame:
        """Appends the relevant data derivatives to the data frame."""
        df = self.df.copy()
        df['mean_vector']  = df.apply(lambda row: self.mean_vector_(row, threshold), axis=1)
        df['most_similar'] = df.apply(lambda row: self.most_similar_(row), axis=1)
        return df

    def vectorize(self) -> pd.DataFrame:
        df = pd.get_dummies(self.df["most_similar"], prefix='ms')
        return df

def main():
    df = pd.read_csv("../data/english_annotations_cleaned.csv", index_col=0).dropna()
    thresholds = np.linspace(start=0.5, stop=1., num=11, endpoint=True)
    lemmatize=False
    enc = Annotations(df, lemmatize=lemmatize)
    #enc.similarities_histogram(save="./similarities_histogram.png")
    for threshold in thresholds:
        print(f">>> Threshold {threshold}")
        df = enc.extract_most_similar(threshold)
        print(f">>> Unique Classes:")
        print(f"{df['most_similar'].unique().shape}, {df['most_similar'].unique()}")
        codes, unique = df["most_similar"].factorize()
        df["codes"] = codes
        tmp = df.drop(columns=["doc", "mean_vector", "similarities"])
        lem = "lemmatization" if lemmatize else "no_lemmatization"
        tmp.to_csv(f"../data/{lem}/derived_terms_{threshold:.2f}.csv")

    print(">>> Finished")

if __name__=="__main__":
    main()
