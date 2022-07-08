from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spacy
from spacy.tokens import Doc


@dataclass
class Annotations():
    df: pd.DataFrame
    threshold: float = 0.9
    similarity_measure: str = "cosine"
    save: bool = False
    lemmatize: bool = False
    nlp = spacy.load("en_core_web_lg")
    def __post_init__(self):
        self.df = self.embed()
        self.df = self.similarities()
        self.df = self.extract_most_similar()

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

    def mean_vector_(self, row: pd.DataFrame) -> np.ndarray:
        """Take the mean vector across the selected similarity threshold."""
        similarities = np.asarray(row["similarities"])
        if self.lemmatize:
            #mean_vector = self.df[similarities>=self.threshold]["doc"].apply(lambda x: self.lemmatize_doc(x).vector).mean()
            doc_series = self.df[similarities>=self.threshold]["doc"].to_list()
            mean_vector = self.lemmatize_doc(Doc.from_docs(doc_series)).vector
        else:
            #mean_vector = self.df[similarities>=self.threshold]["doc"].apply(lambda x: x.vector).mean()
            doc_series = self.df[similarities>=self.threshold]["doc"].to_list()
            mean_vector = Doc.from_docs(doc_series).vector
        return mean_vector

    def most_similar_(self, row: pd.DataFrame) -> str:
        """Retrieve embedded string which is most similar to mean vectors."""
        most_similar, _, distances = self.nlp.vocab.vectors.most_similar(row.mean_vector[np.newaxis], n=1)
        return self.nlp.vocab.strings[most_similar[0][0]].lower()

    def extract_most_similar(self) -> pd.DataFrame:
        """Appends the relevant data derivatives to the data frame."""
        df = self.df.copy()
        df['mean_vector']  = df.apply(lambda row: self.mean_vector_(row), axis=1)
        df['most_similar'] = df.apply(lambda row: self.most_similar_(row), axis=1)
        return df

    def vectorize(self) -> pd.DataFrame:
        df = pd.get_dummies(self.df["most_similar"], prefix='ms')
        return df

def main():
    df = pd.read_csv("../data/english_annotations_cleaned.csv", index_col=0).dropna()
    thresholds = np.linspace(start=0.5, stop=1., num=11, endpoint=True)
    print(thresholds)
    lemmatize=True
    enc = Annotations(df, threshold=0.8, lemmatize=lemmatize)
    #enc.similarities_histogram(save="./similarities_histogram.png")
    for threshold in thresholds:
        print(f">>> Threshold {threshold}")
        enc.threshold = threshold
        enc.extract_most_similar()
        print(f">>> Unique Classes:")
        print(f"{enc.df['most_similar'].unique().shape}, {enc.df['most_similar'].unique()}")
        codes, unique = enc.df["most_similar"].factorize() ### < --- Can I use this to encode annotations / classifications?
        enc.df["codes"] = codes
        tmp = enc.df.drop(columns=["doc", "mean_vector", "similarities"])
        if lemmatize:
            tmp.to_csv(f"../data/lemmatization/derived_terms_{threshold}.csv")
        else:
            tmp.to_csv(f"../data/no_lemmatization/derived_terms_{threshold}.csv")

    print(">>> Finished")

if __name__=="__main__":
    main()
