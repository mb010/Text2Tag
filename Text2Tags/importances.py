from __future__ import annotations
from dataclasses import dataclass

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import shap

import spacy
import sklearn
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import f1_score

from classification_encoding import EncodedData

@dataclass
class Importances():
    df_X: EncodedData
    df_Y: EncodedData
    model_name: str = "RandomForestClassifier"
    scores = dict()
    n_jobs: int = 10
    seed = 42
    cv: int = 10

    def __post_init__(self):
        self.X, self.y = self.format_data()
        self.clf = self.load_model()
        self.train()
        self.scores = self.evaluate()

    def train(self) -> None:
        """Train the model on the full data set"""
        self.clf.fit(self.X, self.y)

    def evaluate(self) -> None:
        """Validation using cross validation."""
        scores = dict()
        y_pred = cross_val_predict(self.clf, self.X, self.y, cv=self.cv)
        scores["micro_f1"] = f1_score(self.y, y_pred, average="micro")
        scores["weighted_f1"] = f1_score(self.y, y_pred, average="weighted")
        return scores

    def load_model(self):
        """Load the correct selection of model based on the model_name provided
        by the user. The default is a RandomForestClassifier. Currently
        implemented: ["RandomForestClassifier"]"""
        # See: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
        if self.model_name == "RandomForestClassifier":
            clf = OneVsRestClassifier(RandomForestClassifier(
                n_estimators=500,
                criterion="gini",
                max_depth=None,
                random_state=self.seed
            ), n_jobs=self.n_jobs)
        elif self.model_name == "XGBoost":
            raise NotImplementedError("XGBoost not yet supported.")
        else:
            raise NotImplementedError(f"{self.model_name} not supported.")
        return clf

    def feature_summary(self) -> pd.DataFrame:
        """Returns a data frame which contains summary of the input featuers of
        the model."""
        target_counts = self.count(self.df_Y)[:,np.newaxis]
        df = {
            "annotations": self.df_X.encoding.keys(), # Annotations
            "count": self.count(self.df_X),
        }

        importance_values = np.nan_to_num(self.permutation_importance())
        permutation_importance = {f"importance_{key}": importance_values[idx] for key, idx in self.df_Y.encoding.items()}
        permutation_importance["importance_mean"]     = np.sum(importance_values, axis=0)
        permutation_importance["importance_weighted"] = np.sum(importance_values*target_counts, axis=0)

        shapley_values = np.abs(np.nan_to_num(self.shapley_values())).sum(axis=(1,2))
        shap = {f"shap_{key}": shapley_values[idx] for key, idx in self.df_Y.encoding.items()}
        shap["shap_mean"]     = np.sum(shapley_values, axis=0)
        shap["shap_weighted"] = np.sum(shapley_values*target_counts, axis=0)

        df = {**df, **permutation_importance, **shap}
        return pd.DataFrame(df)

    def format_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Produces indicies for stratified folds of training and test data."""
        X = self.df_X.create_target_vectors()[self.df_X.mapping]
        y = self.df_Y.create_target_vectors()[self.df_Y.mapping]
        X = np.concatenate(X.values).reshape(X.shape[0], -1)
        y = np.concatenate(y.values).reshape(y.shape[0], -1)
        return X, y

    def permutation_importance(self) -> np.ndarray:
        """Returns the feature importances of the random forrest model using
        permutation importance."""
        value = self.apply_per_estimator(lambda x: x.feature_importances_)
        value = self.correct_nan_shape(value)
        return value

    def shapley_values(self) -> np.ndarray:
        """Returns the shapley values for a given key and the current model."""
        shap_values = self.apply_per_estimator(
            lambda x: np.stack(shap.TreeExplainer(x).shap_values(
                self.X, check_additivity=True))
        )
        shap_values = self.correct_nan_shape(shap_values)
        return shap_values

    def count(self, encoded: EncodedData) -> np.ndarray:
        """Returns the count of the classes in a given encoding."""
        return encoded.df.onehot.sum()

    def apply_per_estimator(self, func: callable):
        """Applies a function to each individual estimator from within the
        OneVsRestClassifier."""
        out = []
        for i in range(self.clf.n_classes_):
            try:
                out.append(func(self.clf.estimators_[i]))
            except:
                #out.append([np.nan for i in range(self.df_X.class_count)])
                out.append(np.nan)
        return out

    def correct_nan_shape(self, array_list) -> np.ndarray:
        """Makes all nan entries in a list the same shape as the first array in
        the list, and then returns a stacked np.ndarray with a new axis at 0."""
        # Get the shape
        shape = None
        for idx, i in enumerate(array_list):
            if i is not np.nan:
                shape = array_list[idx].shape
                break
        # Return original data if shape was not determined
        if shape is None:
            return array_list
        # Adapt np.nan entries to be correct shape
        arr = np.empty(shape)
        arr.fill(np.nan)
        for idx, i in enumerate(array_list):
            if i is np.nan:
                array_list[idx] = arr
        return np.stack(array_list, axis=0)

    def target_summary(self) -> pd.DataFrame:
        """Returns a data frame which contains a summary of the target data."""
        df = {
            "annotation": [key for key, value in self.df_Y.encoding.items()],
            "count": self.count(self.df_Y),
            "index": [values for key, value in self.df_Y.encoding.items()]
        }
        return pd.DataFrame(df)


def correlations():
    """Visualise and / or find a way to deal with tags which are strongly
    correlated."""
    raise NotImplementedError("Correclation calculations not yet implemented.")

def plot_importances(clf, df: pd.DataFramesummary, save: None | str=None) -> None:
    """Make a pretty plot of the importances and the occurances of the tags."""
    raise NotImplementedError("Plotting not yet implemented.")
    top_ranked = pd.DataFrame({
        "most_similar_importances": clf.feature_importances_,
        "most_similar":df.most_similar.unique(),
        "counts": df.most_similar.value_counts().to_numpy()
    })
    top = top_ranked.sort_values('most_similar_importances', ascending=False).reset_index(drop=True)
    uniform_sampling_threshold = 1/top.most_similar_importances.shape[0]

    fig, ax = plt.subplots(figsize=(20,5))
    ax.scatter(
        np.arange(top.most_similar_importances.shape[0]),
        top.most_similar_importances,
        s=0.1*top.counts.to_numpy(),
        c=np.linspace(0,1,top.most_similar.shape[0])
    )
    ax.set_xticks(np.arange(top.most_similar_importances.shape[0]), top.most_similar, rotation=90)
    ax.grid()
    ax.axhline(
        uniform_sampling_threshold,
        linestyle='--', c='red',
        label=f'Uniform importance: {top.most_similar_importances[top.most_similar_importances>uniform_sampling_threshold].shape[0]} terms'
    )
    ax.axvline(10.5, linestyle='dashdot', c='black', label=f'Top 10')
    ax.legend()
    ax.set_title(f"Importances of Derived Tags for Science Classifications (trained on composite classes)")
    ax.set_xlabel(f"Derived Tags")
    ax.set_ylabel(f"Feature Importance")
    plt.show()

    top.head(15)

def single_main_call(
        lemmatization: bool, similarity_threshold: float, expert_threshold: float
        ) -> Importances:
    """Returns an Importance object contianing all of the evaulations of the
    model for the provided conditions. Models are retrained with a seed."""
    lem = "lemmatization" if lemmatization else "no_lemmatization"
    file_path = f"../data/{lem}/derived_terms_{similarity_threshold:.2f}.csv"
    df_X = pd.read_csv(file_path, index_col=0)
    df_X = df_X.rename(columns={
        "annotations": "raw_annotations",
        "most_similar": "annotations"})
    df_X = EncodedData(df_X)
    df_Y = EncodedData(
        pd.read_csv("../data/expert_classifications.csv", index_col=0),
        agreement=expert_threshold/5
    )
    im = Importances(df_X, df_Y)
    return im

def main():
    # Parameters to iterate over for ablation studies
    lemmatizations        = [True, False]
    similarity_thresholds = np.linspace(start=0.5, stop=1., num=11, endpoint=True)
    expert_thresholds     = np.linspace(start=1, stop=4, num=4, endpoint=True)

    overview_dict_entries = [
        "lemmatization", "similarity_threshold", "expert_threshold",
        "tag_count", "weighted_f1", "macro_f1"
    ]
    overview_dict = {k: [] for k in overview_dict_entries}

    # Loop over each iterable parameter and run & evaluate experiment
    for lemmatization in lemmatizations:
        lem = "lemmatization" if lemmatization else "no_lemmatization"
        for similarity_threshold in similarity_thresholds:
            # Load in training data
            file_path = f"../data/{lem}/derived_terms_{similarity_threshold:.2f}.csv"
            df_X = pd.read_csv(
                file_path,
                index_col=0)
            df_X = df_X.rename(columns={
                "annotations": "raw_annotations",
                "most_similar": "annotations"
            })
            df_X = EncodedData(df_X)
            for expert_threshold in expert_thresholds:
                # Load in target data & extract encoding table
                df_Y = pd.read_csv("../data/expert_classifications.csv", index_col=0)
                df_Y = EncodedData(df_Y, agreement=expert_threshold/5)
                importances = Importances(df_X, df_Y)
                #return importances
                # Add to output data frame
                df = importances.feature_summary()
                df.to_csv(f"../data/{lem}/importances_{similarity_threshold:.2f}_{expert_threshold}.csv")

                # Save evaluation / iteration to DataFrame
                overview_dict["tag_count"].append(df_X.class_count)
                overview_dict["lemmatization"].append(lem)
                overview_dict["similarity_threshold"].append(similarity_threshold)
                overview_dict["expert_threshold"].append(expert_threshold)
                overview_dict["weighted_f1"].append(importances.scores["weighted_f1"])
                overview_dict["macro_f1"].append(importances.scores["micro_f1"])

    # Save oob metrics
    df = pd.DataFrame(overview_dict)
    df.to_csv("../data/results_overviews.csv")


if __name__ == "__main__":
    main()
