from __future__ import annotations

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import spacy
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split


def correlations():
    """Visualise and / or find a way to deal with tags which are strongly correlated."""
    raise NotImplementedError

def plot_importances(clf, df: pd.DataFrame, save: None | str=None) -> None:
    """Make a pretty plot of the importances and the occurances of the tags."""
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


def main():
    accuracies = {"oob":[]}
    # Load in data
    lemmatization = "lemmatization"
    thresholds = np.linspace(start=0.5, stop=1., num=11, endpoint=True)
    expert_thresholds = np.linspace(start=1, stop=5, num=5, endpoint=True)
    input_data_paths = ["../data/derived_terms_{}.csv".format(threshold) for threshold in thresholds]
    target_data_paths = ["../data/expert_classifications.csv"]

    df_exp = pd.from_csv(f"../data/expert_classifications.csv")
    # Construct target multi dimensional data DataFrame
    iterables = [[list(thresholds)], [list(target_data_paths)], list(df_exp.annotations.unique()).append("all_classes")]]
    columns = pd.MultiIndex.from_product(iterables, names=["similarity_threshold", "expert_threshold", "classification_task")
    df = pd.DataFrame(columns=columns)
    print(f"df: {df}")
    for threshold in thresholds:
        df_eng = pd.from_csv(f"../data/{lemmatization}/derived_terms_{threshold}.csv")
        #### SEE NOTEBOOK FOR TARGET DERIVATION
        ### Need to derive input vectors.
        # Load data
        for target_class in class_specification:
            # Extract one hot encodings
            # Train & Evaluate
            clf = RandomForestClassifier(
                n_estimators=1000,
                max_depth=None,
                random_state=42,
                min_samples_leaf=2,
                oob_score=True
            )
            clf.fit(X_train, y_train)
            accuracies["oob"].append(clf.oob_score_)

            break

    # Calculate Importances
    # Permutation Importance
    # Miss One out Importance
    # Construct data frame of Importances

if __name__ == "__main__":
    main()
