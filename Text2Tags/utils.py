from __future__ import annotations
import pandas as pd
import numpy as np
import json

def find_original_annotations(
        annotation: str,
        lemmatization: str,
        similarity_threshold: float,
        expert_threshold: float,
        folder: str
    ) -> pd.DataFrame:
    file = "../{folder}/{lemmatization}/derived_terms_{similarity_threshold:.2f}.csv"
    df = pd.read_csv(file.format(
        folder=folder,
        lemmatization=lemmatization,
        similarity_threshold=similarity_threshold
        ))
    cleaned = pd.read_csv("../data/english_annotations_cleaned.csv")
    #cleaned = pd.read_csv("../data/english_annotations.csv")
    matches = df[df["most_similar"]==annotation]
    #matching = cleaned[cleaned["classification_id"].isin(local_matching)]
    return matches

def original_annotation_summary(
        annotation: str,
        lemmatization: str = 'lemmatization',
        similarity_threshold: float = 0.8,
        expert_threshold: float = 0.6,
        folder: str = 'data'
    ) -> None:
    matches = find_original_annotations(
        annotation = annotation,
        lemmatization = lemmatization,
        similarity_threshold = similarity_threshold,
        expert_threshold = expert_threshold,
        folder = folder)
    matches = matches.sort_values(by="original_annotations")
    print(matches.head())
    text = "\n"
    text += "Annotation:\n"
    text += f"  {annotation}\n"
    text += "Non-identical elements:\n"
    text += f"  {matches.original_annotations.values[matches.original_annotations.apply(lambda x : x.strip(' ')).values!=annotation]}\n"
    text += f"Total Count: {matches.shape[0]}\n"
    text += f"Perfect Matches: {matches.annotations.values[matches.annotations.values==annotation].shape[0]}\n"
    text += f"Imperfect Matches: {matches.annotations.values[matches.annotations.values!=annotation].shape[0]}"
    print(text)

def return_subject_id_mapper(raw_data):
    """Takes dataframe of zooniverse data and returns a
    dict which maps subject ids to file names."""
    mapping = {}
    subject_data = [json.loads(q) for q in raw_data['subject_data']]
    for d in subject_data:
        for k, v in d.items():
            mapping[k] = v["Filename"]
    return mapping
