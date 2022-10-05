from __future__ import annotations
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from PIL import Image
import os

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

def population_recovery(
        required_tags: list,
        excluded_tags: list,
        mock_catalogue:pd.DataFrame = pd.read_csv("./data/mock_catalogue.csv")
        ):
    """Selection function for populations based on semantic taxonomy.
    """
    tmp = mock_catalogue.copy()
    for tag in required_tags:
        tmp = tmp.loc[tmp[tag]>0]
    for tag in excluded_tags:
        tmp = tmp.loc[tmp[tag]==0]

    coordinates = list(tmp["coordinates"])
    return coordinates

def population_summary(
        required_tags,
        excluded_tags,
        derived_tags="default",
        df=pd.read_csv(
                "./data/lemmatization/derived_terms_0.80.csv",
                index_col=0),
        save=False,
        print_auxiliary=False,
        id_map="default"
    ):
    """Dirty cutout recovery function. Not fully acurate and does not use
    derived taxonomy, but rather intermediary derived tags.
    E.g. 'trace' instead of 'traces host galaxy'
    Cutout png data must be saved in data/cutouts/ for this function to
    operate correctly. Cutout file names must follow J123456+123456_overlay.png
    
    """
    if derived_tags=="default":
        derived_tags = [
            "trace", "double","asymmetric","extend","amorphous",
            "compact","counterpart","component","bent","bridge",
            "brighten","diffuse","peak","hourglass","jet","host",
            "faint","brightness","lobes","merger","small","middle",
            "plume","tail"
        ]
        derived_tags.sort()
    if id_map == "default":
        exp_data = pd.read_csv("./data/expert_classifications_anon.csv", index_col=0)
        id_map = return_subject_id_mapper(exp_data)
    tags = df
    tag = [required_tags, excluded_tags]
    inclusion_ids = set(tags["subject_ids"].unique())
    for idx, t in enumerate(tag[0]):
        ids_ = tags[tags["most_similar"]==t]
        ids = ids_["subject_ids"].unique()
        # Intersection
        inclusion_ids = inclusion_ids & set(list(ids))
        #inclusion_ids = [i for i in list(ids) if i in inclusion_ids]
        print(f"{t}: {len(inclusion_ids)}")


    exclusion_ids = set([])
    for t in tag[1]:
        exclusion_ids_ = tags[tags["most_similar"]==t]
        exclusion_ids_ = exclusion_ids_["subject_ids"].unique()
        # Union
        exclusion_ids = exclusion_ids | set(list(exclusion_ids_))

    print("inclusionary entries: ",len(inclusion_ids))
    print("excluded entries: ", len(exclusion_ids))
    ids = [i for i in inclusion_ids if i not in exclusion_ids]
    print("final entries: ", len(ids))
    files = [id_map[str(i)] for i in ids]
    if save:
        tag_folder = "_".join(tag[0])
        tag_folder += "-".join(tag[1])
        os.mkdir(f"./data/cutouts/subsets/{tag_folder}")
    for i in ids:
        file = id_map[str(i)]
        all_image_tags = tags[tags["subject_ids"]==i]["most_similar"].unique()
        image_tags = [tag for tag in all_image_tags if tag in derived_tags]
        if print_auxiliary:
            image_tags.sort()
            print(file)
            print(f"Tags assgigned to this target: ", image_tags)
        with Image.open(f"./data/cutouts/{file}") as im:
            fig, ax = plt.subplots(figsize=(20,5))
            ax.imshow(im)
            plt.axis("off")
            if save:
                plt.savefig(
                    f"./data/cutouts/subsets/{tag_folder}/{'_'.join(image_tags)}_{file}",
                    bbox_inches='tight', pad_inches=0)
            plt.show()
