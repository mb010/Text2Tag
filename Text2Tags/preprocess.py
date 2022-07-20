from __future__ import annotations
import pandas as pd
import spacy
import json
import contractions
import unidecode


def raw_to_public(data_path: str, outpath: str, force: bool=False):
    """Removes sensitive information and
    transforms the annotations into strings.
    """
    if "raw_" not in data_path and not force:
        return
    df = pd.read_csv(data_path)
    df = df[['classification_id', 'user_id', 'workflow_id', 'annotations', 'subject_ids']]

    df['annotations'] = [json.loads(annotation)[0]['value'] for annotation in df['annotations']]
    if "expert" in outpath:
        df.replace(to_replace=r'\*', value='', regex=True)
        df['annotations'] = df['annotations'].str.join(",")
    df.to_csv(outpath)
    return

def case_correction(df: pd.DataFrame) -> pd.DataFrame:
    """Make all annotations lower case."""
    df.annotations = df.annotations.str.lower()
    return df

def remove_accents(df: pd.DataFrame) -> pd.DataFrame:
    df.annotations = df.annotations.apply(lambda x: unidecode.unidecode(str(x)))
    return df

def edge_case_slashes(df: pd.DataFrame) -> pd.DataFrame:
    """Replace slashes with commas as appropriate for the uncleaned data."""
    df.annotations = df.annotations.str.replace('/', ',')
    return df

def edge_case_periods(df: pd.DataFrame) -> pd.DataFrame:
    """Handle periods as a replacement for commas in the annotations."""
    df.annotations = df.annotations.str.strip('.')
    df.annotations = df.annotations.str.replace("3.4", "", regex=False)
    df.annotations = df.annotations.str.replace("host on. the image edge", "host on the image edge", regex=False)
    df.annotations = df.annotations.str.replace(".", ",", regex=False)
    return df

def edge_case_double_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    """Process double white spaces as accidental occurances."""
    df.annotations = df.annotations.str.replace("  ", " ", regex=False)
    return df

def edge_case_ambersands(df:pd.DataFrame) -> pd.DataFrame:
    """Process ambersands within annotations."""
    #print(df[df.annotations.str.contains("&", na=False)])
    df.loc[df.annotations.str.contains("\S&", regex=True, na=False)].annotations.replace("& ", " and ", regex=False, inplace=True)
    df.loc[df.annotations.str.contains("&\S", regex=True, na=False)].annotations.replace(" &", " and ", regex=False, inplace=True)
    df.annotations = df.annotations.str.replace("&", "and", regex=False)
    #print(df[df.annotations.str.contains("&", na=False)])
    return df

def edge_case_remove_new_lines(df: pd.DataFrame) -> pd.DataFrame:
    """Remove any occurance of newlines '\n'."""
    df.annotations = df.annotations.str.replace('\n', ',', regex=False)
    return df

def explode_annotations(df: pd.DataFrame) -> pd.DataFrame:
    """Explode to a row for each annotation, rather than per source."""
    df.annotations = df.annotations.str.split(pat=',')
    df = df.explode("annotations", ignore_index=True)
    return df

def manual_corrections(tag: str) -> str:
    manual_corrections_ = {
        # Correcting concatenated words.
        # Corrections determined through through visual inspection of data.
        "outofframe": "out of frame",
        "sshape": "s-shape",
        "head tail": "head tail",
        "overedge": "over edge",
        "galaxyoverlap": "galaxy overlap",
        "doublehourglass": "double hourglass",
        "overedge": "over edge",
        "multiplesource": "multiple sources",
        "multiplesources": "multiple sources",
        "emissionoverlap": "emission overlap",
        "elongatedcompact": "elongated compact",
        "mergedemission": "merged emission"
    }
    if tag in manual_corrections_.keys():
        tag = manual_corrections_[tag]
    return tag

def edge_case_manual_corrections(df: pd.DataFrame) -> pd.DataFrame:
    """Manually correct annotations written as a single word to a annotation."""
    df.annotations = df.annotations.apply(manual_corrections)
    return df

def edge_case_hyphen(df: pd.DataFrame) -> pd.DataFrame:
    """Replace hyphens with spaces."""
    df.annotations = df.annotations.str.replace("-", " ")
    return df

def edge_case_host_references(df: pd.DataFrame, references: list[str] = ["optical", "dss", "wise", "host", "disk", "associated"]) -> pd.DataFrame:
    """Remove annotations which contain references to host morphology."""
    for term in references:
        df = df[~df.annotations.str.contains(term, na=False)]
    return df

def expand_contractions(df: pd.DataFrame) -> pd.DataFrame:
    """Expand contractions within the annotations."""
    df.annotations = df.annotations.apply(lambda x: contractions.fix(str(x)))
    return df

def lemmatization(df: pd.DataFrame) -> pd.DataFrame:
    """Lemmatize strings."""
    raise NotImplementedError

def remove_stop_words(words: list[str], stopwords: set, join: bool=True):
    """Removes all stop words from the list of elements
    and returns a joined string by default."""
    for word in reversed(words):
        if word in stopwords:
            words.remove(word)
    return " ".join(words) if join else words

def apply_stopwords(
        df: pd.DataFrame,
        custom_stopwords: list[str]=["emu", "galaxy", "galactic", "emission",
            "like", "side", "source", "north", "south", "east", "west"],
        nlp=spacy.load("en_core_web_lg")) -> pd.DataFrame:
    """Remove stopwords."""
    stopwords = nlp.Defaults.stop_words
    stopwords.update(custom_stopwords)
    df.annotations = df.annotations.str.split().apply(remove_stop_words, stopwords=stopwords)
    return df


def main():
    nlp = spacy.load("en_core_web_lg")
    try:
        raw_to_public("../data/raw_english.csv", "../data/english_annotations.csv")
        raw_to_public("../data/raw_expert_classifications.csv", "../data/expert_classifications.csv")
    except:
        pass
    df = pd.read_csv("../data/english_annotations.csv", index_col=0)
    # First some edge case handling
    df = case_correction(df)
    df = remove_accents(df)
    df = edge_case_slashes(df)
    df = edge_case_periods(df)
    df = edge_case_double_whitespace(df)
    df = edge_case_ambersands(df)
    df = edge_case_remove_new_lines(df)
    # Explode into individual annotations
    df = explode_annotations(df)
    # Using exploded data, we can now drop host referencing annotations
    df = edge_case_manual_corrections(df)
    df = edge_case_hyphen(df)
    df = edge_case_host_references(df)
    # Using host removed data, we can alter the words if appropriate.
    df = expand_contractions(df)
    #df = lemmatization(df) # Maybe I dont need lemmatization if it is no longer the default in spaCy v3.
    df = apply_stopwords(df)
    print(f"Final shape: {df.shape}")
    df.to_csv("../data/english_annotations_cleaned.csv")

if __name__=="__main__":
    main()
