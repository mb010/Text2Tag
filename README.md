# Text2Tag
Code base for the analysis published in Bowles+22: "Radio Galaxy Zoo: EMU - Towards a Semantically Meaningful Radio Morphology Ontology"

# Quick Start
This repo was built on python 3.8 and a [virtual environment](https://docs.python.org/3/library/venv.html)
Use the [requirements.txt](./requirements.txt) file to install package
dependencies into your environment by running:
```
pip install -r requirements.txt
```

The spaCy model will likely not install off of the `requirements.txt` install.
To install the model used, run:
```
python -m spacy download en_core_web_lg
```
and / or consult the [spaCy](https://spacy.io/usage#_title) documentation for more details.

# Re-Running Experiments
With your environment configured, you should be able to run all of our
processing. The steps are documented in the [paper]() and in the jupyter notebooks.
We highlight which files or notebooks were used to create each figure:
