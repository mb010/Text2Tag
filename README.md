# Text2Tag
Code base for the analysis presented in:
- [Bowles+2022 "A New Task: Deriving Semantic Class Targets for the Physical Sciences", accepted at NeurIPS 2022, Machine Learning and the Physical Sciences Workshop at the 36th Conference on Neural Information Processing Systems](https://arxiv.org/abs/2210.14760)
- Bowles+22 "Radio Galaxy Zoo EMU: Towards a Semantic Radio Galaxy Morphology Taxonomy", submitted to MNRAS.

The full data used in these works is available [on this zenodo](https://zenodo.org/record/7254123#.Y3d5EdLP2xE) including all of the composite cutouts of galaxies presented to citizen scientists.

# Quick Start
This repo was built on python 3.8 and a [virtual environment](https://docs.python.org/3/library/venv.html).
Use the [requirements.txt](./requirements.txt) file to install package
dependencies into your environment by running:
```
pip install -r requirements.txt
```

The SpaCy model requires seperate installation. To install the model used, run
```
python -m spacy download en_core_web_lg
```
and / or consult the [spaCy](https://spacy.io/usage#_title) documentation for more details.

# Re-Running Experiments
With your environment configured, you should be able to run all of our
processing. The steps are documented in the (submitted) paper and in the
jupyter notebooks.

The main processing steps are implemented in the order presented in [main.py](./Text2Tags/main.py).

The [utils.py](./Text2Tags/utils.py) module contains helper functions to
summarise and show original annotations which formed a given entry.

# Citing
For now, to cite this work (incl. code / data) please cite:
```
@ARTICLE{Bowles2022_SemanticTags,
       author = {{Bowles}, Micah and {Tang}, Hongming and {Vardoulaki}, Eleni and {Alexander}, Emma L. and {Luo}, Yan and {Rudnick}, Lawrence and {Walmsley}, Mike and {Porter}, Fiona and {Scaife}, Anna M.~M. and {Slijepcevic}, Inigo Val and {Segal}, Gary},
        title = "{A New Task: Deriving Semantic Class Targets for the Physical Sciences}",
      journal = {NeurIPS 2022: Machine Learning and the Physical Sciences Workshop},
     keywords = {Astrophysics - Instrumentation and Methods for Astrophysics, Computer Science - Computation and Language},
         year = 2022,
        month = oct,
          eid = {arXiv:2210.14760},
        pages = {arXiv:2210.14760},
archivePrefix = {arXiv},
       eprint = {2210.14760},
 primaryClass = {astro-ph.IM},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2022arXiv221014760B},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
