"""
'Radio Galaxy Zoo: EMU - Towards a Semantically Meaningful Radio
Morphology Ontology'

Run the pipeline from this script to create the
data prodcuts used in Bowles+22.
"""
import classification_encoding
import embedding
import preprocess
import importances

def main():
    preprocess.main()
    classification_encoding.main()
    embedding.main(lemmatize=True)
    importances.main()

if __name__=="__main__":
    main()
