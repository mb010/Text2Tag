"""
'Radio Galaxy Zoo: EMU - Towards a Semantically Meaningful Radio
Morphology Ontology'

Run the pipeline from this script to create the
data prodcuts used in Bowles+22.
"""
import classification_encoding
import embedding
import preprocess

def main():
    preprocess.main()
    classification_encoding.main()
    embedding.main()

if __name__=="__main__":
    main()
