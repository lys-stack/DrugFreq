

# DrugFreq

## Environment Reproduce

- In order to get DrugFreq, you need to clone this repo:

  ```
  https://github.com/lys-stack/DrugFreq.git
  ```

## Files

* **Data** -- Data used in the training and testing process, including drugs, side effects, and drug-side effect frequencies.
* **model2.py** -- Model file for predicting drug-side effect associations.
* **model3.py** -- Model file for predicting drug-side effect frequencies.
* **combine.py** -- Code file for feature fusion using a similarity matrix.
* **myutils** -- Other utility files, such as the definition of loss functions, etc.

## Data sets

* **664_drug_disease_jaccard_matrix.csv** -- We obtain the association information between drugs and diseases from the CTD database, where each element s(i,j) in the association matrix represents the relationship between drug i and disease j. The similarity matrix is constructed using the Jaccard similarity measure.

* **664_drug_drug_scores.csv** -- We obtain the compound IDs corresponding to each drug and extract the compound-compound association scores from the STITCH database.

* **664_drug_fingerprint_jaccard_similarity_matrix_new.csv** -- We collected the SMILES sequences for each drug from the STITCH database, and then input these SMILES sequences into RDKit to generate a 2048-dimensional vector representation for each compound. The Jaccard coefficient is used to measure the structural similarity between drugs.

* **kvplm_Side_Effect_Similarity_Matrix.csv** -- We constructed a similarity matrix based on biomedical text information of side effects. The text data related to side effects were collected from Wikipedia and the PubChem database. To prevent data leakage, we excluded all descriptions involving relationships between drugs and side effects.

* **semantic.csv** -- We adopt an existing measurement method to represent the semantic descriptors of each side effect by constructing a directed acyclic graph.

* **word_new.csv** -- Each side effect term is mapped into a 300-dimensional pre-trained word embedding space. To quantify the semantic relatedness between different side effects, cosine similarity is introduced as the evaluation metric.

* **Drug-Side_Effect_Frequency664.csv** -- This file is the frequency matrix of drug side effects.

  

## Run for warm-start

- Warm-start file for classification status

  ```
  python new1_random2.py
  ```

- Warm-start file for classification status

  ```
  python new1_random3.py
  ```

## Run for cold-start

- Cold-start file for classification status

  ```
  python cold_start2.py
  ```

- Cold-start file for classification status

  ```
  python cold_start3.py
  ```