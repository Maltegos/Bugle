import pandas as pd
from sentence_transformers import SentenceTransformer, util


def get_similar_bugs(description):
    df = pd.read_csv('duplicate_detection/data/bug_reports.csv', sep=';', on_bad_lines='skip', usecols=["Description"])
    df = df.replace('\n', '', regex=True)
    sentences = []

    for row in range(df.shape[0]):
        doc_1 = str(df.values[row])
        sentences.append(doc_1)

    model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L3-v2')

    # Store this locally for faster performance
    embeddings1 = model.encode(sentences, convert_to_tensor=True)
    embeddings2 = model.encode(description, convert_to_tensor=True)

    # Compute cosine-similarities
    cosine_scores = util.cos_sim(embeddings1, embeddings2)

    result = []

    # Output the pairs with their score
    for i in range(len(cosine_scores)):
        if cosine_scores[i] > 0.3:
            potential_duplicate = [float(cosine_scores[i][0]), sentences[i]]
            result.append(potential_duplicate)

    ordered = sorted(result, key=lambda x: x[0])

    return ordered[-3:]
