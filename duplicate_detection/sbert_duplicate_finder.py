import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from .embedder import load_embeddings


def get_bug_reports():
    # pandas DataFrame containing bug report data
    #TODO: fix npy file path
    data = np.load('duplicate_detection/data/bug_reports.npy', allow_pickle=True)
    bug_reports = [{'issue_id': row[0], 'summary': row[1], 'description': row[2], 'created': row[3]} for row in data]

    # to store bug reports as dictionaries of {'issue_id', 'description'}
    '''bug_reports = []

    for row in range(df.shape[0]):
        # put row values in dictionary {'issue_id': , 'description': } and append to bug_reports
        row_values = dict(issue_id=df.values[row][0], description=df.values[row][1], created=df.values[row][2])
        bug_reports.append(row_values)'''

    return bug_reports


def get_similar_bugs(description):

    bug_reports = get_bug_reports()

    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    # Encode embedding for input description
    input_embedding = model.encode(str(description))
    input_embedding = input_embedding.reshape(1, -1)
    # Load embeddings for bug reports
    bug_report_embeddings = load_embeddings()
    # Compute cosine-similarities
    cosine_scores = cosine_similarity(bug_report_embeddings, input_embedding)

    result = []

    # Output the pairs with their score
    for i in range(len(cosine_scores)):
        if cosine_scores[i][0] > 0.2:
            potential_duplicate = [float(cosine_scores[i][0]), bug_reports[i]]
            result.append(potential_duplicate)

    ordered = sorted(result, key=lambda x: x[0])
    New_list = ordered.copy()
    New_list.reverse()

    #print(type(ordered))
    #print("Result reversed is: ", ordered.reverse())
    return New_list[:6]
