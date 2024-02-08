# This class evaluates the performance of the model based on the percentage of correctly identified duplicates from the
# Mozilla Firefox repository
import time
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf


def preprocess_csv():
    # pandas DataFrame containing bug report data
    # TODO: make filename dynamic, i.e. pass as arg upon call
    df = pd.read_csv('../data/mozilla_firefox.csv', sep=',', on_bad_lines='skip',
                     usecols=["Issue_id", "Duplicated_issue", "Description"])
    df = df.replace('\n', '', regex=True)

    np_array = df.to_numpy()
    # TODO: make filename dynamic, i.e. pass as arg upon call
    np.save("../mozilla_firefox.npy", np_array)


def embed_bug_reports():
    # pandas DataFrame containing bug report data
    df = pd.read_csv('../data/mozilla_firefox.csv', sep=',', on_bad_lines='skip', usecols=["Issue_id", "Description"])
    df = df.replace('\n', '', regex=True)

    # to store bug report descriptions as strings
    descriptions = []

    for row in range(df.shape[0]):
        # stringify description values and append to sentences
        sentences = str(df.values[row][1])
        descriptions.append(sentences)

    model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L3-v2')

    bug_report_embeddings = model.encode(descriptions)

    # Store embeddings on disc in NPY format
    np.save('../test_embeddings.npy', bug_report_embeddings)


# this function should be called for loading the embeddings file when finding duplicates
def load_embeddings():
    # Load embeddings from disc in NPY format
    stored_embeddings = tf.convert_to_tensor(np.load('../test_embeddings.npy'))
    return stored_embeddings


def get_bug_reports():
    # read data from npy file
    data = np.load('../mozilla_firefox.npy', allow_pickle=True)

    # to store bug reports as dictionaries of {'issue_id', 'description'}
    bug_reports = []

    for row in data:
        # put row values in dictionary {'issue_id': , 'description': } and append to bug_reports
        row_values = dict(issue_id=row[0], description=row[2])
        bug_reports.append(row_values)

    return bug_reports


def get_similar_bugs(description):
    bug_reports = get_bug_reports()
    model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L3-v2')
    # Encode embedding for input description
    input_embedding = model.encode(description)
    # Load embeddings for bug reports
    bug_report_embeddings = load_embeddings()
    # Compute cosine-similarities
    cosine_scores = util.cos_sim(bug_report_embeddings.numpy(), input_embedding)
    result = []

    # Output the pairs with their score
    for i in range(len(cosine_scores)):
        if cosine_scores[i][0] > 0.3:
            potential_duplicate = [float(cosine_scores[i][0]), bug_reports[i]]
            result.append(potential_duplicate)

    ordered = sorted(result, key=lambda x: x[0])

    return ordered[-3:]


def get_accuracy():
    data = np.load('../mozilla_firefox.npy', allow_pickle=True)
    total_number_of_duplicates = sum(1 for row in data if row[1] is not None)

    number_of_correctly_identified_duplicates = 0

    for row in data:
        if row[1] is None:
            continue

        description = row[2]

        most_similar_bugs = get_similar_bugs(description)

        for bug in most_similar_bugs:
            if bug[1]['issue_id'] == row[1]:
                number_of_correctly_identified_duplicates += 1
                break

    accuracy = number_of_correctly_identified_duplicates / total_number_of_duplicates

    print(f"Accuracy: {accuracy:.4f}")


start = time.time()

get_accuracy()
end = time.time()
print(f"\nTime to read: {round(end - start, 5)} seconds.")
