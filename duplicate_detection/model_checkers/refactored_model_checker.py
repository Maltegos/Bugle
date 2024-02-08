import time
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def preprocess_csv(filename):
    df = pd.read_csv(filename, sep=',', on_bad_lines='skip',
                     usecols=["Issue_id", "Duplicated_issue", "Description"])
    df = df.replace('\n', '', regex=True)

    np_array = df.to_numpy()
    np.save("../mozilla_firefox.npy", np_array)


def embed_bug_reports(filename):
    df = pd.read_csv(filename, sep=',', on_bad_lines='skip', usecols=["Issue_id", "Description"])
    df = df.replace('\n', '', regex=True)

    descriptions = [str(df.values[row][1]) for row in range(df.shape[0])]

    model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L3-v2')

    bug_report_embeddings = model.encode(descriptions)

    np.save('../test_embeddings.npy', bug_report_embeddings)


def load_embeddings():
    return np.load('../test_embeddings.npy')


def get_bug_reports():
    data = np.load('../mozilla_firefox.npy', allow_pickle=True)

    bug_reports = [{'issue_id': row[0], 'description': row[2]} for row in data]

    return bug_reports


def get_similar_bugs(input_embedding, bug_report_embeddings, bug_reports):
    # Reshape input_embedding to a 2D array
    input_embedding = input_embedding.reshape(1, -1)
    cosine_scores = cosine_similarity(bug_report_embeddings, input_embedding)

    result = []

    for i in range(len(cosine_scores)):
        if cosine_scores[i][0] > 0.3:
            potential_duplicate = [float(cosine_scores[i][0]), bug_reports[i]]
            result.append(potential_duplicate)

    ordered = sorted(result, key=lambda x: x[0])

    return ordered[-3:]


def get_accuracy():
    data = np.load('../mozilla_firefox.npy', allow_pickle=True)
    total_number_of_duplicates = sum(1 for row in data if row[1] is not None)

    bug_reports = get_bug_reports()
    bug_report_embeddings = load_embeddings()

    model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L3-v2')

    number_of_correctly_identified_duplicates = sum(
        1 for row in data if row[1] is not None
        for bug in get_similar_bugs(
            model.encode(row[2]), bug_report_embeddings, bug_reports
        ) if bug[1]['issue_id'] == row[1]
    )

    accuracy = number_of_correctly_identified_duplicates / total_number_of_duplicates

    print(f"Accuracy: {accuracy:.4f}")


start = time.time()

get_accuracy()
end = time.time()
print(f"\nTime to read: {round(end - start, 5)} seconds.")
