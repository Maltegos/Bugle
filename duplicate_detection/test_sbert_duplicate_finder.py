import pandas as pd
from sentence_transformers import SentenceTransformer, util
import test_embedder

# Load the embeddings once
bug_report_embeddings = test_embedder.load_embeddings()


def get_bug_reports(df):
    """"# pandas DataFrame containing bug report data
    df = pd.read_csv('data/mozilla_firefox.csv', sep=',', on_bad_lines='skip', usecols=["Issue_id", "Description"])
    df = df.replace('\n', '', regex=True)
    """
    # to store bug reports as dictionaries of {'issue_id', 'description'}
    bug_reports = []

    for row in range(df.shape[0]):
        # put row values in dictionary {'issue_id': , 'description': } and append to bug_reports
        row_values = dict(issue_id=df.values[row][0], description=df.values[row][2])
        bug_reports.append(row_values)

    return bug_reports


def get_similar_bugs(description, bug_reports, model):
    # Encode embedding for input description
    input_embedding = model.encode(description)
    # Compute cosine-similarities
    cosine_scores = util.cos_sim(bug_report_embeddings, input_embedding)

    result = []

    # Output the pairs with their score
    for i in range(len(cosine_scores)):
        if cosine_scores[i][0] > 0.3:
            potential_duplicate = [float(cosine_scores[i][0]), bug_reports[i]]
            result.append(potential_duplicate)

    ordered = sorted(result, key=lambda x: x[0])

    return ordered[-3:]
