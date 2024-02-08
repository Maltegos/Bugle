# This class evaluates the performance of the model based on the percentage of correctly identified duplicates from the
# Mozilla Firefox repository

import pandas as pd
import test_sbert_duplicate_finder


def get_accuracy():
    # reads Mozilla Firefox bug reports into pandas DataFrame df and pickles into mozilla_firefox.npy
    # pandas DataFrame containing bug report data
    # TODO: make filename dynamic, i.e. pass as arg upon call
    df = pd.read_csv('../data/mozilla_firefox.csv', sep=',', on_bad_lines='skip',
                     usecols=["Issue_id", "Duplicated_issue", "Description"])
    df = df.replace('\n', '', regex=True)

    np_array = df.to_numpy()
    # TODO: make filename dynamic, i.e. pass as arg upon call
    np.save("mozilla_firefox.npy", np_array)

    # rows in df
    df_rows = range(df.shape[0])

    number_of_correctly_identified_duplicates = 0
    total_number_of_duplicates = len(df_rows) - df['Duplicated_issue'].isna().sum()

    print(total_number_of_duplicates)

    # iterates over bug reports and evaluates if our model finds the correct duplicates
    for row in df_rows:
        description = df[r'Description'][row]
        # get_similar_bugs() on each row of the DataFrame
        most_similar_bugs = test_sbert_duplicate_finder.get_similar_bugs(description, df)
        print(row)
        for bug in most_similar_bugs:
            # if most_similar_bugs contains the Duplicated_issue for that row
            if bug[1]['issue_id'] == df[r'Duplicated_issue'][row]:
                # update number_of_correctly_identified_duplicates
                number_of_correctly_identified_duplicates += 1
            else:
                continue
    
    # accuracy = number_of_correctly_identified_duplicates / total_number_of_duplicates
    accuracy = number_of_correctly_identified_duplicates / total_number_of_duplicates
    
    print(accuracy)


get_accuracy()
