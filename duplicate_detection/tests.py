import time

import sbert_duplicate_finder
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Create your tests here.
def get_top_three():
    top_3 = sbert_duplicate_finder.get_similar_bugs('''*Comment*
    The connectors on the powerboards ar to small.
    
    *Action*
    Change to 6,3mm whit 0,8mm thicktes flat connectors.
    
    
    *Implementation*
    Shematic:*DONE*
    Layout: *DONE*
    
    
    *Verification*''')

    print(top_3)


start = time.time()
data = np.load('mozilla_firefox.npy', allow_pickle=True)

# test_model_checker.get_accuracy()
end = time.time()
print(f"\nTime to read: {round(end - start, 5)} seconds.")