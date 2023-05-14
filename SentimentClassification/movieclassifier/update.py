import pickle
import sqlite3
import numpy as np
import os
#importing HashingVectorizer from the local ldir
from vectorizer import vect


def update_model(db_path, model, batch_size=10000):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('SELECT * from review_db')
    
    results = c.fetchmany(batch_size)
    while results:
        data = np.array(results)
        X = data[:, 0]
        y = data[:, 1].astype(int)
        
        classes = np.array([0,1])
        xtrain = vect.transform(X)
        model.partial_fit(xtrain, y, classes=classes)
        results = c.fetchmany(batch_size)
    conn.close()
    return model

crnt_dir = os.path.dirname(__file__)

clf = pickle.load(open(os.path.join(crnt_dir, 'pkl_objects', 'classifier.pkl'), 'rb'))
db = os.path.join(crnt_dir, 'reviews.sqlite')

clf = update_model(db_path=db, model=clf, batch_size=10000)

# Updating classifier.pkl file permmanently

pickle.dump(clf, open(os.path.join(crnt_dir, 'pkl_objects', 'classifier.pkl'), 'wb'), protocol=4)