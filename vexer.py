import json
from gensim.models import Word2Vec
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import cython


def logs_to_vectors(input_file, output_file):
    with open(input_file, 'r') as f:
        logs = json.load(f)

    sentences = []
    for entry in logs:
        token_list = [entry['ip'], entry['method'], entry['path'], entry['status'], entry['user_agent']]
        sentences.append(token_list)

    #train word2vec
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=24)

    #create the vectors
    log_vectors = []
    for sentence in sentences:
        vecs = [model.wv[word] for word in sentence if word in model.wv]
        if vecs:
            avg_vec = np.mean(vecs, axis=0)
            log_vectors.append(avg_vec.tolist())

    with open(output_file, 'w') as f:
        json.dump(log_vectors, f)
    
    print(f"Vectorization complete. Saved {len(log_vectors)} vectors.")

    #return the data for storage and identifying the pre-vectorized log entry
    return model, log_vectors

#create the model
model, log_vectors = logs_to_vectors('access_out.txt', 'vectors_out.txt')
model.save("logfcb.model")

#train the model
clf = IsolationForest(contamination=0.01, n_jobs=4).fit(log_vectors)
joblib.dump(clf, "detector.pkl")

print("Models have been saved successfully.")
