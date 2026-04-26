import re
import joblib
import csv
import numpy as np
from gensim.models import Word2Vec

#config
LOG_PATTERN = r'^(?P<ip>\S+) \S+ \S+ \[(?P<timestamp>.*?)\] "(?P<method>\S+) (?P<path>\S+) (?P<protocol>\S+)" (?P<status>\d{3}) (?P<size>\S+) "(?P<referrer>.*?)" "(?P<user_agent>.*?)"'
W2V_MODEL_PATH = "logfcb.model"
DETECTOR_PATH = "detector.pkl"
ANOMALY_LOG_CSV = "detected_anomalies.csv"
ANOMALY_VEC_CSV = "anomaly_vectors.csv"

#sensitivity (lower is more sensitive)
CUSTOM_THRESHOLD = -0.05 

def get_culprit(token_list, original_score, w2v_model, clf):
#identify cause for outlier
  features = ['ip', 'method', 'path', 'status', 'user_agent']
    contributions = {}
    
    for i, name in enumerate(features):
        reduced_tokens = [t for j, t in enumerate(token_list) if i != j]
        vecs = [w2v_model.wv[w] for w in reduced_tokens if w in w2v_model.wv]
        
        if vecs:
            reduced_vec = np.mean(vecs, axis=0).reshape(1, -1)
            new_score = clf.decision_function(reduced_vec)[0]
            contributions[name] = new_score - original_score
            
    return max(contributions, key=contributions.get) if contributions else "unknown"

#load prior trained models (vector and anomaly!)
def analyze_and_export(input_log_file):
    w2v_model = Word2Vec.load(W2V_MODEL_PATH)
    clf = joblib.load(DETECTOR_PATH)
    anomaly_counter = 0

    with open(ANOMALY_LOG_CSV, 'w', newline='') as f_log, \
         open(ANOMALY_VEC_CSV, 'w', newline='') as f_vec:
        

        #write original logs back to file when anomaly detected, with anomaly analysis
           
        log_fields = ['id', 'anomaly_score', 'top_culprit', 'ip', 'timestamp', 'method', 'path', 'status', 'user_agent']
        log_writer = csv.DictWriter(f_log, fieldnames=log_fields, extrasaction='ignore')
        log_writer.writeheader()

        vec_writer = csv.writer(f_vec)
        vec_writer.writerow(['id'] + [f'dim_{i}' for i in range(100)])

        print(f"Scanning {input_log_file} with threshold {CUSTOM_THRESHOLD}...")

        with open(input_log_file, 'r') as logs:
            for line in logs:
                match = re.match(LOG_PATTERN, line.strip())
                if not match: continue
                
                entry = match.groupdict()
                token_list = [entry['ip'], entry['method'], entry['path'], entry['status'], entry['user_agent']]
                vecs = [w2v_model.wv[w] for w in token_list if w in w2v_model.wv]
                
                if vecs:
                    avg_vec = np.mean(vecs, axis=0).reshape(1, -1)
                    score = clf.decision_function(avg_vec)[0]
                    
                    if score < CUSTOM_THRESHOLD:
                        anomaly_counter += 1
                        
                        culprit = get_culprit(token_list, score, w2v_model, clf)
                        
                        entry.update({
                            'id': anomaly_counter,
                            'anomaly_score': round(score, 4),
                            'top_culprit': culprit.upper()
                        })
                        log_writer.writerow(entry)
                        vec_writer.writerow([anomaly_counter] + avg_vec.flatten().tolist())

    print(f"Identified {anomaly_counter} anomalies.")

if __name__ == "__main__":
    analyze_and_export('access.log')
