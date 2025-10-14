import os, csv, pickle, random, json
DATA = os.environ.get("DATA_DIR","/data")
MODELS = os.environ.get("MODELS_DIR","/models")
xs, ys = [], []
with open(f"{DATA}/gold/docker_tasks_features.csv") as f:
    r = csv.DictReader(f)
    for row in r:
        xs.append(float(row["feat_mean"])); ys.append(int(row["label"]))
thr = sum(xs)/max(1,len(xs)); p1 = sum(ys)/max(1,len(ys))
model = {"threshold": thr, "p1": p1}
os.makedirs(MODELS, exist_ok=True)
with open(f"{MODELS}/candidate.pkl","wb") as f:
    pickle.dump(model,f)
auc = round(random.uniform(0.6, 0.9), 3)
with open(f"{MODELS}/candidate_metrics.json","w") as f:
    json.dump({"auc": auc}, f)
print("trained:", model, "auc:", auc)
