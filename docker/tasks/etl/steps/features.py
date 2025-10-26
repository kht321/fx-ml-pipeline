import os, csv
DATA = os.environ.get("DATA_DIR","/data")
os.makedirs(f"{DATA}/gold", exist_ok=True)
src = f"{DATA}/silver/docker_tasks_clean.csv"
dst = f"{DATA}/gold/docker_tasks_features.csv"
agg = {}
with open(src) as f:
    r = csv.DictReader(f)
    for row in r:
        uid = row["user_id"]
        agg.setdefault(uid, {"sum":0,"cnt":0,"label":row["label"]})
        agg[uid]["sum"] += int(row["feature_a"]); agg[uid]["cnt"] += 1
with open(dst,"w",newline="") as f:
    w = csv.writer(f); w.writerow(["user_id","feat_mean","label"])
    for uid, a in agg.items():
        w.writerow([uid, a["sum"]/max(1,a["cnt"]), a["label"]])
print("features →", dst)
