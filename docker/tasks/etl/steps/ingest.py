import os, csv, random
DATA = os.environ.get("DATA_DIR","/data")
os.makedirs(f"{DATA}/bronze", exist_ok=True)
path = f"{DATA}/bronze/docker_tasks_events.csv"
with open(path,"w",newline="") as f:
    w = csv.writer(f); w.writerow(["user_id","feature_a","label"])
    for i in range(100):
        w.writerow([i, random.randint(0,10), random.randint(0,1)])
print("ingest wrote", path)
