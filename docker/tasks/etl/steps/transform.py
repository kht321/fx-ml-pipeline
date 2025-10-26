import os, csv
DATA = os.environ.get("DATA_DIR","/data")
src = f"{DATA}/bronze/docker_tasks_events.csv"
os.makedirs(f"{DATA}/silver", exist_ok=True)
dst = f"{DATA}/silver/docker_tasks_clean.csv"
with open(src) as fin, open(dst,"w",newline="") as fout:
    r = csv.DictReader(fin)
    w = csv.DictWriter(fout, fieldnames=r.fieldnames)
    w.writeheader(); kept = 0
    for row in r:
        if int(row["feature_a"]) >= 3:
            w.writerow(row); kept += 1
print("transform kept", kept, "rows →", dst)
