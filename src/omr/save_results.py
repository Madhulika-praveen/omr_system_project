import csv

SUBJECTS = ["Python", "EDA", "SQL", "POWER BI", "Statistics"]

def save_results(results, out_file="results.csv"):
    with open(out_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sheet", "set"] + SUBJECTS + ["Total"])
        for r in results:
            writer.writerow([
                r["sheet"],
                r["set"],
                *[r["scores"].get(subj, 0) for subj in SUBJECTS],
                r["total"]
            ])
