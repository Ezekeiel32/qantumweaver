import os
import json
import csv

JOBS_DIR = '/home/chezy/Desktop/tetrazpe/training_jobs'
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), '..', 'zpe_dataset.csv')

rows = []

for fname in os.listdir(JOBS_DIR):
    if fname.endswith('.json'):
        job_id = fname.replace('.json', '')
        with open(os.path.join(JOBS_DIR, fname), 'r') as f:
            job = json.load(f)
            for entry in job.get('zpe_history', []):
                epoch = entry.get('epoch')
                zpe_effects = entry.get('zpe_effects', [])
                # Pad or trim to 6 values
                zpe_effects = (zpe_effects + [None]*6)[:6]
                rows.append([
                    job_id,
                    epoch,
                    *zpe_effects
                ])

with open(OUTPUT_CSV, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['job_id', 'epoch', 'zpe1', 'zpe2', 'zpe3', 'zpe4', 'zpe5', 'zpe6'])
    writer.writerows(rows)

print(f"Exported {len(rows)} rows to {OUTPUT_CSV}") 