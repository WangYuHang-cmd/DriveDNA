#!/usr/bin/env python3
"""
E5 (internal check): descriptor-anchor T2 metrics on the SAME evaluation
population as the learned models (val ∪ few_shot_heldout drivers), to verify
the Table 2 descriptor row is population-consistent.
Writes results/e5_descriptor_pop.json.
"""
import sys, json
import numpy as np
import pandas as pd

BASE = "."
sys.path.insert(0, f"{BASE}/code/eval")
from harness import enrollment_protocol, STATS
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

W = pd.read_parquet(f"{BASE}/data/segments/windows.parquet").reset_index(drop=True)
folds = json.load(open(f"{BASE}/data/splits/driver_folds.json"))
pop = set(folds["val"]) | set(folds["few_shot_heldout"])
m = W.driver.isin(pop).to_numpy()
print(f"eval windows {m.sum()} drivers {W.driver[m].nunique()}", flush=True)

D = W.loc[m, STATS].to_numpy(float)
D = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(D))
res = enrollment_protocol(D, W.driver[m].to_numpy(), W.route[m].to_numpy())
out = {str(k): v for k, v in res.items()}
for k, v in out.items():
    print(f"k={k}: top1 {v['top1']:.3f} top5 {v['top5']:.3f} auroc {v['auroc']:.3f} "
          f"eer {v['eer']:.3f} n_drivers {v['n_drivers']}", flush=True)
json.dump(out, open(f"{BASE}/results/e5_descriptor_pop.json", "w"), indent=1)
print("saved results/e5_descriptor_pop.json", flush=True)
