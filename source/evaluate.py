#!/usr/bin/env python3
"""Evaluate one retained outer-fold checkpoint; export aggregate metrics only."""
from pathlib import Path
import argparse
import json
import os
import sys
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source'))
from para_drugrank import load_data, load_model
from para_drugrank.model import score_predictions
from para_drugrank.panels import select_panel

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--data',required=True,type=Path)
    p.add_argument('--panel',choices=['common','full','custom'],default='common')
    p.add_argument('--config',type=Path,default=ROOT/'configs/model.json')
    args=p.parse_args()
    data=select_panel(load_data(args.data),args.panel);model=load_model(args.checkpoint)
    if model.metadata['cohort']!=str(data['cohort'].item()):p.error('Checkpoint/data cohort mismatch')
    fold=int(model.metadata['outer_fold']);query=data['outer_fold']==fold
    prediction,experts=model.predict(data['expression'][query],data['source_response'][query],
                                    data['gene_names'],data['drug_names'],return_experts=True)
    config=json.loads(args.config.read_text());minimum=config['cohorts'][model.metadata['cohort']]['minimum_finite_drugs']
    methods={**experts,'PARA-DrugRank':prediction}
    results={name:{metric:float(np.mean(values)) for metric,values in score_predictions(data['target_response'][query],pred,minimum).items()}
             for name,pred in methods.items()}
    print(json.dumps({'cohort':model.metadata['cohort'],'outer_fold':fold,'query_rows':int(query.sum()),
                      'aggregation':'cell-model-macro within this single held-out fold','metrics':results},indent=2))

if __name__=='__main__':main()
