"""Recompute archived v4 statistics offline from released per-question grades."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from context_ledger.eval.analyze_v4 import analyze_v4

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis',choices=['core','latest'],default='core')
    p.add_argument('--out',type=Path,default=ROOT/'reproduced/core')
    p.add_argument('--resamples',type=int,default=10000)
    a=p.parse_args()
    rows=[json.loads(line) for file in sorted((ROOT/'runs/v4/per-question').glob('*.jsonl')) for line in file.open()]
    if a.analysis=='core':
        rows=[r for r in rows if not(r['model']=='claude-sonnet-5-5' and r['arm']=='L4-nogate')]
    label='grid-analysis-c260-fix1' if a.analysis=='core' else 'grid-analysis-c260-nogate1'
    meta=json.loads((ROOT/'runs/v4'/label/'tests.json').read_text())['meta']
    result=analyze_v4(rows,a.out,resamples=a.resamples,seed=0,meta=meta)
    print(json.dumps({'analysis':a.analysis,'graded_rows':len(rows),'resamples':a.resamples,'seed':0,'confirmatory_tests':sum(t['role']=='confirmatory' for t in result['tests']),'output':str(a.out)}))
if __name__=='__main__':main()
