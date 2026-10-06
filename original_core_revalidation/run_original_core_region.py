from __future__ import annotations
import argparse, hashlib, json, random
from pathlib import Path
import numpy as np
import pandas as pd

COLS=[f'n{i}' for i in range(1,8)]

def state(nums):
    a=np.asarray(nums,dtype=float)
    c=float(a.mean()); v=float(((a-c)**2).mean())
    s=np.sort(a); gaps=np.diff(s)
    return {'center':c,'variance':v,'range':float(s[-1]-s[0]),'max_gap':float(gaps.max()) if len(gaps) else 0.0}

def sha256_file(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def quantile_region(successor_states,qlo=.25,qhi=.75):
    cs=np.array([x['center'] for x in successor_states]); vs=np.array([x['variance'] for x in successor_states])
    return {'center_lo':float(np.quantile(cs,qlo)),'center_hi':float(np.quantile(cs,qhi)),
            'variance_lo':float(np.quantile(vs,qlo)),'variance_hi':float(np.quantile(vs,qhi))}

def hit(region,s):
    return int(region['center_lo']<=s['center']<=region['center_hi'] and region['variance_lo']<=s['variance']<=region['variance_hi'])

def predict_region(states,target_idx,k,qlo,qhi):
    current_idx=target_idx-1
    hist=states[:target_idx]
    x=np.array([[s['center'],s['variance']] for s in hist],dtype=float)
    mu=x.mean(axis=0); sd=x.std(axis=0); sd=np.where(sd==0,1.0,sd)
    cur=(x[current_idx]-mu)/sd
    cand=list(range(0,current_idx))
    d=[]
    for j in cand:
        z=(x[j]-mu)/sd
        dist=float(np.linalg.norm(z-cur))
        d.append((dist,j))
    d.sort(key=lambda z:(z[0],z[1]))
    neigh=[j for _,j in d[:k]]
    succ=[states[j+1] for j in neigh]
    return quantile_region(succ,qlo,qhi),neigh,hist

def random_same_width_regions(region,hist_states,repeats,seed,mode='empirical_anchor'):
    cvals=[s['center'] for s in hist_states]; vvals=[s['variance'] for s in hist_states]
    cmin,cmax=min(cvals),max(cvals); vmin,vmax=min(vvals),max(vvals)
    cw=region['center_hi']-region['center_lo']; vw=region['variance_hi']-region['variance_lo']
    rng=random.Random(seed)
    out=[]
    for _ in range(repeats):
        if mode=='uniform_box':
            clo=cmin if cmax-cmin<=cw else rng.uniform(cmin,cmax-cw)
            vlo=vmin if vmax-vmin<=vw else rng.uniform(vmin,vmax-vw)
        elif mode=='empirical_anchor':
            anchor=hist_states[rng.randrange(len(hist_states))]
            clo=cmin if cmax-cmin<=cw else max(cmin,min(anchor['center']-cw/2,cmax-cw))
            vlo=vmin if vmax-vmin<=vw else max(vmin,min(anchor['variance']-vw/2,vmax-vw))
        else:
            raise ValueError(mode)
        out.append({'center_lo':clo,'center_hi':clo+cw,'variance_lo':vlo,'variance_hi':vlo+vw})
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--csv',required=True)
    ap.add_argument('--targets',type=int,default=100)
    ap.add_argument('--k',type=int,default=20)
    ap.add_argument('--qlo',type=float,default=.25)
    ap.add_argument('--qhi',type=float,default=.75)
    ap.add_argument('--random-repeats',type=int,default=1000)
    ap.add_argument('--seed',type=int,default=20261006)
    ap.add_argument('--out',required=True)
    a=ap.parse_args()

    df=pd.read_csv(a.csv).sort_values('round').reset_index(drop=True)
    draws=[list(map(int,row)) for row in df[COLS].to_numpy()]
    states=[state(d) for d in draws]
    if len(df)<a.targets+a.k+2:
        raise SystemExit('insufficient history')

    target_indices=list(range(len(df)-a.targets,len(df)))
    null_emp=[0]*a.random_repeats
    null_uniform=[0]*a.random_repeats
    rows=[]
    core_total=0

    for ti in target_indices:
        reg,neigh,hist=predict_region(states,ti,a.k,a.qlo,a.qhi)
        actual=states[ti]
        ch=hit(reg,actual)
        core_total+=ch
        seed=a.seed+int(df.loc[ti,'round'])*1009
        nulls_emp=random_same_width_regions(reg,hist,a.random_repeats,seed+777,'empirical_anchor')
        nulls_uni=random_same_width_regions(reg,hist,a.random_repeats,seed,'uniform_box')
        nh_emp=[hit(r,actual) for r in nulls_emp]
        nh_uni=[hit(r,actual) for r in nulls_uni]
        null_emp=[x+y for x,y in zip(null_emp,nh_emp)]
        null_uniform=[x+y for x,y in zip(null_uniform,nh_uni)]
        rows.append({
            'target_round':int(df.loc[ti,'round']),
            'history_end_round':int(df.loc[ti-1,'round']),
            'core_hit':ch,
            'predicted_region':reg,
            'actual_state':actual,
            'neighbor_rounds':[int(df.loc[j,'round']) for j in neigh],
            'empirical_null_hit_rate':float(sum(nh_emp)/len(nh_emp)),
            'uniform_null_hit_rate':float(sum(nh_uni)/len(nh_uni))
        })

    arr=np.array(null_emp,dtype=float)
    uni=np.array(null_uniform,dtype=float)
    summary={
      'experiment_id':'original_core_revalidation_v0',
      'target_round_start':int(df.loc[target_indices[0],'round']),
      'target_round_end':int(df.loc[target_indices[-1],'round']),
      'targets':len(target_indices),
      'features':['center','variance'],
      'distance':'past_only_zscore_euclidean',
      'neighbor_k':a.k,
      'region_quantiles':[a.qlo,a.qhi],
      'random_repeats':a.random_repeats,
      'random_seed':a.seed,
      'past_only':True,
      'core_hits':core_total,
      'core_hit_rate':core_total/len(target_indices),
      'primary_null':'same_width_empirical_anchor',
      'random_hits_mean':float(arr.mean()),
      'random_hit_rate_mean':float(arr.mean()/len(target_indices)),
      'random_hits_median':float(np.median(arr)),
      'random_hits_p05':float(np.quantile(arr,.05)),
      'random_hits_p95':float(np.quantile(arr,.95)),
      'difference_hits_vs_random_mean':float(core_total-arr.mean()),
      'difference_rate_vs_random_mean':float(core_total/len(target_indices)-arr.mean()/len(target_indices)),
      'null_percentile_le_core':float(np.mean(arr<=core_total)),
      'null_p_ge_core':float(np.mean(arr>=core_total)),
      'secondary_uniform_null':{
          'mean_hits':float(uni.mean()),
          'median_hits':float(np.median(uni)),
          'p05':float(np.quantile(uni,.05)),
          'p95':float(np.quantile(uni,.95)),
          'p_ge_core':float(np.mean(uni>=core_total))
      },
      'csv_sha256':sha256_file(a.csv),
    }
    out={'summary':summary,'rows':rows}
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    Path(a.out).write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
