from __future__ import annotations
import copy
from pathlib import Path
import pandas as pd
import run_original_core_region as m

CSV=Path(__file__).resolve().parents[1] / 'data' / 'loto7.csv'
df=pd.read_csv(CSV).sort_values('round').reset_index(drop=True)
draws=[list(map(int,row)) for row in df[m.COLS].to_numpy()]
states=[m.state(d) for d in draws]

def sig(ti, sts):
    r,n,_=m.predict_region(sts,ti,20,.25,.75)
    return r,n

def test_target_hidden():
    ti=len(states)-1; a=sig(ti,states)
    s2=copy.deepcopy(states); s2[ti]=m.state([1,2,3,4,5,6,7])
    assert sig(ti,s2)==a

def test_future_hidden():
    ti=len(states)-20; a=sig(ti,states)
    s2=copy.deepcopy(states)
    for k in range(ti,len(s2)):
        s2[k]=m.state([31,32,33,34,35,36,37])
    assert sig(ti,s2)==a

def test_determinism():
    ti=len(states)-1
    assert sig(ti,states)==sig(ti,states)

def test_neighbor_successor_is_past():
    ti=len(states)-1
    _,ns,_=m.predict_region(states,ti,20,.25,.75)
    assert max(j+1 for j in ns) <= ti-1

if __name__=='__main__':
    for f in [test_target_hidden,test_future_hidden,test_determinism,test_neighbor_successor_is_past]:
        f(); print('PASS',f.__name__)
