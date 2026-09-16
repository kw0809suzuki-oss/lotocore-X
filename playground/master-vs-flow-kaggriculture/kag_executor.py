"""Shared minimal Kaggriculture executor for the isolated Master/Flow match.
Derived structurally from the public Apache-2.0 Seyamalam/Kaggriculture submission_v1 executor.
Policy differences are confined to Policy and crop/market decisions below.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any

CROPS={"WHEAT":(10,4,4),"CARROT":(20,3,3),"TOMATO":(50,11,4),"STRAWBERRY":(100,16,4),"MELON":(80,10,6)}
PRODUCTS=("MELON","STRAWBERRY","TOMATO","CARROT","WHEAT","MILK","WOOL","EGG")
LAND_PRICES=(1000,2000,4000)
HIRE=(1,1,2,3,5,8,13,21,34,55)

@dataclass(frozen=True)
class Policy:
    name:str; reserve:int; invest_until:int; liquidate_from:int; wheat_only:bool
MASTER=Policy("MASTER",150,17,21,True)
FLOW=Policy("FLOW",500,15,17,False)

def counts(farm):
    out={c:0 for c in CROPS}
    for row in farm["tiles"]:
        for t in row:
            if isinstance(t,dict) and t.get("kind")=="PLANT" and t.get("crop") in out: out[t["crop"]]+=1
    return out

def score_crop(obs,crop):
    cost, maturity, yld=CROPS[crop]; day=int(obs.get("day",0)); left=30-day
    if maturity>=left:return -1e9
    price=int(obs["market"]["prices"].get(crop,1))
    own=counts(obs["farms"][obs["player"]])[crop]
    other=counts(obs["farms"][1-obs["player"]])[crop]
    crowd=1.0+0.025*(own+other)
    return (yld*price/crowd-cost)/maturity

def choose_crop(obs,p):
    if p.wheat_only:return "WHEAT"
    ranked=[c for c in CROPS if c!="WHEAT"]
    best=max(ranked,key=lambda c:(score_crop(obs,c),c))
    return best if score_crop(obs,best)>0 else "WHEAT"

def should_harvest(t,obs,p):
    if int(t.get("yield_units",0))<=0:return False
    day=int(obs.get("day",0)); hour=int(obs.get("hour",0))
    if day>=29:return hour<13
    if day>=p.liquidate_from:return True
    crop=t.get("crop"); planted=int(t.get("planted_day",day)); age=day-planted
    return age>=CROPS.get(crop,(0,99,0))[1]

def task_list(obs,p):
    farm=obs["farms"][obs["player"]]; private=obs["private"]; tasks=[]; empty=[]
    for y,row in enumerate(farm["tiles"]):
        for x,t in enumerate(row):
            if t is None: empty.append((x,y)); continue
            if not isinstance(t,dict):continue
            if t.get("kind")=="WEED":tasks.append((3,x,y,["DIG"]));continue
            if t.get("kind")!="PLANT":continue
            ready=should_harvest(t,obs,p)
            if not t.get("watered_today",False) and int(obs.get("day",0))<29:
                tasks.append((-1 if int(t.get("consecutive_unwatered",0))>=1 else (0 if ready else 1),x,y,["WATER"]))
            elif ready:tasks.append((0,x,y,["HARVEST"]))
    if int(obs.get("hour",0))>=15 or int(obs.get("day",0))>p.invest_until:return tasks
    seeds={c:int(private.get("seeds",{}).get(c,0)) for c in CROPS}; crop=choose_crop(obs,p)
    for x,y in empty:
        if seeds[crop]<=0:break
        tasks.append((4,x,y,["PLANT",crop]));seeds[crop]-=1
    return tasks

def dist(a,b):return abs(a[0]-b[0])+abs(a[1]-b[1])
def step(a,b):
    x,y=a;tx,ty=b;dx,dy=tx-x,ty-y
    if abs(dx)>=abs(dy) and dx:return ["EAST" if dx>0 else "WEST"]
    if dy:return ["SOUTH" if dy>0 else "NORTH"]
    return ["PASS"]

def unit_actions(obs,p):
    farm=obs["farms"][obs["player"]]; pos=[tuple(farm["farmer"]),*(tuple(q) for q in farm.get("hands",[]))]
    inv=obs.get("private",{}).get("inventories",[]); shed=obs.get("private",{}).get("shed",{}); used=sum(map(int,shed.values())); room=100 if used==0 else 0
    n=len(farm["tiles"]); h=n//2; access=((h-1,h-1),(h,h-1),(h-1,h),(h,h)); assign={}; free=set(range(len(pos))); hour=int(obs.get("hour",0))
    for i in list(free):
        carried=sum(int(v) for v in (inv[i] if i<len(inv) else {}).values())
        urgent=int(obs.get("day",0))>=p.liquidate_from
        if carried<=0 or (not urgent and hour<17 and carried<24):continue
        target=min(access,key=lambda q:(dist(pos[i],q),q))
        act=["DROP"] if pos[i]==target and carried<=room else (["PASS"] if pos[i]==target else step(pos[i],target))
        if act[0]=="DROP":room-=carried
        assign[i]=(target[0],target[1],act);free.remove(i)
    tasks=task_list(obs,p)
    for i in sorted(list(free)):
        local=[(t[0],j) for j,t in enumerate(tasks) if t[0]<=0 and (t[1],t[2])==pos[i]]
        if local:
            _,j=min(local);t=tasks.pop(j);assign[i]=(t[1],t[2],t[3]);free.remove(i)
    for pri in sorted({t[0] for t in tasks}):
        group=[t for t in tasks if t[0]==pri]
        while group and free:
            _,i,j=min((dist(pos[i],(t[1],t[2])),i,j) for i in free for j,t in enumerate(group));t=group.pop(j);assign[i]=(t[1],t[2],t[3]);free.remove(i)
    acts=[]
    for i,q in enumerate(pos):
        a=assign.get(i);acts.append(["PASS"] if a is None else (a[2] if q==(a[0],a[1]) else step(q,(a[0],a[1]))))
    return acts[0],acts[1:]

def hire_cost(start,count):return sum(HIRE[start:start+count])
def market_actions(obs,p):
    farm=obs["farms"][obs["player"]];private=obs["private"];day=int(obs.get("day",0));hour=int(obs.get("hour",0));money=float(farm["money"]);orders=[]
    shed=private.get("shed",{})
    # Both policies convert inventory once their frozen liquidation boundary is reached;
    # before that, Flow also sells mature goods promptly to preserve liquidity.
    can_sell=day>=p.liquidate_from or (p.name=="FLOW" and day>=3)
    if can_sell:
        for product in sorted(PRODUCTS,key=lambda x:-int(obs["market"]["prices"].get(x,0))):
            q=int(shed.get(product,0))
            if q>0 and len(orders)<10:orders.append(["SELL",product,q])
    active=sum(counts(farm).values()); seeds=sum(int(private.get("seeds",{}).get(c,0)) for c in CROPS)
    unlocked=len(farm.get("unlocked_quadrants",["NW"])); land=LAND_PRICES[unlocked-1] if unlocked<=len(LAND_PRICES) else None
    if land is not None and unlocked<3 and day<=p.invest_until and money>=land+p.reserve+500 and len(orders)<10:
        orders.append(["BUY_LAND"]);money-=land
    hands=len(farm.get("hands",[])); workload=active+seeds; target=min(10,max(4,math.ceil(workload/14)+2))
    if hour<=2 and day<=p.invest_until and hands<target:
        desired=min(target-hands,10-len(orders));aff=0
        for k in range(1,desired+1):
            if hire_cost(int(farm.get("hires_today",0)),k)<=max(0,money-p.reserve):aff=k
        for _ in range(aff):orders.append(["HIRE"])
        money-=hire_cost(int(farm.get("hires_today",0)),aff)
    capacity=min(75,25*unlocked);gap=max(0,capacity-active-seeds)
    if gap>0 and day<=p.invest_until and len(orders)<10:
        crop=choose_crop(obs,p);cost=CROPS[crop][0];spend=max(0,money-p.reserve)
        q=min(12,gap,int(spend//cost))
        if q>0:orders.append(["BUY_SEED",crop,q])
    return orders[:10]

def act(obs,p):
    try:
        farmer,hands=unit_actions(obs,p)
        return {"farmer":farmer,"hands":hands,"market":market_actions(obs,p)}
    except Exception:
        farms=obs.get("farms",[]);pl=int(obs.get("player",0));n=len(farms[pl].get("hands",[])) if pl<len(farms) else 0
        return {"farmer":["PASS"],"hands":[["PASS"] for _ in range(n)],"market":[]}
