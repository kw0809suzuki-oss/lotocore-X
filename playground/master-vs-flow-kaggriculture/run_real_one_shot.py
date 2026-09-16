"""Run exactly one authoritative Kaggriculture episode: MASTER seat0 vs FLOW seat1."""
import json, pathlib, sys
from kaggle_environments import make
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from kag_executor import MASTER,FLOW,act

def master(obs): return act(obs,MASTER)
def flow(obs): return act(obs,FLOW)

def main():
    seed=20260916
    env=make("kaggriculture",configuration={"episodeSteps":720,"seed":seed},debug=True)
    env.run([master,flow])
    final=env.steps[-1]
    rewards=[float(s.reward or 0) for s in final]
    result={"engine":"kaggle-environments kaggriculture","version":"1.32.4","seed":seed,"episodes":1,"seats":{"0":"MASTER","1":"FLOW"},"terminal_money":{"MASTER":rewards[0],"FLOW":rewards[1]},"margin_master_minus_flow":rewards[0]-rewards[1],"winner":"MASTER" if rewards[0]>rewards[1] else "FLOW" if rewards[1]>rewards[0] else "TIE","statuses":[str(s.status) for s in final],"frozen_no_rematch":True}
    out=HERE/"real_one_shot_result.json";out.write_text(json.dumps(result,indent=2)+"\n")
    (HERE/"real_one_shot_replay.json").write_text(json.dumps(env.toJSON())+"\n")
    print(json.dumps(result,indent=2))
if __name__=="__main__":main()
