"""Small, repeatable software example using the repository's graph and score code.

This is a maintenance example, not a reproduction of the article's experiments.
"""
import argparse
import copy
import csv
import json
from pathlib import Path
import platform
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from SwarmGraph import SwarmGraph
from COOP_Calc import calculate_coop_score


def choose_node(graph, method, original_data, step):
    if method == "greedy":
        # Every trial starts from the same state; scoring updates node Theta.
        def score_after_removal(node):
            candidate = copy.deepcopy(graph)
            candidate.remove_node(node)
            return calculate_coop_score(candidate,step,{},copy.deepcopy(original_data))[0]
        return min(sorted(graph),key=score_after_removal)
    if method == "degree":
        scores = dict(graph.degree())
    elif method == "betweenness":
        scores = nx.betweenness_centrality(graph)
    elif method == "closeness":
        scores = nx.closeness_centrality(graph)
    elif method == "weight":
        scores = nx.get_node_attributes(graph,"weight")
    else:
        raise ValueError("unknown method")
    return max(sorted(graph),key=lambda node:scores[node])


def run(nodes=27, steps=3, seed=10, method="degree", output=Path("outputs/demo")):
    if not 2 <= nodes <= 1000 or not 0 <= steps <= nodes:
        raise ValueError("nodes must be 2-1000 and steps must be 0-nodes")
    random.seed(seed)
    np.random.seed(seed)
    swarm=SwarmGraph(num_nodes=nodes)
    graph=swarm.get_graph()
    records=[]
    original_data=None
    initial_score=None
    removed=None
    for step in range(steps+1):
        score,_,factor,original_data=calculate_coop_score(graph,step,{},original_data)
        if initial_score is None:
            initial_score=score
        records.append({"step":step,"removed_node":removed,"remaining_nodes":len(graph),
                        "edges":graph.number_of_edges(),
                        "largest_component":max(map(len,nx.connected_components(graph)),default=0),
                        "score":score,"score_ratio":score/initial_score if initial_score else None,
                        "component_factor":factor})
        if step==steps:
            break
        removed=choose_node(graph,method,original_data,step+1)
        graph.remove_node(removed)
        swarm.update_positions()
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    with (output/"metrics.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]))
        writer.writeheader();writer.writerows(records)
    metadata={"nodes":nodes,"steps":steps,"seed":seed,"method":method,
              "python":platform.python_version(),"numpy":np.__version__,
              "networkx":nx.__version__,"matplotlib":matplotlib.__version__,
              "scope":"maintenance demo; not a paper reproduction",
              "movement":"centroid translation; no edge rebuilding after movement"}
    (output/"config.json").write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8")
    fig,ax=plt.subplots(figsize=(6,4))
    ax.plot([r["step"] for r in records],[r["score"] for r in records],marker="o")
    ax.set(xlabel="Removal step",ylabel="Repository cooperative score",title=f"{method}, seed {seed}")
    fig.tight_layout();fig.savefig(output/"score.png",dpi=150);plt.close(fig)
    return records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes",type=int,default=27)
    parser.add_argument("--steps",type=int,default=3)
    parser.add_argument("--seed",type=int,default=10)
    parser.add_argument("--method",choices=["degree","betweenness","closeness","weight","greedy"],default="degree")
    parser.add_argument("--output",type=Path,default=Path("outputs/demo"))
    args=parser.parse_args()
    try:
        records=run(**vars(args))
    except ValueError as error:
        parser.error(str(error))
    print(f"Wrote {len(records)} rows, config.json and score.png to {args.output}")


if __name__=="__main__":
    main()
