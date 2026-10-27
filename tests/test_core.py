import copy
import importlib
import math
from pathlib import Path
import random
import tempfile
import unittest

import networkx as nx
import numpy as np

from SwarmGraph import SwarmGraph
from COOP_Calc import calculate_coop_score, dist_Rlink
from run_demo import choose_node, run


class CoreTests(unittest.TestCase):
    def setUp(self):
        random.seed(10); np.random.seed(10)

    def test_requested_node_count_and_degree_cap(self):
        for n in [2,28,50,101]:
            graph=SwarmGraph(num_nodes=n).get_graph()
            self.assertEqual(len(graph),n)
            self.assertLessEqual(max(dict(graph.degree()).values()),5)
            self.assertFalse(list(nx.selfloop_edges(graph)))
            for a,b in graph.edges:
                self.assertLessEqual(np.linalg.norm(graph.nodes[a]['pos']-graph.nodes[b]['pos']),1)

    def test_empty_movement(self):
        swarm=SwarmGraph(num_nodes=2)
        swarm.G.clear()
        swarm.update_positions()
        self.assertEqual(len(swarm.G),0)

    def test_pair_normalisation(self):
        graph=nx.path_graph(2)
        for node in graph:
            graph.nodes[node].update(weight=1.0,pos=np.array([0.,0.,4.]))
        score,_,factor,data=calculate_coop_score(graph,0,{},alpha=0)
        self.assertEqual(data['original_possible_connections'],1)
        self.assertAlmostEqual(factor,math.exp(-.5))
        self.assertAlmostEqual(score,math.exp(-.5))
        graph.remove_node(0)
        score,_,_,_=calculate_coop_score(graph,1,{},data)
        self.assertEqual(score,0)

    def test_disconnected_and_zero_weight(self):
        graph=nx.empty_graph(2)
        for node in graph:
            graph.nodes[node].update(weight=0.,pos=np.zeros(3))
        self.assertEqual(dist_Rlink(graph,0,1),float('inf'))
        self.assertEqual(calculate_coop_score(graph,0,{})[0],0)
        self.assertEqual(calculate_coop_score(nx.Graph(),1,{})[0],0)

    def test_greedy_trials_do_not_modify_parent(self):
        graph=SwarmGraph(num_nodes=8).get_graph()
        _,_,_,data=calculate_coop_score(graph,0,{})
        before=copy.deepcopy(graph)
        node=choose_node(graph,'greedy',data,1)
        self.assertIn(node,graph)
        for key in graph:
            self.assertEqual(graph.nodes[key]['Theta'],before.nodes[key]['Theta'])
            np.testing.assert_array_equal(graph.nodes[key]['pos'],before.nodes[key]['pos'])

    def test_all_demo_methods_and_repeatability(self):
        with tempfile.TemporaryDirectory() as tmp:
            for method in ['degree','betweenness','closeness','weight','greedy']:
                folder=Path(tmp)/method
                first=run(nodes=8,steps=3,method=method,output=folder)
                second=run(nodes=8,steps=3,method=method,output=folder)
                self.assertEqual(first,second)
                self.assertEqual([r['remaining_nodes'] for r in first],[8,7,6,5])
                for name in ['metrics.csv','config.json','score.png']:
                    self.assertGreater((folder/name).stat().st_size,0)

    def test_remove_every_node(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows=run(nodes=2,steps=2,method='greedy',output=tmp)
            self.assertEqual(rows[-1]['score'],0)
            self.assertEqual(rows[-1]['remaining_nodes'],0)

    def test_invalid_demo_dimensions(self):
        with self.assertRaises(ValueError):run(nodes=0)
        with self.assertRaises(ValueError):run(nodes=8,steps=9)

    def test_non_dqn_modules_import_without_tensorflow(self):
        for module in ['Drone_Swarms_Code','Greedy_Heuristic_Code','Removal_Methods','Power_of_attck_Optimal_Check','utilities']:
            importlib.import_module(module)

    def test_exhaustive_worker_does_not_reuse_mutable_state(self):
        from Power_of_attck_Optimal_Check import worker
        swarm=SwarmGraph(num_nodes=8)
        graph=swarm.get_graph()
        initial=copy.deepcopy(graph)
        first=worker(graph,(0,1),np.array([9,9,4]),swarm)
        second=worker(graph,(0,1),np.array([9,9,4]),swarm)
        self.assertEqual(first,second)
        self.assertEqual(set(graph),set(initial))
        for node in graph:
            np.testing.assert_array_equal(graph.nodes[node]['pos'],initial.nodes[node]['pos'])
            self.assertEqual(graph.nodes[node]['Theta'],initial.nodes[node]['Theta'])


if __name__=='__main__':unittest.main()
