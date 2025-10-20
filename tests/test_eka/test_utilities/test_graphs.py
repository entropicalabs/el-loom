"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

import unittest
import networkx as nx
import numpy as np

from loom.eka import (
    Stabilizer,
    cartesian_product_tanner_graphs,
    verify_css_code_stabilizers,
    ClassicalTannerGraph,
    TannerGraph,
    ClassicalParityCheckMatrix,
    ParityCheckMatrix,
)
from loom.eka.utilities import (
    find_maximum_matching,
    minimum_edge_coloring,
    extract_subgraphs_from_edge_labels,
    cardinality_distribution,
)


# pylint: disable=invalid-name, too-many-instance-attributes, too-many-lines, too-many-statements, too-many-locals, too-many-public-methods, duplicate-code
class TestGraphsUtilities(unittest.TestCase):
    """Unit tests for graph utilities."""

    def setUp(self):

        ### HAMMING AND STEANE CODE

        # Parity-check matrix for a component of the Hamming code
        self.H_hamming = np.array(
            [[1, 1, 1, 1, 0, 0, 0], [0, 1, 1, 0, 1, 1, 0], [0, 0, 1, 1, 0, 1, 1]],
            dtype=int,
        )

        # Edges corresponding to the Tanner graph of Hamming code
        data_supports_ham = [[0, 1, 2, 3], [1, 2, 4, 5], [2, 3, 5, 6]]
        datas_ham = list(range(7))
        checks_ham = list(range(7, 10))

        self.nodes_hamming = [((i,), {"label": "check"}) for i in checks_ham] + [
            ((d,), {"label": "data"}) for d in datas_ham
        ]
        self.edges_hamming = [
            ((c,), (d,))
            for c, supp in zip(checks_ham, data_supports_ham, strict=True)
            for d in supp
        ]

        # Tanner Graph for Hamming code
        self.T_hamming = nx.Graph()
        self.T_hamming.add_nodes_from(self.nodes_hamming)
        self.T_hamming.add_edges_from(self.edges_hamming)

        # Parity-check matrix for full quantum code
        self.H_steane = np.vstack(
            (
                np.hstack((self.H_hamming, np.zeros(self.H_hamming.shape, dtype=int))),
                np.hstack((np.zeros(self.H_hamming.shape, dtype=int), self.H_hamming)),
            )
        )

        # Steane code Tanner graph
        self.T_steane = nx.Graph()

        # Add nodes
        self.x_nodes_steane = [((i,), {"label": "X"}) for i in checks_ham]
        self.z_nodes_steane = [((i + 3,), {"label": "Z"}) for i in checks_ham]
        self.data_nodes_steane = [((i,), {"label": "data"}) for i in datas_ham]
        self.T_steane.add_nodes_from(self.x_nodes_steane)
        self.T_steane.add_nodes_from(self.z_nodes_steane)
        self.T_steane.add_nodes_from(self.data_nodes_steane)

        # Add edges
        self.x_edges_steane = self.edges_hamming
        self.z_edges_steane = [((c[0] + 3,), d) for c, d in self.edges_hamming]
        self.T_steane.add_edges_from(self.x_edges_steane + self.z_edges_steane)

        ### SHOR CODE VARIABLES

        # Parity-check matrices for the Shor code
        self.Hx_shor = np.array(
            [[1, 1, 1, 1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1, 1, 1, 1]]
        )
        self.Hz_shor = np.array(
            [
                [1, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 1],
            ]
        )
        self.H_shor = np.vstack(
            (
                np.hstack((self.Hx_shor, np.zeros(self.Hx_shor.shape, dtype=int))),
                np.hstack((np.zeros(self.Hz_shor.shape, dtype=int), self.Hz_shor)),
            )
        )

        # Tanner graph components and full graph for Shor Code
        self.Tx_shor = nx.Graph()
        self.Tz_shor = nx.Graph()
        self.T_shor = nx.Graph()

        self.data_nodes_shor = [((i,), {"label": "data"}) for i in range(9)]
        self.x_nodes_shor = [((i,), {"label": "X"}) for i in range(9, 11)]
        self.z_nodes_shor = [((i,), {"label": "Z"}) for i in range(11, 17)]

        self.Tx_shor.add_nodes_from(self.x_nodes_shor)
        self.Tx_shor.add_nodes_from(self.data_nodes_shor)

        self.Tz_shor.add_nodes_from(self.z_nodes_shor)
        self.Tz_shor.add_nodes_from(self.data_nodes_shor)

        self.T_shor.add_nodes_from(self.x_nodes_shor)
        self.T_shor.add_nodes_from(self.z_nodes_shor)
        self.T_shor.add_nodes_from(self.data_nodes_shor)

        self.x_edges_shor = [
            ((i + 9,), (j + 3 * i,)) for i in range(2) for j in range(6)
        ]
        self.z_edges_shor = [
            ((i + 11,), (j + 3 * (i // 2) + (i % 2),))
            for i in range(6)
            for j in range(2)
        ]
        self.Tx_shor.add_edges_from(self.x_edges_shor)
        self.Tz_shor.add_edges_from(self.z_edges_shor)

        self.T_shor.add_edges_from(self.x_edges_shor + self.z_edges_shor)

        ### d=3 ROTATED SURFACE CODE VARIABLES

        # Parity-check matrix for the full quantum code
        self.H_rsc = np.array(
            [
                [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
            ]
        )

        # Tanner Graph
        self.T_rsc = nx.Graph()

        # Add notes
        self.data_nodes_rsc = [((i,), {"label": "data"}) for i in range(9)]
        self.x_nodes_rsc = [((i,), {"label": "X"}) for i in range(9, 13)]
        self.z_nodes_rsc = [((i,), {"label": "Z"}) for i in range(13, 17)]
        self.T_rsc.add_nodes_from(self.x_nodes_rsc)
        self.T_rsc.add_nodes_from(self.z_nodes_rsc)
        self.T_rsc.add_nodes_from(self.data_nodes_rsc)

        # Add edges
        x_data_supports_rsc = [[0, 3], [1, 2, 4, 5], [3, 4, 6, 7], [5, 8]]
        self.x_edges_rsc = [
            ((x,), (d,))
            for x, supp in zip(range(9, 13), x_data_supports_rsc, strict=True)
            for d in supp
        ]

        z_data_supports_rsc = [[1, 2], [0, 1, 3, 4], [4, 5, 7, 8], [6, 7]]
        self.z_edges_rsc = [
            ((z,), (d,))
            for z, supp in zip(range(13, 17), z_data_supports_rsc, strict=True)
            for d in supp
        ]

        self.T_rsc.add_edges_from(self.x_edges_rsc + self.z_edges_rsc)

        ### REPETITION CODE VARIABLES
        self.distance_rep = 21

        self.H_rep = np.eye(
            self.distance_rep - 1, self.distance_rep, dtype=int
        ) + np.eye(self.distance_rep - 1, self.distance_rep, k=1, dtype=int)

        # Nodes and edges  corresponding to a Tanner graph with a single component
        self.nodes_bitflip_rep = [
            ((i,), {"label": "data"}) for i in range(self.distance_rep)
        ] + [
            ((i,), {"label": "Z"})
            for i in range(self.distance_rep, 2 * self.distance_rep - 1)
        ]
        self.edges_bitflip_rep = [
            ((i + j,), (self.distance_rep + i,))
            for i in range(self.distance_rep - 1)
            for j in range(2)
        ]

        # Tanner Graph for bitflip repetition code
        self.T_bitflip_rep = nx.Graph()
        self.T_bitflip_rep.add_nodes_from(self.nodes_bitflip_rep)
        self.T_bitflip_rep.add_edges_from(self.edges_bitflip_rep)

        # [[5,1,3]] LAFLAMME CODE (aka 5q perfect code)
        self.H_laflamme = np.array(
            [
                [1, 0, 0, 1, 0, 0, 1, 1, 0, 0],  # XZZXI = (XIIXI)(IZZII)
                [0, 1, 0, 0, 1, 0, 0, 1, 1, 0],  # IXZZX = (IXIIX)(IIZZI)
                [1, 0, 1, 0, 0, 0, 0, 0, 1, 1],  # XIXZZ = (XIXII)(IIIZZ)
                [0, 1, 0, 1, 0, 1, 0, 0, 0, 1],  # ZXIXZ = (IXIXI)(ZIIIZ)
            ]
        )

        ### ERROR MESSAGES

        # Check for non-binary matrix inputs
        self.H_nb_err = np.array([[1, 2, 3], [4, 5, 6]])
        self.err_msg_H_nb = "Parity-check matrix contains non-binary elements."

        # Check for empty matrices
        self.H_zeros = np.array([[0, 0], [0, 0]])
        self.err_msg_H_empty = "Parity-check matrix is empty."

        # Check for erroneous input in quantum CSS codes Tanner graph
        self.T_err1 = nx.Graph()  # Non-allowed label
        self.T_err1.add_nodes_from([1, 2, 3], label="X")
        self.T_err1.add_nodes_from([4, 5, 6], label="entropica")
        self.T_err1.add_nodes_from([7, 8], label="Z")
        self.T_err1.add_nodes_from([9, 10], label="data")
        self.err1_msg = (
            "Tanner graph should describe a quantum CSS code and thus"
            " can only contain 'X', 'Z', and 'data' nodes."
        )

        self.T_err2 = nx.Graph()  # Missing data nodes
        self.T_err2.add_nodes_from([1, 2, 3], label="X")
        self.T_err2.add_nodes_from([4, 5, 6], label="Z")
        self.err2_msg = "Tanner graph does not contain any 'data' nodes."

        self.T_err3 = nx.Graph()  # Missing check nodes
        self.T_err3.add_nodes_from([1, 2, 3], label="data")
        self.err3_msg = "Tanner graph must contain 'X' and 'Z' check nodes."

        self.T_err4 = nx.Graph()  # Unlabelled edges
        labeled_edges = [
            (0, 1, {"cardinality": "E"}),
            (1, 2, {"cardinality": "N"}),
            (2, 3, {"cardinality": "S"}),
        ]
        unlabelled_edges = [(0, 5), (3, 4), (5, 4)]
        self.T_err4.add_edges_from(labeled_edges + unlabelled_edges)
        self.err4_msg = f"Edges {unlabelled_edges} do not contain input label."

    def test_find_maximum_matching(self):
        """Test the correct generation of a maximum matching of a bipartite graph."""

        # Create a simple connected bipartite graph
        G = nx.Graph()
        G.add_nodes_from([1, 2, 3], bipartite=0)  # Set 1
        G.add_nodes_from([4, 5, 6], bipartite=1)  # Set 2
        G.add_edges_from([(1, 4), (2, 5), (3, 6)])

        expected_matching = {1: 4, 4: 1, 2: 5, 5: 2, 3: 6, 6: 3}
        result = find_maximum_matching(G)
        self.assertEqual(result, expected_matching)

        # Create a disconnected bipartite graph
        G = nx.Graph()
        G.add_nodes_from([1, 2], bipartite=0)  # Set 1 of component 1
        G.add_nodes_from([3, 4], bipartite=1)  # Set 2 of component 1
        G.add_edges_from([(1, 3), (2, 4)])

        G.add_nodes_from([5, 6], bipartite=0)  # Set 1 of component 2
        G.add_nodes_from([7, 8], bipartite=1)  # Set 2 of component 2
        G.add_edges_from([(5, 7), (6, 8)])

        expected_matching = {1: 3, 3: 1, 2: 4, 4: 2, 5: 7, 7: 5, 6: 8, 8: 6}
        result = find_maximum_matching(G)
        self.assertEqual(result, expected_matching)

        # Error when non-bipartite graph is passed
        # Create a non-bipartite graph
        G = nx.Graph()
        G.add_nodes_from([1, 2, 3], bipartite=0)  # Set 1
        G.add_nodes_from([4, 5], bipartite=1)  # Set 2
        G.add_edges_from([(1, 4), (2, 4), (1, 5), (2, 5), (3, 5), (4, 5)])

        err_msg = "Graph is not bipartite."
        with self.assertRaises(ValueError) as cm:
            _ = find_maximum_matching(G)
        self.assertEqual(str(cm.exception), err_msg)

        # Error when the graph has a non-bipartite component
        # Create a graph with a non-bipartite component
        G = nx.Graph()
        G.add_nodes_from([1, 2, 3], bipartite=0)  # Set 1
        G.add_nodes_from([4, 5], bipartite=1)  # Set 2
        G.add_edges_from(
            [(1, 4), (2, 4), (1, 5), (2, 5), (3, 5)]
        )  # Bipartite component

        G.add_nodes_from([6, 7], bipartite=0)  # Set 1
        G.add_nodes_from([8, 9], bipartite=1)  # Set 2
        G.add_edges_from([(6, 8), (7, 8), (6, 9), (8, 9)])  # Non-bipartite component

        err_msg = "Graph is not bipartite."
        with self.assertRaises(ValueError) as cm:
            _ = find_maximum_matching(G)
        self.assertEqual(str(cm.exception), err_msg)

    def test_minimum_edge_coloring(self):
        """Test the correct generation of a minimum edge coloring of a graph.
        This tested by ensuring a proper coloring is produced, without enforcing a
        particular coloring scheme."""

        # EXAMPLE 1 - MEC for a generic bipartite graph
        G1 = nx.Graph()
        G1.add_edges_from([(1, 4), (1, 5), (2, 5), (2, 6)])

        # EXAMPLE 2 - MEC for a bipartite star graph
        G2 = nx.Graph()
        G2.add_edges_from([(1, 4), (1, 5), (1, 2), (1, 3)])

        # EXAMPLE 3 - MEC for a bipartite disconnected graph
        G3 = nx.Graph()
        G3.add_edges_from([(1, 4), (1, 5), (2, 5), (6, 7), (7, 8)])

        # EXAMPLE 4 - MEC for a bipartite graph with a perfect matching
        G4 = nx.Graph()
        G4.add_edges_from([(1, 4), (2, 5), (3, 6)])

        # TEST EXAMPLES
        G_list = [G1, G2, G3, G4]

        for G in G_list:
            # Compute the edge coloring
            coloring = minimum_edge_coloring(G)

            # Validate the number of colors used matches the maximum degree of the graph
            max_degree = max(dict(G.degree()).values())
            self.assertEqual(len(coloring), max_degree)

            # Ensure no two adjacent edges share the same color
            for edges in coloring.values():
                for u, v in edges:
                    for neighbor in G[u]:
                        if (u, neighbor) in edges or (neighbor, u) in edges:
                            self.assertEqual((u, v), (u, neighbor))

    def test_extract_subgraph_from_edge_labels(self):  # pylint: disable=too-many-locals
        """Test the extraction of a subgraph from a graph based on the edge labels."""

        # EXAMPLE 1 - Graph with random attribute labellings
        G = nx.Graph()
        edge_labels = ["Leonardo", "Donatello", "Michelangelo", "Raphael"]
        labelled_edges = [
            (0, i, {"ninja_turtle_name": edge_labels[i % 4]}) for i in range(12)
        ]

        G.add_edges_from(labelled_edges)

        # Extract subgraph based on edge labels
        subgraph_dict_rand = extract_subgraphs_from_edge_labels(
            G, label_attribute="ninja_turtle_name"
        )

        # Build correct subgraphs
        subgraph_list = [nx.Graph() for i in range(len(edge_labels))]
        _ = [
            subgraph_list[i].add_edges_from(
                [labelled_edges[4 * j + i] for j in range(3)]
            )
            for i in range(4)
        ]
        correct_subgraph_dict_rand = {
            label: subgraph_list[i] for i, label in enumerate(edge_labels)
        }

        # EXAMPLE 2 - Graph with cardinal labellings
        G = nx.Graph()
        edge_labels = ["E", "N", "S", "W"]
        labelled_edges = [
            (i, i + 1, {"cardinality": edge_labels[i % 4]}) for i in range(11)
        ] + [(0, 11, {"cardinality": "W"})]

        G.add_edges_from(labelled_edges)

        # Extract subgraph based on edge labels
        subgraph_dict_card = extract_subgraphs_from_edge_labels(G)

        # Build correct subgraphs
        subgraph_list = [nx.Graph() for i in range(len(edge_labels))]
        _ = [
            subgraph_list[i].add_edges_from(
                [labelled_edges[4 * j + i] for j in range(3)]
            )
            for i in range(4)
        ]
        correct_subgraph_dict_card = {
            label: subgraph_list[i] for i, label in enumerate(edge_labels)
        }

        ### VERIFY EXAMPLES
        subgraph_dict_list = [subgraph_dict_rand, subgraph_dict_card]
        correct_subgraph_dict_list = [
            correct_subgraph_dict_rand,
            correct_subgraph_dict_card,
        ]

        for subgraph_dict, correct_subgraph_dict in zip(
            subgraph_dict_list, correct_subgraph_dict_list, strict=True
        ):
            for l, g in subgraph_dict.items():
                self.assertEqual(
                    g.nodes(data=True), correct_subgraph_dict[l].nodes(data=True)
                )
                self.assertEqual(set(g.edges), set(correct_subgraph_dict[l].edges))
                self.assertEqual(
                    {v for _, _, attr in g.edges(data=True) for v in attr.values()},
                    set([l]),
                )

        ## ERROR MESSAGES

        # Check for invalid label attribute type
        G = nx.random_regular_graph(3, 4)
        wrong_label = 3
        err_msg = "Label attribute must be a string."
        with self.assertRaises(TypeError) as cm:
            _ = extract_subgraphs_from_edge_labels(G, label_attribute=wrong_label)
        self.assertEqual(str(cm.exception), err_msg)

        # Check for missing label attribute
        wrong_G = nx.Graph()
        wrong_G.add_edges_from([(0, i, {"singapore": i}) for i in range(3)])
        err_msg = "Edge attribute cardinality not present in graph."
        with self.assertRaises(ValueError) as cm:
            _ = extract_subgraphs_from_edge_labels(wrong_G)
        self.assertEqual(str(cm.exception), err_msg)

        # Check for unlabelled edges
        with self.assertRaises(ValueError) as cm:
            _ = extract_subgraphs_from_edge_labels(self.T_err4)
        self.assertEqual(str(cm.exception), self.err4_msg)

    def test_cardinality_distribution(self):
        """Test the correct computation of the cardinality distribution of a graph."""

        cardinalities = ["E", "N", "S", "W"]

        # EXAMPLE 1 - Uniform distribution
        G1 = nx.Graph()
        G1.add_edges_from(
            [
                (0, i, {"cardinality": cardinalities[i]})
                for i in range(len(cardinalities))
            ]
        )
        dist1 = {"E": [(0, 0)], "N": [(0, 1)], "S": [(0, 2)], "W": [(0, 3)]}

        # EXAMPLE 2 - Non-uniform distribution
        G2 = nx.Graph()
        G2.add_edges_from(
            [(0, i, {"cardinality": "N"}) for i in range(5)]
            + [(i, 5, {"cardinality": "E"}) for i in range(4)]
            + [(0, i, {"cardinality": "W"}) for i in range(6, 10)]
            + [(i, 10, {"cardinality": "S"}) for i in range(4, 6)]
        )
        dist2 = {
            "N": [(0, i) for i in range(5)],
            "E": [(i, 5) for i in range(4)],
            "S": [(i, 10) for i in range(4, 6)],
            "W": [(0, i) for i in range(6, 10)],
        }

        # EXAMPLE 3 - Incomplete set of cardinalities
        G3 = nx.Graph()
        G3.add_edges_from(
            [(2, i, {"cardinality": "N"}) for i in range(5)]
            + [(1, i, {"cardinality": "E"}) for i in range(2)]
        )
        dist3 = {"E": [(i, 1) for i in range(2)], "N": [(2, i) for i in range(5)]}

        ### VERIFY EXAMPLES
        G_list = [G1, G2, G3]
        dist_list = [dist1, dist2, dist3]

        for G, dist in zip(G_list, dist_list, strict=True):
            computed_dist = cardinality_distribution(G)
            self.assertEqual(computed_dist, dist)

        ## ERROR MESSAGES

        # Check for invalid label attribute name
        wrong_G1 = nx.Graph()
        wrong_G1.add_edges_from([(0, i, {"singapore": "S"}) for i in range(3)])

        err_msg1 = (
            f"Only allowed attribute name is 'cardinality', but input contains:"
            f" {set(['singapore'])}."
        )
        with self.assertRaises(ValueError) as cm:
            _ = cardinality_distribution(wrong_G1)
        self.assertEqual(str(cm.exception), err_msg1)

        # Check for invalid attributes
        wrong_G2 = nx.Graph()
        wrong_G2.add_edges_from([(1, i, {"cardinality": i}) for i in range(3)])

        err_msg2 = (
            f"Only cardinal values 'E','N','S','W' are allowed attributes, but"
            f" input contains invalid ones: {set(range(3))}."
        )
        with self.assertRaises(ValueError) as cm:
            _ = cardinality_distribution(wrong_G2)
        self.assertEqual(str(cm.exception), err_msg2)

        # Check for unlabelled edges
        with self.assertRaises(ValueError) as cm:
            _ = extract_subgraphs_from_edge_labels(self.T_err4)
        self.assertEqual(str(cm.exception), self.err4_msg)

    def test_cartesian_product_tanner_graphs(self):
        """Test the computation of the cartesian product of two Tanner graphs."""

        # EXAMPLE 1 - Toric code from a pair of repetition codes
        G_toric = nx.Graph()

        # Extract indices for data and check nodes in the repetition code
        data_nodes_rep = [
            n for n, info in self.nodes_bitflip_rep if info["label"] == "data"
        ]
        check_nodes_rep = [
            n for n, info in self.nodes_bitflip_rep if info["label"] != "data"
        ]
        nodes_rep = data_nodes_rep + check_nodes_rep
        len_x = len(nodes_rep)
        len_z = len(nodes_rep)

        # Combine checks from first code and datas from second to form X checks
        x_nodes_toric = [(i[0], j[0]) for i in check_nodes_rep for j in data_nodes_rep]

        # Combine datas from first code and checks from second to form Z checks
        z_nodes_toric = [(j[0], i[0]) for i in check_nodes_rep for j in data_nodes_rep]

        # Combine pairs of datas and pairs of checks (either X or Z) to form data nodes
        data_nodes_toric = [
            (i[0], j[0]) for i in check_nodes_rep for j in check_nodes_rep
        ] + [(i[0], j[0]) for i in data_nodes_rep for j in data_nodes_rep]

        # Add nodes
        G_toric.add_nodes_from(x_nodes_toric, label="X")
        G_toric.add_nodes_from(z_nodes_toric, label="Z")
        G_toric.add_nodes_from(data_nodes_toric, label="data")

        # Compute horizontal edges and their cardinality
        east_edges = []
        west_edges = []

        for j in nodes_rep:
            for i, k in self.edges_bitflip_rep:
                edge = ((i[0], j[0]), (k[0], j[0]))
                diff = (
                    k[0] - i[0]
                    if G_toric.nodes[edge[1]]["label"] == "data"
                    else i[0] - k[0]
                )
                if diff % len_x <= (len_x / 2):
                    east_edges.append(edge)
                else:
                    west_edges.append(edge)

        # Compute vertical edges and their cardinality
        north_edges = []
        south_edges = []
        for j in nodes_rep:
            for i, k in self.edges_bitflip_rep:
                edge = ((j[0], i[0]), (j[0], k[0]))
                diff = (
                    k[0] - i[0]
                    if G_toric.nodes[edge[1]]["label"] == "data"
                    else i[0] - k[0]
                )
                if diff % len_z <= (len_z / 2):
                    north_edges.append(edge)
                else:
                    south_edges.append(edge)

        # Add edges
        G_toric.add_edges_from(east_edges, cardinality="E")
        G_toric.add_edges_from(west_edges, cardinality="W")
        G_toric.add_edges_from(north_edges, cardinality="N")
        G_toric.add_edges_from(south_edges, cardinality="S")

        # Extract properties
        Tx_rep = ClassicalTannerGraph(self.T_bitflip_rep)
        Tz_rep = ClassicalTannerGraph(self.T_bitflip_rep)
        T_toric = TannerGraph(G_toric)

        # EXAMPLE 2 - Create code for two random made up Tanner graphs(/codes)
        G_prod_rand = nx.Graph()

        # Component to be transformed into X checks
        Gx_rand = nx.Graph()
        Gx_rand.add_nodes_from([0], label="data")
        Gx_rand.add_nodes_from([1], label="X")
        Gx_rand.add_edges_from([(0, 1)])
        len_x = len(Gx_rand.nodes())
        Tx_rand = ClassicalTannerGraph(Gx_rand)

        # Component to be transformed into Z checks
        Gz_rand = nx.Graph()
        Gz_rand.add_nodes_from([0, 2], label="Z")
        Gz_rand.add_nodes_from([1], label="data")
        Gz_rand.add_edges_from([(0, 1), (2, 1)])
        len_z = len(Gz_rand.nodes())
        Tz_rand = ClassicalTannerGraph(Gz_rand)

        # Generate nodes manually and add to graph
        x_nodes_rand = [(1, 0)]
        z_nodes_rand = [(0, 1), (0, 2)]
        data_nodes_rand = [(0, 0), (1, 1), (1, 2)]

        G_prod_rand.add_nodes_from(x_nodes_rand, label="X")
        G_prod_rand.add_nodes_from(z_nodes_rand, label="Z")
        G_prod_rand.add_nodes_from(data_nodes_rand, label="data")

        # Generate edges manually and add to graph
        # No west edges in this configuration!
        east_edges = [
            ((1, 0), (0, 0)),
            ((0, 1), (1, 1)),
            ((0, 2), (1, 2)),
        ]
        south_edges = [((0, 1), (0, 0)), ((1, 0), (1, 2))]
        north_edges = [((1, 0), (1, 1)), ((0, 2), (0, 0))]

        G_prod_rand.add_edges_from(east_edges, cardinality="E")
        G_prod_rand.add_edges_from(north_edges, cardinality="N")
        G_prod_rand.add_edges_from(south_edges, cardinality="S")

        # Extract properties
        T_prod_rand = TannerGraph(G_prod_rand)

        ### VERIFY EXAMPLES
        properties_list = [(Tx_rep, Tz_rep, T_toric), (Tx_rand, Tz_rand, T_prod_rand)]

        for Tx, Tz, T in properties_list:

            # Compute the cartesian product
            computed_T = cartesian_product_tanner_graphs(Tx, Tz)

            # Check Tanner structures are equal
            self.assertEqual(T, computed_T)

            # Check cardinalities are correct
            for edge in T.graph.edges():
                self.assertEqual(
                    T.graph.edges[edge]["cardinality"],
                    computed_T.graph.edges[edge]["cardinality"],
                )

        # Error message - invalid type
        Tx_shor = self.Tx_shor
        T_bitflip_rep = ClassicalTannerGraph(self.T_bitflip_rep)

        err_msg = "Both inputs must be of type ClassicalTannerGraph."

        for T, T_err in [(Tx_shor, T_bitflip_rep), (T_bitflip_rep, Tx_shor)]:
            with self.assertRaises(TypeError) as cm:
                _ = cartesian_product_tanner_graphs(T, T_err)
            self.assertEqual(str(cm.exception), err_msg)

    def test_verify_css_code_stabilizers(self):
        """Test the correct verification of whether Stabilizers represent a CSS code."""

        # EXAMPLE 1 - All X Stabiliziers
        stabilizers_all_x = [
            Stabilizer(
                pauli="XX", data_qubits=[(0, 0), (1, 0)], ancilla_qubits=[(0, 1)]
            ),
            Stabilizer(
                pauli="XX", data_qubits=[(1, 0), (2, 0)], ancilla_qubits=[(1, 1)]
            ),
        ]

        # EXAMPLE 2 - All Z Stabilizers
        stabilizers_all_z = (
            Stabilizer(
                pauli="ZZZZ",
                data_qubits=[(0, 0), (1, 0), (2, 0), (3, 0)],
                ancilla_qubits=[(0, 1)],
            ),
        )

        # EXAMPLE 3 - Mixed Stabilizers
        stabilizers_mixed = (
            Stabilizer(
                pauli="ZZ", data_qubits=[(0, 0), (1, 0)], ancilla_qubits=[(0, 1)]
            ),
            Stabilizer(
                pauli="XX", data_qubits=[(0, 0), (1, 0)], ancilla_qubits=[(1, 1)]
            ),
        )

        # EXAMPLE 4 - Non-CSS stabilizers
        stabilizers_non_css = [
            Stabilizer(
                pauli="XZ", data_qubits=[(0, 0), (1, 0)], ancilla_qubits=[(0, 1)]
            ),
            Stabilizer(
                pauli="ZXZ",
                data_qubits=[(1, 0), (2, 0), (3, 0)],
                ancilla_qubits=[(1, 1)],
            ),
        ]

        # EXAMPLE 5 - Non-CSS stabilizers
        stabilizers_non_css2 = [
            Stabilizer(
                pauli="YZ", data_qubits=[(0, 0), (1, 0)], ancilla_qubits=[(0, 1)]
            )
        ]

        # Verify examples
        stab_list = [
            stabilizers_all_x,
            stabilizers_all_z,
            stabilizers_mixed,
            stabilizers_non_css,
            stabilizers_non_css2,
        ]
        css_list = [True, True, True, False, False]

        for stabs, is_css in zip(stab_list, css_list, strict=True):
            self.assertEqual(verify_css_code_stabilizers(stabs), is_css)

        # Invalid inputs
        # ERROR 1 - Non tuple/list input
        invalid_input = [
            {
                "stab_1": Stabilizer(
                    pauli="ZZ", data_qubits=[(0, 0), (1, 0)], ancilla_qubits=[(0, 1)]
                ),
                "stab_2": Stabilizer(
                    pauli="XX", data_qubits=[(0, 0), (1, 0)], ancilla_qubits=[(0, 1)]
                ),
            },
            Stabilizer(
                pauli="ZZ", data_qubits=[(0, 0), (1, 0)], ancilla_qubits=[(0, 1)]
            ),
            None,
            "kiklian",
        ]
        err_msg = "Input must be a list or tuple."

        for invalid in invalid_input:
            with self.assertRaises(TypeError) as cm:
                _ = verify_css_code_stabilizers(invalid)
            self.assertEqual(str(cm.exception), err_msg)

        # ERROR 2 - Empty list/tuple
        for invalid in [[], ()]:
            err_msg = "No stabilizers provided."
            with self.assertRaises(ValueError) as cm:
                _ = verify_css_code_stabilizers(invalid)
            self.assertEqual(str(cm.exception), err_msg)

        # ERROR 3 - Non-Stabilizer elements
        invalid_inputs = [
            (
                Stabilizer(
                    pauli="ZZ", data_qubits=[(0, 0), (1, 0)], ancilla_qubits=[(0, 1)]
                ),
                "stab_2",
            ),
            [
                Stabilizer(
                    pauli="ZZ", data_qubits=[(0, 0), (1, 0)], ancilla_qubits=[(0, 1)]
                ),
                3,
            ],
            (
                Stabilizer(
                    pauli="ZZ", data_qubits=[(0, 0), (1, 0)], ancilla_qubits=[(0, 1)]
                ),
                None,
            ),
        ]

        err_msg = "Input must be a list or tuple of Stabilizer objects."
        for invalid in invalid_inputs:
            with self.assertRaises(TypeError) as cm:
                _ = verify_css_code_stabilizers([invalid])
            self.assertEqual(str(cm.exception), err_msg)

        # ERROR 4 - Non-commuting stabilizers
        invalid_inputs = [
            [
                Stabilizer(
                    pauli="XXX",
                    data_qubits=[(0, 0), (1, 0), (2, 0)],
                    ancilla_qubits=[(0, 1)],
                ),
                Stabilizer(pauli="Z", data_qubits=[(0, 0)], ancilla_qubits=[(1, 1)]),
                Stabilizer(
                    pauli="ZZ", data_qubits=[(0, 0), (1, 0)], ancilla_qubits=[(0, 1)]
                ),
            ],
            [
                Stabilizer(
                    pauli="X" * 5,
                    data_qubits=[(i, 0) for i in range(5)],
                    ancilla_qubits=[(0, 1)],
                ),
                Stabilizer(
                    pauli="Z" * 5,
                    data_qubits=[(i, 0) for i in range(5)],
                    ancilla_qubits=[(1, 1)],
                ),
            ],
        ]

        err_msgs = [
            (f"Input Stabilizers {stab_list[0]} and {stab_list[1]} do" " not commute.")
            for stab_list in invalid_inputs
        ]

        for invalid_input, err_msg in zip(invalid_inputs, err_msgs, strict=True):
            with self.assertRaises(ValueError) as cm:
                _ = verify_css_code_stabilizers(invalid_input)
            self.assertEqual(str(cm.exception), err_msg)

    ### CLASSICAL TANNER TESTS

    def test_classical_tanner_graph_wrong_input(self):
        """Test the handling of invalid inputs for the Classical Tanner Graph."""

        invalid_output = [{"a": 1, "b": 2}, 3, "Tanner", None, (True, False)]
        err_msg = (
            "A networkx.Graph, a tuple of Stabilizer or a "
            "ClassicalParityCheckMatrix must be provided."
        )

        for invalid in invalid_output:
            with self.assertRaises(TypeError) as cm:
                _ = ClassicalTannerGraph(invalid)
            self.assertEqual(str(cm.exception), err_msg)

    def test_classical_tanner_graph_from_graph(self):
        """Test the correct creation of ClassicalTannerGraph from a networkx Graph."""

        # EXAMPLE 1 - Hamming code Classical Tanner Graph
        G_hamming = self.T_hamming
        check_hamming = "check"

        # EXAMPLE 2 - Repetition code Tanner Graph
        G_rep = self.T_bitflip_rep
        check_rep = "Z"

        # Example 3 - X component of Shor Code
        G_shor_x = self.Tx_shor
        check_shor_x = "X"

        # Check all examples
        G_list = [G_hamming, G_rep, G_shor_x]
        check_types = [check_hamming, check_rep, check_shor_x]

        for check_type, G in zip(check_types, G_list, strict=True):

            # Compute graph
            T = ClassicalTannerGraph(G)

            # Check graph
            self.assertEqual(T.graph.nodes(data=True), G.nodes(data=True))
            self.assertEqual(T.graph.edges(), G.edges())

            # Check check_type
            self.assertEqual(T.check_type, check_type)

            # Check data_nodes and check_nodes attributes
            self.assertEqual(
                T.data_nodes, [n for n in G.nodes if G.nodes[n]["label"] == "data"]
            )
            self.assertEqual(
                T.check_nodes, [n for n in G.nodes if G.nodes[n]["label"] == check_type]
            )

        ### Invalid graph inputs

        # ERROR 0 - Empty graph input
        G_err0 = nx.Graph()
        err_msg0 = "Input graph is empty. Please provide a non-empty graph."

        with self.assertRaises(ValueError) as cm:
            _ = ClassicalTannerGraph(G_err0)
        self.assertEqual(str(cm.exception), err_msg0)

        # ERROR 1 - Wrong attributes in the nodes
        G_err1 = nx.Graph()
        nodes = [(i, {"label": "data"}) for i in range(3)] + [
            (i, {"lechat": "check"}) for i in range(5, 9)
        ]
        G_err1.add_nodes_from(nodes)
        err_msg1 = (
            "Missing node labels. All nodes should contain a 'label' "
            "attribute, with values 'data', 'X', 'Z' or 'check'."
        )

        with self.assertRaises(ValueError) as cm:
            _ = ClassicalTannerGraph(G_err1)
        self.assertEqual(str(cm.exception), err_msg1)

        # ERROR 2 - Wrong labels
        G_list_err2 = [nx.Graph() for i in range(6)]
        nodes_list = [
            [(i, {"label": "data"}) for i in range(3)]
            + [(i, {"label": "Kiklian"}) for i in range(3, 9)],
            [(i, {"label": "Kiklian"}) for i in range(3)]
            + [(i, {"label": "check"}) for i in range(3, 9)],
            [(i, {"label": "Panik"}) for i in range(3)]
            + [(i, {"label": "Z"}) for i in range(3, 9)],
            [(i, {"label": "quantum"}) for i in range(3)]
            + [(i, {"label": "X"}) for i in range(3, 9)],
            [(i, {"label": "ninja"}) for i in range(3)]
            + [(i, {"label": "turtle"}) for i in range(3, 9)],
            [(i, {"label": "data"}) for i in range(3)]
            + [(i, {"label": "X"}) for i in range(3, 6)]
            + [(i, {"label": "Z"}) for i in range(6, 9)],
        ]
        _ = [
            G.add_nodes_from(nodes)
            for G, nodes in zip(G_list_err2, nodes_list, strict=True)
        ]

        err_msg2 = (
            "Invalid node labels in the input graph. Must be 'data' for data"
            " nodes and only 'X', 'Z' or 'check' for check nodes."
        )

        for G in G_list_err2:
            with self.assertRaises(ValueError) as cm:
                _ = ClassicalTannerGraph(G)
            self.assertEqual(str(cm.exception), err_msg2)

        # ERROR 3 - Non-bipartite input graph
        G_err3 = nx.Graph()
        G_err3.add_nodes_from(
            [(i, {"label": "data"}) for i in range(1, 4)]
            + [(i, {"label": "check"}) for i in range(4, 6)]
        )
        G_err3.add_edges_from([(1, 4), (2, 4), (1, 5), (2, 5), (3, 5), (4, 5)])
        err_msg3 = "Graph is not bipartite."
        with self.assertRaises(ValueError) as cm:
            _ = ClassicalTannerGraph(G_err3)
        self.assertEqual(str(cm.exception), err_msg3)

        # ERROR 4 - One or both partitions contain a mixture of data and check nodes
        G_list_err4 = [nx.Graph() for i in range(3)]
        edges = [(1, 4), (2, 4), (1, 5), (2, 5), (3, 5)]
        nodes_list = [
            [(i, {"label": "data"}) for i in range(1, 3)]
            + [(i, {"label": "X"}) for i in range(3, 6)],
            [(i, {"label": "data"}) for i in range(1, 5)]
            + [(i, {"label": "check"}) for i in range(5, 6)],
            [(i, {"label": "data"}) for i in [1, 3, 5]]
            + [(i, {"label": "Z"}) for i in [2, 4]],
        ]
        _ = [
            G.add_nodes_from(nodes)
            for G, nodes in zip(G_list_err4, nodes_list, strict=True)
        ]
        _ = [G.add_edges_from(edges) for G in G_list_err4]

        err_msgs4 = [
            "Graph contains invalid edges between 'X' nodes.",
            "Graph contains invalid edges between 'data' nodes.",
            "Graph contains invalid edges between 'data' nodes and 'Z' nodes.",
        ]

        for G, err_msg4 in zip(G_list_err4, err_msgs4, strict=True):
            with self.assertRaises(ValueError) as cm:
                _ = ClassicalTannerGraph(G)
            self.assertEqual(str(cm.exception), err_msg4)

    def test_classical_tanner_from_stabilizers(self):
        """Test the creation of a classical Tanner graph from a set of stabilizers."""

        # EXAMPLE 1 - List of Stabilizers of X type
        stabilizers_x = (
            Stabilizer(
                pauli="XXX",
                data_qubits=[(0, 0, 0), (0, 1, 0), (1, 0, 0)],
                ancilla_qubits=[(0, 0, 1)],
            ),
            Stabilizer(
                pauli="XXX",
                data_qubits=[(0, 0, 0), (0, 2, 0), (2, 0, 0)],
                ancilla_qubits=[(0, 1, 1)],
            ),
            Stabilizer(
                pauli="XXX",
                data_qubits=[(0, 0, 0), (0, 3, 0), (3, 0, 0)],
                ancilla_qubits=[(1, 0, 1)],
            ),
        )

        correct_nodes_dict_x = {
            (0, 0, 0): {"label": "data"},
            (0, 1, 0): {"label": "data"},
            (1, 0, 0): {"label": "data"},
            (0, 2, 0): {"label": "data"},
            (2, 0, 0): {"label": "data"},
            (0, 3, 0): {"label": "data"},
            (3, 0, 0): {"label": "data"},
            (0, 0, 1): {"label": "X"},
            (0, 1, 1): {"label": "X"},
            (1, 0, 1): {"label": "X"},
        }

        correct_edges_x = [
            ((0, 0, 0), (0, 0, 1)),
            ((0, 0, 0), (0, 1, 1)),
            ((0, 0, 0), (1, 0, 1)),
            ((0, 1, 0), (0, 0, 1)),
            ((1, 0, 0), (0, 0, 1)),
            ((0, 2, 0), (0, 1, 1)),
            ((2, 0, 0), (0, 1, 1)),
            ((0, 3, 0), (1, 0, 1)),
            ((3, 0, 0), (1, 0, 1)),
        ]

        # EXAMPLE 2 - List of Stabilizers of Z type
        stabilizers_z = (
            Stabilizer(
                pauli="ZZ", data_qubits=[(0, 0), (1, 0)], ancilla_qubits=[(0, 1)]
            ),
            Stabilizer(
                pauli="ZZ", data_qubits=[(1, 0), (2, 0)], ancilla_qubits=[(1, 1)]
            ),
        )

        correct_nodes_dict_z = {
            (0, 0): {"label": "data"},
            (1, 0): {"label": "data"},
            (2, 0): {"label": "data"},
            (0, 1): {"label": "Z"},
            (1, 1): {"label": "Z"},
        }

        correct_edges_z = [
            ((0, 0), (0, 1)),
            ((1, 0), (0, 1)),
            ((1, 0), (1, 1)),
            ((2, 0), (1, 1)),
        ]

        # Verify examples

        stabilizers_list = [stabilizers_x, stabilizers_z]
        correct_nodes_dict_list = [correct_nodes_dict_x, correct_nodes_dict_z]
        correct_edges_list = [correct_edges_x, correct_edges_z]
        check_types = ["X", "Z"]

        for stabilizers, nodes_dict, edges, check in zip(
            stabilizers_list,
            correct_nodes_dict_list,
            correct_edges_list,
            check_types,
            strict=True,
        ):

            # Compute Classical Tanner graph
            T = ClassicalTannerGraph(stabilizers)

            # Check graph
            self.assertEqual(dict(T.graph.nodes(data=True)), nodes_dict)
            self.assertEqual(set(T.graph.edges), set(edges))

            # Check check_type
            self.assertEqual(T.check_type, check)

            # Check data_nodes and check_nodes attributes
            data_nodes = [n for n in nodes_dict if nodes_dict[n]["label"] == "data"]
            self.assertEqual(set(T.data_nodes), set(data_nodes))

            check_nodes = [n for n in nodes_dict if nodes_dict[n]["label"] == check]
            self.assertEqual(set(T.check_nodes), set(check_nodes))

        ### Invalid Stabilizer inputs

        # ERROR 1 - Stabilizer sets associated with Non-CSS codes
        stab_err1_list = [
            (
                Stabilizer(
                    pauli="XX", data_qubits=[(0, 0), (1, 0)], ancilla_qubits=[(0, 1)]
                ),
                Stabilizer(
                    pauli="ZZ", data_qubits=[(1, 0), (2, 0)], ancilla_qubits=[(1, 1)]
                ),
            ),
            (
                Stabilizer(
                    pauli="XZ", data_qubits=[(0, 0), (1, 0)], ancilla_qubits=[(0, 1)]
                ),
            ),
        ]

        err_msg1 = (
            "Input stabilizers must be of the same type to define a classical"
            " Tanner graph."
        )

        for stab_err1 in stab_err1_list:
            with self.assertRaises(ValueError) as cm:
                _ = ClassicalTannerGraph(stab_err1)
            self.assertEqual(str(cm.exception), err_msg1)

        # ERROR 2 - Stabilizers with more than one ancilla qubit associated
        stab_err2 = (
            Stabilizer(
                pauli="ZZ",
                data_qubits=[(0, 0), (1, 0)],
                ancilla_qubits=[(0, 1), (1, 1)],
            ),
            Stabilizer(
                pauli="ZZ", data_qubits=[(1, 0), (2, 0)], ancilla_qubits=[(2, 1)]
            ),
        )

        err_msg2 = "All Stabilizers must contain a single ancilla qubit."

        with self.assertRaises(ValueError) as cm:
            _ = ClassicalTannerGraph(stab_err2)
        self.assertEqual(str(cm.exception), err_msg2)

        # ERROR 3 - Stabilizers sharing the same ancilla qubits
        stab_err3 = (
            Stabilizer(
                pauli="XX", data_qubits=[(0, 0), (1, 0)], ancilla_qubits=[(0, 1)]
            ),
            Stabilizer(
                pauli="XX", data_qubits=[(1, 0), (2, 0)], ancilla_qubits=[(1, 1)]
            ),
            Stabilizer(
                pauli="XX", data_qubits=[(2, 0), (3, 0)], ancilla_qubits=[(1, 1)]
            ),
        )

        err_msg3 = "All ancilla qubits must be different."

        with self.assertRaises(ValueError) as cm:
            _ = ClassicalTannerGraph(stab_err3)
        self.assertEqual(str(cm.exception), err_msg3)

    def test_classical_tanner_to_stabilizer(self):
        """Test the conversion of a Classical Tanner graph to a list of stabilizers."""

        # EXAMPLE 1 - Hamming code with Z checks
        T_hamming = ClassicalTannerGraph(self.T_hamming)
        stabilizers_hamming = [
            Stabilizer(
                pauli="ZZZZ",
                data_qubits=[(i, 0) for i in [0, 1, 2, 3]],
                ancilla_qubits=[(7, 1)],
            ),
            Stabilizer(
                pauli="ZZZZ",
                data_qubits=[(i, 0) for i in [1, 2, 4, 5]],
                ancilla_qubits=[(8, 1)],
            ),
            Stabilizer(
                pauli="ZZZZ",
                data_qubits=[(i, 0) for i in [2, 3, 5, 6]],
                ancilla_qubits=[(9, 1)],
            ),
        ]

        # EXAMPLE 2 - Bitflip Repetition code
        T_rep = ClassicalTannerGraph(self.T_bitflip_rep)
        stabilizers_rep = [
            Stabilizer(
                pauli="Z" * 2,
                data_qubits=[(i, 0), (i + 1, 0)],
                ancilla_qubits=[(self.distance_rep + i, 1)],
            )
            for i in range(self.distance_rep - 1)
        ]

        # EXAMPLE 3 - X component Shor code overriding with pauli type Z
        T_shor = ClassicalTannerGraph(self.Tx_shor)
        stabilizers_shor = [
            Stabilizer(
                pauli="Z" * 6,
                data_qubits=[(i, 0) for i in range(6)],
                ancilla_qubits=[(9, 1)],
            ),
            Stabilizer(
                pauli="Z" * 6,
                data_qubits=[(i, 0) for i in range(3, 9)],
                ancilla_qubits=[(10, 1)],
            ),
        ]

        # Verify examples
        T_list = [T_hamming, T_rep, T_shor]
        stabilizers_list = [stabilizers_hamming, stabilizers_rep, stabilizers_shor]
        pauli_types = ["Z", None, "Z"]

        for pauli_type, T, stabilizers in zip(
            pauli_types, T_list, stabilizers_list, strict=True
        ):
            stabilizers_extracted = T.to_stabilizers(pauli_type=pauli_type)
            self.assertEqual(stabilizers_extracted, stabilizers)

        ### Error messages

        # Error 1 - check_nodes with "check" label and no pauli_type specified
        err_msg1 = (
            "Check nodes have not been assigned a pauli type. Please provide a"
            " pauli_type input."
        )

        with self.assertRaises(ValueError) as cm:
            _ = T_hamming.to_stabilizers()
        self.assertEqual(str(cm.exception), err_msg1)

        # Error 2 - Invalid pauli_type input
        err_msg2 = "Pauli type must be either 'X' or 'Z'."

        with self.assertRaises(ValueError) as cm:
            _ = T_hamming.to_stabilizers(pauli_type="M")
        self.assertEqual(str(cm.exception), err_msg2)

        # Error 3 - Nodes are not convertible into coordinates

        G_err3_list = [nx.Graph() for i in range(3)]
        nodes_list = [
            [((i, 0, 0), {"label": "data"}) for i in range(3)]
            + [((i, 1), {"label": "X"}) for i in range(3, 6)],
            [((i,), {"label": "data"}) for i in range(3)]
            + [((i + 0.02,), {"label": "Z"}) for i in range(3, 6)],
            [(i, {"label": "data"}) for i in range(3)]
            + [((i,), {"label": "Z"}) for i in range(3, 6)],
        ]
        edges_list = [
            [(n[i][0], n[j + 3][0]) for i in range(3) for j in range(3)]
            for n in nodes_list
        ]
        _ = [
            G.add_nodes_from(nodes)
            for G, nodes in zip(G_err3_list, nodes_list, strict=True)
        ]
        _ = [
            G.add_edges_from(edges)
            for G, edges in zip(G_err3_list, edges_list, strict=True)
        ]

        err_msg3 = (
            "Nodes are not tuples of equal size and cannot be "
            "converted to list of stabilizers. Consider using "
            "relabelled_graph() method, to generate a re-indexed graph and later"
            " convert to Stabilizer list."
        )

        for G_err3 in G_err3_list:
            T_err = ClassicalTannerGraph(G_err3)
            with self.assertRaises(ValueError) as cm:
                _ = T_err.to_stabilizers()
            self.assertEqual(str(cm.exception), err_msg3)

    def test_classical_tanner_relabel(self):
        """Test the correct creation of a relabelled Classical Tanner graph."""

        G_list = [nx.Graph() for i in range(2)]
        nodes_list = [
            [(i, {"label": "data"}) for i in range(3)]
            + [(i, {"label": "X"}) for i in range(3, 6)],
            [("data" + str(i), {"label": "data"}) for i in range(3)]
            + [("check" + str(i), {"label": "Z"}) for i in range(3, 6)],
        ]

        edges_list = [
            [(n[i][0], n[j + 3][0]) for i in range(3) for j in range(3)]
            for n in nodes_list
        ]
        _ = [G.add_nodes_from(n) for G, n in zip(G_list, nodes_list, strict=True)]
        _ = [G.add_edges_from(e) for G, e in zip(G_list, edges_list, strict=True)]

        correct_new_nodes_dict_list = [
            {(i,): {"label": "data", "original_node": i} for i in range(3)}
            | {(i,): {"label": "X", "original_node": i} for i in range(3, 6)},
            {
                (i,): {"label": "data", "original_node": "data" + str(i)}
                for i in range(3)
            }
            | {
                (i,): {"label": "Z", "original_node": "check" + str(i)}
                for i in range(3, 6)
            },
        ]

        correct_new_edges = [((i,), (j,)) for i in range(3) for j in range(3, 6)]
        correct_data_nodes = [(i,) for i in range(3)]
        correct_check_nodes = [(i,) for i in range(3, 6)]

        for G, correct_new_nodes in zip(
            G_list, correct_new_nodes_dict_list, strict=True
        ):
            T = ClassicalTannerGraph(G)
            T_relabelled = T.relabelled_graph()

            # Check the graph nodes and edges
            self.assertEqual(
                dict(T_relabelled.graph.nodes(data=True)), correct_new_nodes
            )
            self.assertEqual(set(T_relabelled.graph.edges()), set(correct_new_edges))

            # Check the check_type
            self.assertEqual(T_relabelled.check_type, T.check_type)

            # Check the data_nodes and check_nodes attributes
            self.assertEqual(T_relabelled.data_nodes, correct_data_nodes)
            self.assertEqual(T_relabelled.check_nodes, correct_check_nodes)

    def test_classical_tanner_set_check_type(self):
        """Test the correct assignment of a check type to a Classical Tanner graph."""

        # EXAMPLE 1 - Hamming code with Z checks
        T_hamming = ClassicalTannerGraph(self.T_hamming)
        assigned_type_ham = "X"

        # EXAMPLE 2 - Bitflip Repetition code
        T_rep = ClassicalTannerGraph(self.T_bitflip_rep)
        assigned_type_rep = "check"

        # EXAMPLE 3 - X component Shor code overriding with pauli type Z
        T_shor = ClassicalTannerGraph(self.Tx_shor)
        assigned_type_shor = "Z"

        # Verify examples
        T_list = [T_hamming, T_rep, T_shor]
        assigned_types_list = [assigned_type_ham, assigned_type_rep, assigned_type_shor]

        for assigned_type, T in zip(assigned_types_list, T_list, strict=True):

            # Check the check_type is correctly updated
            T.set_check_type(assigned_type)
            self.assertEqual(T.check_type, assigned_type)

            # Check that the check node labes are correctly updated
            new_labels = {T.graph.nodes[n]["label"] for n in T.check_nodes}
            self.assertEqual(new_labels, {assigned_type})

        # Error 1 - Invalid check_type input
        err_msg1 = "Check type must be either 'X', 'Z' or 'check'."

        with self.assertRaises(ValueError) as cm:
            T_hamming.set_check_type("L")
        self.assertEqual(str(cm.exception), err_msg1)

    def test_classical_tanner_graph_from_classical_parity_check_matrix(self):
        """Test the correct creation of a Classical Tanner Graph from a
        ClassicalParityCheckMatrix."""

        # EXAMPLE 1 - Hamming code
        correct_nodes_ham = self.nodes_hamming

        # EXAMPLE 2 - Repetition code
        correct_nodes_rep = self.nodes_bitflip_rep

        # EXAMPLE 3 - X component Shor code
        correct_nodes_shor = self.data_nodes_shor + self.x_nodes_shor

        # Verify examples
        H_list = [self.H_hamming, self.H_rep, self.Hx_shor]
        correct_nodes_list = [correct_nodes_ham, correct_nodes_rep, correct_nodes_shor]

        for H, nodes in zip(H_list, correct_nodes_list, strict=True):

            # Compute Classical Parity Check Matrix
            P = ClassicalParityCheckMatrix(H)
            edges = [
                (j, P.matrix.shape[1] + i)
                for i, j in zip(*np.nonzero(P.matrix), strict=True)
            ]

            # Compute Classical Tanner graph
            T = ClassicalTannerGraph(P)

            # Check graph nodes
            self.assertEqual(set(T.graph.nodes), set(n[0] for n in nodes))

            sorted_edges = sorted([sorted(e) for e in edges])
            sorted_tanner_edges = sorted([sorted(e) for e in T.graph.edges])
            self.assertEqual(sorted_tanner_edges, sorted_edges)

            # Check check_type
            self.assertEqual(T.check_type, "check")

            # Check data_nodes and check_nodes attributes
            data_nodes = [n[0] for n in nodes if n[1]["label"] == "data"]
            self.assertEqual(set(T.data_nodes), set(data_nodes))

            check_nodes = [n[0] for n in nodes if n[1]["label"] != "data"]
            self.assertEqual(set(T.check_nodes), set(check_nodes))

    def test_classical_tanner_graph_eq_method(self):
        """Test the equality method of Classical Tanner Graphs."""

        # EXAMPLE 1 - Hamming code Classical Tanner Graph
        T_hamming = ClassicalTannerGraph(self.T_hamming)

        # EXAMPLE 2 - Repetition code Classical Tanner Graph
        T_rep = ClassicalTannerGraph(self.T_bitflip_rep)

        # Example 3 - X component of Shor Code Classical Tanner Graph
        T_shor_x = ClassicalTannerGraph(self.Tx_shor)

        # Check equality
        self.assertEqual(T_hamming, T_hamming)
        self.assertNotEqual(T_hamming, T_rep)
        self.assertNotEqual(T_hamming, T_shor_x)

        # Check error messages for invalid inputs
        err_msg = "Comparison is only supported between ClassicalTannerGraph objects."
        with self.assertRaises(TypeError) as cm:
            _ = T_hamming == self.T_hamming
        self.assertEqual(str(cm.exception), err_msg)

    ### TANNER TESTS

    def test_tanner_graph_wrong_input(self):
        """Test the handling of invalid inputs for the Tanner Graph."""

        invalid_output = [{"a": 1, "b": 2}, 3, "Tanner", None, (True, False)]
        err_msg = (
            "A networkx.Graph, a tuple of Stabilizers or a ParityCheckMatrix "
            "must be provided."
        )

        for invalid in invalid_output:
            with self.assertRaises(TypeError) as cm:
                _ = TannerGraph(invalid)
            self.assertEqual(str(cm.exception), err_msg)

    def test_tanner_graph_from_graph(self):
        """Test the correct creation of TannerGraph from a networkx Graph."""

        # EXAMPLE 1 - Steane code Tanner Graph
        G_steane = self.T_steane

        # EXAMPLE 2 - Rotated surface code Tanner Graph
        G_rsc = self.T_rsc

        # Example 3 - X component of Shor Code
        G_shor = self.T_shor

        # Example 4 - Bitflip repetiion code
        G_bitflip_rep = self.T_bitflip_rep

        # Check all examples
        G_list = [G_steane, G_rsc, G_shor, G_bitflip_rep]

        for G in G_list:

            # Compute Tanner graph
            T = TannerGraph(G)

            # Check graph
            self.assertEqual(T.graph.nodes(data=True), G.nodes(data=True))
            self.assertEqual(T.graph.edges(), G.edges())

            # Check data_nodes and check nodes attributes
            self.assertEqual(
                T.data_nodes, [n for n in G.nodes if G.nodes[n]["label"] == "data"]
            )
            self.assertEqual(
                T.x_nodes, [n for n in G.nodes if G.nodes[n]["label"] == "X"]
            )

            self.assertEqual(
                T.z_nodes, [n for n in G.nodes if G.nodes[n]["label"] == "Z"]
            )

        ### Invalid graph inputs

        # ERROR 0 - Empty graph input
        G_err0 = nx.Graph()
        err_msg0 = "Input graph is empty. Please provide a non-empty graph."

        with self.assertRaises(ValueError) as cm:
            _ = TannerGraph(G_err0)
        self.assertEqual(str(cm.exception), err_msg0)

        # ERROR 1 - Wrong attributes in the nodes
        G_err1 = nx.Graph()
        nodes = (
            [(i, {"label": "data"}) for i in range(3)]
            + [(i, {"lechat": "X"}) for i in range(3, 5)]
            + [(i, {"label": "Z"}) for i in range(5, 9)]
        )

        G_err1.add_nodes_from(nodes)
        err_msg1 = (
            "Missing node labels. All nodes should contain a 'label' "
            "attribute, with values 'X', 'Z' or 'data'."
        )

        with self.assertRaises(ValueError) as cm:
            _ = TannerGraph(G_err1)
        self.assertEqual(str(cm.exception), err_msg1)

        # ERROR 2 - Wrong labels
        G_list_err2 = [nx.Graph() for i in range(4)]
        nodes_list = [
            [(i, {"label": "data"}) for i in range(3)]
            + [(i, {"label": "X"}) for i in range(3, 6)]
            + [(i, {"label": "Kiklian"}) for i in range(6, 9)],
            [(i, {"label": "data"}) for i in range(3)]
            + [(i, {"label": "Jullian"}) for i in range(3, 6)]
            + [(i, {"label": "Z"}) for i in range(6, 9)],
            [(i, {"label": "panik"}) for i in range(3)]
            + [(i, {"label": "X"}) for i in range(3, 6)]
            + [(i, {"label": "Z"}) for i in range(6, 9)],
            [(i, {"label": "quantum"}) for i in range(3)]
            + [(i, {"label": "entropica"}) for i in range(3, 6)]
            + [(i, {"label": "data"}) for i in range(6, 9)],
        ]
        _ = [G.add_nodes_from(n) for G, n in zip(G_list_err2, nodes_list, strict=True)]

        err_msg2 = (
            "Invalid node labels in the input graph. Must be 'X', 'Z', or 'data'."
        )

        for G in G_list_err2:
            with self.assertRaises(ValueError) as cm:
                _ = TannerGraph(G)
            self.assertEqual(str(cm.exception), err_msg2)

        # ERROR 3 - Non-bipartite input graph
        G_err3 = nx.Graph()
        G_err3.add_nodes_from(
            [(i, {"label": "data"}) for i in range(1, 4)]
            + [(i, {"label": "X"}) for i in range(4, 6)]
            + [(i, {"label": "Z"}) for i in range(6, 7)]
        )
        G_err3.add_edges_from([(1, 4), (2, 4), (1, 5), (2, 3), (3, 5), (3, 6)])
        err_msg3 = "Graph is not bipartite."
        with self.assertRaises(ValueError) as cm:
            _ = TannerGraph(G_err3)
        self.assertEqual(str(cm.exception), err_msg3)

        # ERROR 4 - One or both partitions contain a mixture of data and check nodes
        G_list_err4 = [nx.Graph() for i in range(3)]
        edges = [(1, 4), (2, 4), (1, 5), (2, 5), (3, 5), (3, 6)]
        nodes_list = [
            [(i, {"label": "data"}) for i in range(1, 3)]
            + [(i, {"label": "X"}) for i in range(3, 6)]
            + [(6, {"label": "Z"})],
            [(i, {"label": "data"}) for i in range(1, 5)]
            + [(i, {"label": "X"}) for i in range(3, 6)]
            + [(6, {"label": "Z"})],
            [(i, {"label": "data"}) for i in [1, 3, 6]]
            + [(i, {"label": "X"}) for i in [2, 5]]
            + [(4, {"label": "Z"})],
        ]
        _ = [G.add_nodes_from(n) for G, n in zip(G_list_err4, nodes_list, strict=True)]
        _ = [G.add_edges_from(edges) for G in G_list_err4]

        err_msg4 = "Graph contains invalid edges among data or check nodes."

        for G in G_list_err4:
            with self.assertRaises(ValueError) as cm:
                _ = TannerGraph(G)
            self.assertEqual(str(cm.exception), err_msg4)

        # ERROR 5 - Input graph associated with non-commuting checks
        G_list_err5 = [nx.Graph() for i in range(2)]
        edges_list = [
            [(1, 4), (2, 4), (1, 5), (2, 5), (3, 5), (3, 6)],
            [(1, 4), (2, 4), (1, 5), (2, 5), (3, 5), (1, 6), (2, 6), (3, 6)],
        ]

        nodes = (
            [(i, {"label": "data"}) for i in range(1, 4)]
            + [(i, {"label": "X"}) for i in range(4, 6)]
            + [(i, {"label": "Z"}) for i in range(6, 7)]
        )

        _ = [G.add_nodes_from(nodes) for G in G_list_err5]
        _ = [G.add_edges_from(e) for G, e in zip(G_list_err5, edges_list, strict=True)]

        err_msg5 = (
            "X and Z check nodes share an odd number of data qubits. "
            "This results in non-commuting stabilizers and "
            "Tanner graph does not represent a valid stabilizer code."
        )

        for G in G_list_err5:
            with self.assertRaises(ValueError) as cm:
                _ = TannerGraph(G)
            self.assertEqual(str(cm.exception), err_msg5)

    def test_tanner_graph_from_matrix(self):
        """Test the creation of a Tanner graph from a ParityCheckMatrix."""

        # EXAMPLE 1 - Steane code
        P_steane = ParityCheckMatrix(self.H_steane)
        T_steane = TannerGraph(P_steane)
        expected_T_steane = TannerGraph(self.T_steane)

        # Check the Tanner graph
        self.assertEqual(T_steane, expected_T_steane)

        # EXAMPLE 2 - Shor code
        P_shor = ParityCheckMatrix(self.H_shor)
        T_shor = TannerGraph(P_shor)
        expected_T_shor = TannerGraph(self.T_shor)

        # Check the Tanner graph
        self.assertEqual(T_shor, expected_T_shor)

        # Check for error message
        err_msg = "Parity-check matrix does not define a CSS code."
        P_laflamme = ParityCheckMatrix(self.H_laflamme)
        with self.assertRaises(ValueError) as cm:
            _ = TannerGraph(P_laflamme)
        self.assertEqual(str(cm.exception), err_msg)

    def test_tanner_graph_from_stabilizers(self):
        """Test the creation of a Tanner graph from Stabilizers."""

        # EXAMPLE 1 - Shor code
        x_stabs_shor = [
            Stabilizer(
                pauli="X" * 6,
                data_qubits=[(i, 0) for i in np.where(row == 1)[0]],
                ancilla_qubits=[(j + len(row), 1)],
            )
            for j, row in enumerate(self.Hx_shor)
        ]
        z_stabs_shor = [
            Stabilizer(
                pauli="Z" * 2,
                data_qubits=[(i, 0) for i in np.where(row == 1)[0]],
                ancilla_qubits=[(j + len(row) + len(x_stabs_shor), 1)],
            )
            for j, row in enumerate(self.Hz_shor)
        ]
        stabs_shor = tuple(x_stabs_shor + z_stabs_shor)
        correct_nodes_dict_shor = (
            {(n + (0,)): info for n, info in self.data_nodes_shor}
            | {(n + (1,)): info for n, info in self.x_nodes_shor}
            | {(n + (1,)): info for n, info in self.z_nodes_shor}
        )
        correct_edges_shor = [
            (e[1] + (0,), e[0] + (1,)) for e in self.x_edges_shor + self.z_edges_shor
        ]

        # EXAMPLE 2 - Steane code
        x_stabs_steane = [
            Stabilizer(
                pauli="X" * 4,
                data_qubits=[(i, 0) for i in np.where(row == 1)[0]],
                ancilla_qubits=[(j + len(row), 1)],
            )
            for j, row in enumerate(self.H_hamming)
        ]
        z_stabs_steane = [
            Stabilizer(
                pauli="Z" * 4,
                data_qubits=[(i, 0) for i in np.where(row == 1)[0]],
                ancilla_qubits=[(j + len(row) + len(x_stabs_steane), 1)],
            )
            for j, row in enumerate(self.H_hamming)
        ]
        stabs_steane = tuple(x_stabs_steane + z_stabs_steane)
        correct_nodes_dict_steane = (
            {(n + (0,)): info for n, info in self.data_nodes_steane}
            | {(n + (1,)): info for n, info in self.x_nodes_steane}
            | {(n + (1,)): info for n, info in self.z_nodes_steane}
        )
        correct_edges_steane = [
            (e[1] + (0,), e[0] + (1,))
            for e in self.x_edges_steane + self.z_edges_steane
        ]

        # Verify examples
        stabilizers_list = [stabs_shor, stabs_steane]
        correct_nodes_dict_list = [correct_nodes_dict_shor, correct_nodes_dict_steane]
        correct_edges_list = [correct_edges_shor, correct_edges_steane]

        for stabilizers, nodes_dict, edges in zip(
            stabilizers_list, correct_nodes_dict_list, correct_edges_list, strict=True
        ):

            # Compute Tanner graph
            T = TannerGraph(stabilizers)

            # Check graph
            self.assertEqual(dict(T.graph.nodes(data=True)), nodes_dict)
            self.assertEqual(set(T.graph.edges()), set(edges))

            # Check data_nodes and check nodes attributes
            data_nodes = [n for n in nodes_dict if nodes_dict[n]["label"] == "data"]
            self.assertEqual(set(T.data_nodes), set(data_nodes))

            x_nodes = [n for n in nodes_dict if nodes_dict[n]["label"] == "X"]
            self.assertEqual(set(T.x_nodes), set(x_nodes))

            z_nodes = [n for n in nodes_dict if nodes_dict[n]["label"] == "Z"]
            self.assertEqual(set(T.z_nodes), set(z_nodes))

        # Invalid inputs

        # ERROR 1 - Empty stabilizer tuple
        invalid_input = ()
        err_msg = "Input tuple of stabilizers is empty."
        with self.assertRaises(ValueError) as cm:
            _ = TannerGraph(invalid_input)
        self.assertEqual(str(cm.exception), err_msg)

        # ERROR 2 - Non-CSS code input
        noncss_stabs = (
            Stabilizer(
                pauli="XX", data_qubits=[(0, 0), (1, 0)], ancilla_qubits=[(0, 1)]
            ),
            Stabilizer(
                pauli="XZ", data_qubits=[(1, 0), (2, 0)], ancilla_qubits=[(1, 1)]
            ),
        )
        err_msg = "Input stabilizers do not define a CSS code."
        with self.assertRaises(ValueError) as cm:
            _ = TannerGraph(noncss_stabs)
        self.assertEqual(str(cm.exception), err_msg)

    def test_tanner_to_stabilizer(self):
        """Test the conversion from TannerGraph to list of Stabilizer"""

        # EXAMPLE 1 - Shor code
        T_shor = TannerGraph(self.T_shor)

        x_stabs_shor = [
            Stabilizer(
                pauli="X" * 6,
                data_qubits=[(i, 0) for i in np.where(row == 1)[0]],
                ancilla_qubits=[(j + len(row), 1)],
            )
            for j, row in enumerate(self.Hx_shor)
        ]
        z_stabs_shor = [
            Stabilizer(
                pauli="Z" * 2,
                data_qubits=[(i, 0) for i in np.where(row == 1)[0]],
                ancilla_qubits=[(j + len(row) + len(x_stabs_shor), 1)],
            )
            for j, row in enumerate(self.Hz_shor)
        ]
        stabs_shor = x_stabs_shor + z_stabs_shor

        # EXAMPLE 2 - Steane code
        T_steane = TannerGraph(self.T_steane)

        x_stabs_steane = [
            Stabilizer(
                pauli="X" * 4,
                data_qubits=[(i, 0) for i in np.where(row == 1)[0]],
                ancilla_qubits=[(j + len(row), 1)],
            )
            for j, row in enumerate(self.H_hamming)
        ]
        z_stabs_steane = [
            Stabilizer(
                pauli="Z" * 4,
                data_qubits=[(i, 0) for i in np.where(row == 1)[0]],
                ancilla_qubits=[(j + len(row) + len(x_stabs_steane), 1)],
            )
            for j, row in enumerate(self.H_hamming)
        ]
        stabs_steane = x_stabs_steane + z_stabs_steane

        # Verify examples
        T_list = [T_shor, T_steane]
        stabilizers_list = [stabs_shor, stabs_steane]

        for T, stabilizers in zip(T_list, stabilizers_list, strict=True):
            stabilizers_extracted = T.to_stabilizers()
            self.assertEqual(set(stabilizers_extracted), set(stabilizers))

        # Invalid input
        # ERROR 1 - Nodes are not convertible into coordinates
        G_err_list = [nx.Graph() for i in range(3)]
        nodes_list = [
            [((i, 0, 0), {"label": "data"}) for i in range(4)]
            + [((i, 1), {"label": "X"}) for i in range(4, 8)]
            + [((i, 0, 1), {"label": "Z"}) for i in range(8, 12)],
            [((i,), {"label": "data"}) for i in range(4)]
            + [((i + 0.02,), {"label": "X"}) for i in range(4, 8)]
            + [((i,), {"label": "Z"}) for i in range(8, 12)],
            [(i, {"label": "data"}) for i in range(4)]
            + [((i,), {"label": "X"}) for i in range(4, 8)]
            + [((i,), {"label": "Z"}) for i in range(8, 12)],
        ]
        edges_list = [
            [(n[i][0], n[j + 4][0]) for i in range(4) for j in range(8)]
            for n in nodes_list
        ]

        _ = [G.add_nodes_from(n) for G, n in zip(G_err_list, nodes_list, strict=True)]
        _ = [G.add_edges_from(e) for G, e in zip(G_err_list, edges_list, strict=True)]

        err_msg = (
            "Nodes are not tuples of equal size and cannot be "
            "converted to list of stabilizers. Consider using "
            "relabel_graph() method, to re-index your graph and later"
            " convert to Stabilizer list."
        )

        for G_err in G_err_list:
            T_err = TannerGraph(G_err)
            with self.assertRaises(ValueError) as cm:
                _ = T_err.to_stabilizers()
            self.assertEqual(str(cm.exception), err_msg)

    def test_tanner_relabel(self):
        """Test the relabelling of the Tanner Graph."""

        G_list = [nx.Graph() for _ in range(2)]
        nodes_list = [
            [(i, {"label": "data"}) for i in range(4)]
            + [(i, {"label": "X"}) for i in range(4, 8)]
            + [(i, {"label": "Z"}) for i in range(8, 12)],
            [("data" + str(i), {"label": "data"}) for i in range(4)]
            + [("x_check" + str(i), {"label": "X"}) for i in range(4, 8)]
            + [("z_check" + str(i), {"label": "Z"}) for i in range(8, 12)],
        ]

        edges_list = [
            [(n[i][0], n[j + 4][0]) for i in range(4) for j in range(8)]
            for n in nodes_list
        ]
        _ = [G.add_nodes_from(n) for G, n in zip(G_list, nodes_list, strict=True)]
        _ = [G.add_edges_from(e) for G, e in zip(G_list, edges_list, strict=True)]

        correct_new_nodes_dict_list = [
            {(i,): {"label": "data", "original_node": i} for i in range(4)}
            | {(i,): {"label": "X", "original_node": i} for i in range(4, 8)}
            | {(i,): {"label": "Z", "original_node": i} for i in range(8, 12)},
            {
                (i,): {"label": "data", "original_node": "data" + str(i)}
                for i in range(4)
            }
            | {
                (i,): {"label": "X", "original_node": "x_check" + str(i)}
                for i in range(4, 8)
            }
            | {
                (i,): {"label": "Z", "original_node": "z_check" + str(i)}
                for i in range(8, 12)
            },
        ]

        correct_new_edges = [((i,), (j,)) for i in range(4) for j in range(4, 12)]
        correct_data_nodes = [(i,) for i in range(4)]
        correct_x_check_nodes = [(i,) for i in range(4, 8)]
        correct_z_check_nodes = [(i,) for i in range(8, 12)]

        for G, correct_new_nodes in zip(
            G_list, correct_new_nodes_dict_list, strict=True
        ):

            T = TannerGraph(G)
            T_relabelled = T.relabelled_graph()

            # Check the graph nodes and edges
            self.assertEqual(
                dict(T_relabelled.graph.nodes(data=True)), correct_new_nodes
            )
            self.assertEqual(set(T_relabelled.graph.edges()), set(correct_new_edges))

            # Check the data_nodes and check_nodes attributes
            self.assertEqual(T_relabelled.data_nodes, correct_data_nodes)
            self.assertEqual(T_relabelled.x_nodes, correct_x_check_nodes)
            self.assertEqual(T_relabelled.z_nodes, correct_z_check_nodes)

    def test_tanner_get_components(self):
        """Test the correct creation of Tanner subcomponents."""

        # EXAMPLE 1 - Steane code Tanner Graph
        T_steane = TannerGraph(self.T_steane)
        edges_steane = (self.x_edges_steane, self.z_edges_steane)

        # EXAMPLE 2 - Shor code Tanner Graph
        T_shor = TannerGraph(self.T_shor)
        edges_shor = (self.x_edges_shor, self.z_edges_shor)

        # EXAMPLE 3 - d=3 Rotated Surface code Tanner Graph
        T_rsc = TannerGraph(self.T_rsc)
        edges_rsc = (self.x_edges_rsc, self.z_edges_rsc)

        # Loop over examples
        T_list = [T_steane, T_shor, T_rsc]
        edges_list = [edges_steane, edges_shor, edges_rsc]

        for T, edges in zip(T_list, edges_list, strict=True):
            x_edges, z_edges = edges

            # Extract the X and Z components
            Tx, Tz = T.get_components()

            # Check the X component
            self.assertEqual(set(Tx.graph.nodes), set(T.data_nodes + T.x_nodes))
            self.assertEqual(set(Tx.graph.edges), set(x_edges))

            # Check the Z component
            self.assertEqual(set(Tz.graph.nodes), set(T.data_nodes + T.z_nodes))
            self.assertEqual(set(Tz.graph.edges), set(z_edges))

        # Check for Tanner graphs with no X or Z components
        # Example 4 - X component of Shor code
        Tx_shor = TannerGraph(self.Tx_shor)
        Tx, Tz = Tx_shor.get_components()
        self.assertEqual(set(Tx.graph.nodes), set(Tx_shor.data_nodes + Tx_shor.x_nodes))
        self.assertEqual(set(Tx.graph.edges), set(self.x_edges_shor))
        self.assertEqual(Tz, None)

        # Example 5 - Bitflip repetition code
        T_rep = TannerGraph(self.T_bitflip_rep)
        Tx, Tz = T_rep.get_components()
        self.assertEqual(set(Tz.graph.nodes), set(T_rep.data_nodes + T_rep.z_nodes))
        self.assertEqual(set(Tz.graph.edges), set(self.edges_bitflip_rep))
        self.assertEqual(Tx, None)

    def test_tanner_graph_eq_method(self):
        """Test the equality method of Tanner Graphs."""

        # EXAMPLE 1 - Steane code Tanner Graph
        T_steane = TannerGraph(self.T_steane)

        # EXAMPLE 2 - Shor code Tanner Graph
        T_shor = TannerGraph(self.T_shor)

        # Check equality
        self.assertEqual(T_steane, T_steane)
        self.assertNotEqual(T_steane, T_shor)

        # Check error messages for invalid inputs
        err_msg = "Comparison is only supported between TannerGraph objects."
        with self.assertRaises(TypeError) as cm:
            _ = T_steane == self.T_steane
        self.assertEqual(str(cm.exception), err_msg)


if __name__ == "__main__":
    unittest.main()
