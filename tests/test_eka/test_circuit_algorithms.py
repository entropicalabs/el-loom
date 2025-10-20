"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

import unittest
import networkx as nx

from loom.eka import (
    Circuit,
    Channel,
    Stabilizer,
    SyndromeCircuit,
    cardinal_circuit,
    coloration_circuit,
    generate_stabilizer_and_syndrome_circuits_from_algorithm,
    extract_syndrome_circuit,
    ClassicalTannerGraph,
    TannerGraph,
    cartesian_product_tanner_graphs,
)


# pylint: disable=invalid-name, too-many-instance-attributes, too-many-locals, too-many-statements, too-many-branches
class TestCircuitAlgorithms(unittest.TestCase):
    """Unit tests for the circuit algorithms in the eka module."""

    def setUp(self):

        self.distance_rep = 3
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

        self.T_bitflip_rep = nx.Graph()
        self.T_bitflip_rep.add_nodes_from(self.nodes_bitflip_rep)
        self.T_bitflip_rep.add_edges_from(self.edges_bitflip_rep)

        # Steane code variables
        self.T_steane = nx.Graph()

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

        self.x_nodes_steane = [((i,), {"label": "X"}) for i in checks_ham]
        self.z_nodes_steane = [((i + 3,), {"label": "Z"}) for i in checks_ham]
        self.data_nodes_steane = [((i,), {"label": "data"}) for i in datas_ham]
        self.T_steane.add_nodes_from(self.x_nodes_steane)
        self.T_steane.add_nodes_from(self.z_nodes_steane)
        self.T_steane.add_nodes_from(self.data_nodes_steane)

        self.x_edges_steane = self.edges_hamming
        self.z_edges_steane = [((c[0] + 3,), d) for c, d in self.edges_hamming]
        self.T_steane.add_edges_from(self.x_edges_steane + self.z_edges_steane)

        # Shor code variables
        self.T_shor = nx.Graph()

        self.data_nodes_shor = [((i,), {"label": "data"}) for i in range(9)]
        self.x_nodes_shor = [((i,), {"label": "X"}) for i in range(9, 11)]
        self.z_nodes_shor = [((i,), {"label": "Z"}) for i in range(11, 17)]

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

        self.T_shor.add_edges_from(self.x_edges_shor + self.z_edges_shor)

    def test_coloration_circuit(self):
        """Test the correct creation of a coloration circuit from a Tanner Graph"""

        # Example 1 - Steane Code
        correct_steane = {
            (7,): [(0,), (1,), (2,), (3,), (), (), (), ()],
            (8,): [(1,), (2,), (4,), (5,), (), (), (), ()],
            (9,): [(2,), (3,), (5,), (6,), (), (), (), ()],
            (10,): [(), (), (), (), (0,), (1,), (2,), (3,)],
            (11,): [(), (), (), (), (1,), (2,), (4,), (5,)],
            (12,): [(), (), (), (), (2,), (3,), (5,), (6,)],
        }

        T_steane = TannerGraph(self.T_steane)

        # Example 2 - Shor code
        correct_shor = {
            (9,): [(0,), (1,), (2,), (3,), (4,), (5,), (), ()],
            (10,): [(3,), (4,), (5,), (6,), (7,), (8,), (), ()],
            (11,): [(), (), (), (), (), (), (0,), (1,)],
            (12,): [(), (), (), (), (), (), (1,), (2,)],
            (13,): [(), (), (), (), (), (), (3,), (4,)],
            (14,): [(), (), (), (), (), (), (4,), (5,)],
            (15,): [(), (), (), (), (), (), (6,), (7,)],
            (16,): [(), (), (), (), (), (), (7,), (8,)],
        }

        T_shor = TannerGraph(self.T_shor)

        # Check correctness of algorithm
        for T, correct_entangle in zip(
            [T_steane, T_shor], [correct_steane, correct_shor], strict=True
        ):
            check_to_data_entangle = coloration_circuit(T)
            self.assertEqual(check_to_data_entangle, correct_entangle)

    def test_cardinal_circuit(self):  # pylint: disable=too-many-locals
        """Test the correct creation of a cardinal circuit from the Tanner Graph of
        a HGP code."""

        # Example 1 - Toric code
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

        # Compute horizontal edges and their cardinality - we use (check,data) ordering
        east_edges = []
        west_edges = []

        for j in nodes_rep:
            for i, k in self.edges_bitflip_rep:
                unordered_edge = ((i[0], j[0]), (k[0], j[0]))
                edge = (
                    unordered_edge
                    if G_toric.nodes[unordered_edge[0]]["label"] != "data"
                    else unordered_edge[::-1]
                )
                diff = k[0] - i[0]
                if diff % len_x <= (len_x / 2):
                    east_edges.append(edge)
                else:
                    west_edges.append(edge)

        # Compute vertical edges and their cardinality - we use (check,data) ordering
        north_edges = []
        south_edges = []
        for j in nodes_rep:
            for i, k in self.edges_bitflip_rep:
                unordered_edge = ((j[0], i[0]), (j[0], k[0]))
                edge = (
                    unordered_edge
                    if G_toric.nodes[unordered_edge[0]]["label"] != "data"
                    else unordered_edge[::-1]
                )
                diff = k[0] - i[0]
                if diff % len_z <= (len_z / 2):
                    north_edges.append(edge)
                else:
                    south_edges.append(edge)

        # Add edges
        G_toric.add_edges_from(east_edges, cardinality="E")
        G_toric.add_edges_from(west_edges, cardinality="W")
        G_toric.add_edges_from(north_edges, cardinality="N")
        G_toric.add_edges_from(south_edges, cardinality="S")

        # Compute Tanner Graph
        T_toric = TannerGraph(G_toric)

        # Compute the correct check to data mapping
        check_nodes = T_toric.x_nodes + T_toric.z_nodes
        correct_check_to_data = {c: [] for c in check_nodes}

        # For each set of edges, in the correct order for commutation, add their
        # connectivity to data nodes, and at each stage add an empty step for the nodes
        # that remain idle while a cardinality layer is applied.
        # NOTE: This approach relies on the fact that, for the toric code, each check
        # node contains at most a single edge of a given cardinality.
        for cardinal_edges in [east_edges, north_edges, south_edges, west_edges]:

            # Get the check nodes participating in the cardinal layer
            cardinal_checks = [edge[0] for edge in cardinal_edges]

            # Extract idling nodes
            idle_nodes = list(set(check_nodes) - set(cardinal_checks))

            # Add connectivity between check nodes and data nodes
            for edge in cardinal_edges:
                check_node, data_node = edge
                correct_check_to_data[check_node].append(data_node)
            # Add empty steps for idle nodes
            for idle_check in idle_nodes:
                correct_check_to_data[idle_check].append(())

        # Call the cardinal_circuit function
        check_to_data = cardinal_circuit(T_toric)

        # Check all check nodes are present
        self.assertEqual(set(check_to_data.keys()), set(correct_check_to_data.keys()))

        # Ensure connectivity is correct
        for check_node in check_nodes:
            self.assertEqual(
                set(check_to_data[check_node]),
                set(correct_check_to_data[check_node]),
            )

        # Example 2 - Random code
        G_random = nx.Graph()

        # Code follows from HGP of a code with three data qubits connected to a single
        # check node, and a single data qubit connected to a single check node.
        x_nodes = [(3, 0)]
        z_nodes = [(0, 1), (1, 1), (2, 1)]
        data_nodes = [(0, 0), (1, 0), (2, 0), (3, 1)]

        # Add nodes
        G_random.add_nodes_from(x_nodes, label="X")
        G_random.add_nodes_from(z_nodes, label="Z")
        G_random.add_nodes_from(data_nodes, label="data")

        # Define and add edges according to their cardinality
        east_edges = [
            ((3, 0), (0, 0)),
            ((3, 0), (1, 0)),
            ((1, 1), (3, 1)),
            ((2, 1), (3, 1)),
        ]
        west_edges = [((3, 0), (2, 0)), ((0, 1), (3, 1))]
        north_edges = [((0, 0), (0, 1)), ((1, 0), (1, 1)), ((2, 0), (2, 1))]
        south_edges = [((3, 0), (3, 1))]

        G_random.add_edges_from(east_edges, cardinality="E")
        G_random.add_edges_from(west_edges, cardinality="W")
        G_random.add_edges_from(north_edges, cardinality="N")
        G_random.add_edges_from(south_edges, cardinality="S")

        # Compute Tanner Graph
        T_random = TannerGraph(G_random)

        # Compute the correct check to data mapping
        correct_check_to_data = {
            (0, 1): [(), (), (0, 0), (), (3, 1)],
            (1, 1): [(3, 1), (), (1, 0), (), ()],
            (2, 1): [(), (3, 1), (2, 0), (), ()],
            (3, 0): [(0, 0), (1, 0), (), (3, 1), (2, 0)],
        }

        # Call the cardinal_circuit function
        check_to_data = cardinal_circuit(T_random)

        # Check all check nodes are present
        self.assertEqual(set(check_to_data), set(correct_check_to_data))

        # Ensure connectivity is correct, pylint: disable=consider-using-dict-items
        for check_node in correct_check_to_data:
            self.assertEqual(
                set(check_to_data[check_node]),
                set(correct_check_to_data[check_node]),
            )

        ### Invalid inputs

        # ERROR 0 - Cardinality label attribute missing
        G = nx.Graph()
        nodes = [(0, {"label": "data"}), (1, {"label": "data"}), (2, {"label": "X"})]
        edges = [(0, 2, {"skibidi": "E"}), (1, 2, {"skibidi": "N"})]
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        T = TannerGraph(G)
        err_msg0 = "All edges most have a 'cardinality' attribute."

        with self.assertRaises(ValueError) as cm:
            check_to_data = cardinal_circuit(T)
        self.assertEqual(str(cm.exception), err_msg0)

        # ERROR 1 - Cardinality labels not in [E,N,S,W]
        G = nx.Graph()
        nodes = [(0, {"label": "data"}), (1, {"label": "data"}), (2, {"label": "X"})]
        edges = [(0, 2, {"cardinality": "E"}), (1, 2, {"cardinality": "Kiklian"})]
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        T = TannerGraph(G)
        err_msg1 = "Cardinality labels should be either 'E', 'W', 'N' or 'S'."

        with self.assertRaises(ValueError) as cm:
            check_to_data = cardinal_circuit(T)
        self.assertEqual(str(cm.exception), err_msg1)

    def test_extract_syndrome_circuit(self):
        """Test the extraction of syndrome circuits for a stabilizer."""

        # Example 1 - XX stabilizer
        xx_pauli = "XX"
        xx_name = xx_pauli + "_syndrome_extraction"
        xx_entangling_order = [(0, 0), (0, 2)]

        d_channels = [Channel(label=f"d{i}") for i in range(len(xx_entangling_order))]
        a_channels = [Channel(label=f"a{i}", type="quantum") for i in range(1)]
        c_channels = [Channel(label=f"c{i}", type="classical") for i in range(1)]
        xx_syndrome_circuit = SyndromeCircuit(
            pauli=xx_pauli,
            name=xx_name,
            circuit=Circuit(
                name=xx_name,
                circuit=[
                    [Circuit("Reset_0", channels=a_channels)],
                    [Circuit("H", channels=a_channels)],
                    [Circuit("CX", channels=[a_channels[0], d_channels[0]])],
                    [Circuit("CX", channels=[a_channels[0], d_channels[1]])],
                    [Circuit("H", channels=a_channels)],
                    [Circuit("Measurement", channels=a_channels + c_channels)],
                ],
            ),
        )

        # Example 2 - ZZZZ stabilizer
        z4_pauli = "ZZZZ"
        z4_name = z4_pauli + "_syndrome_extraction"
        z4_entangling_order = [(0, 0), (0, 2), (1, 1), (1, 2)]

        d_channels = [Channel(label=f"d{i}") for i in range(len(z4_entangling_order))]
        a_channels = [Channel(label=f"a{i}", type="quantum") for i in range(1)]
        c_channels = [Channel(label=f"c{i}", type="classical") for i in range(1)]
        z4_syndrome_circuit = SyndromeCircuit(
            pauli=z4_pauli,
            name=z4_name,
            circuit=Circuit(
                name=z4_name,
                circuit=[
                    [Circuit("Reset_0", channels=a_channels)],
                    [Circuit("H", channels=a_channels)],
                    [Circuit("CZ", channels=[a_channels[0], d_channels[0]])],
                    [Circuit("CZ", channels=[a_channels[0], d_channels[1]])],
                    [Circuit("CZ", channels=[a_channels[0], d_channels[2]])],
                    [Circuit("CZ", channels=[a_channels[0], d_channels[3]])],
                    [Circuit("H", channels=a_channels)],
                    [Circuit("Measurement", channels=a_channels + c_channels)],
                ],
            ),
        )

        paulis = [xx_pauli, z4_pauli]
        entangling_orders = [xx_entangling_order, z4_entangling_order]
        correct_circs = [xx_syndrome_circuit, z4_syndrome_circuit]

        # Verify syndrome circuits
        for pauli, order, correct_circ in zip(
            paulis, entangling_orders, correct_circs, strict=True
        ):
            syndrome_circuit = extract_syndrome_circuit(pauli, order)
            self.assertEqual(syndrome_circuit, correct_circ)

    def test_generate_stabilizer_and_syndrome_circuits_from_algorithm(self):
        """Test the generation of stabilizers and syndrome circuits using the coloration
        and cardinal circuit algorithm."""

        # EXAMPLE 1 - Unrotated Surface code as a HGP code using cardinal algorithm

        # Define the correct stabilizers
        x_surf_stabs = [
            Stabilizer(
                pauli="X" * 3,
                data_qubits=[(0, 0), (3, 3), (1, 0)],
                ancilla_qubits=[(3, 0)],
            ),
            Stabilizer(
                pauli="X" * 4,
                data_qubits=[(0, 1), (3, 3), (3, 4), (1, 1)],
                ancilla_qubits=[(3, 1)],
            ),
            Stabilizer(
                pauli="X" * 3,
                data_qubits=[(0, 2), (3, 4), (1, 2)],
                ancilla_qubits=[(3, 2)],
            ),
            Stabilizer(
                pauli="X" * 3,
                data_qubits=[(1, 0), (4, 3), (2, 0)],
                ancilla_qubits=[(4, 0)],
            ),
            Stabilizer(
                pauli="X" * 4,
                data_qubits=[(1, 1), (4, 3), (4, 4), (2, 1)],
                ancilla_qubits=[(4, 1)],
            ),
            Stabilizer(
                pauli="X" * 3,
                data_qubits=[(1, 2), (4, 4), (2, 2)],
                ancilla_qubits=[(4, 2)],
            ),
        ]

        z_surf_stabs = [
            Stabilizer(
                pauli="Z" * 3,
                data_qubits=[(0, 0), (0, 1), (3, 3)],
                ancilla_qubits=[(0, 3)],
            ),
            Stabilizer(
                pauli="Z" * 3,
                data_qubits=[(0, 1), (0, 2), (3, 4)],
                ancilla_qubits=[(0, 4)],
            ),
            Stabilizer(
                pauli="Z" * 4,
                data_qubits=[(3, 3), (1, 0), (1, 1), (4, 3)],
                ancilla_qubits=[(1, 3)],
            ),
            Stabilizer(
                pauli="Z" * 4,
                data_qubits=[(3, 4), (1, 1), (1, 2), (4, 4)],
                ancilla_qubits=[(1, 4)],
            ),
            Stabilizer(
                pauli="Z" * 3,
                data_qubits=[(4, 3), (2, 0), (2, 1)],
                ancilla_qubits=[(2, 3)],
            ),
            Stabilizer(
                pauli="Z" * 3,
                data_qubits=[(4, 4), (2, 1), (2, 2)],
                ancilla_qubits=[(2, 4)],
            ),
        ]

        correct_stabilizers = x_surf_stabs + z_surf_stabs

        # Define the expected syndrome schedules
        correct_syndrome_circuits = []
        d_channels = [Channel(label=f"d{i}") for i in range(4)]
        a_channels = [Channel(label=f"a{i}", type="quantum") for i in range(1)]
        c_channels = [Channel(label=f"c{i}", type="classical") for i in range(1)]

        reset = [Circuit("Reset_0", channels=a_channels)]
        init_hadamard = [Circuit("H", channels=a_channels)]
        final_hadamard = [Circuit("H", channels=a_channels)]
        measurement = [Circuit("Measurement", channels=a_channels + c_channels)]

        # Position for idling steps from cardinal algorithm for each ancilla qubit
        padding_indices = {
            (0, 3): 0,
            (0, 4): 0,
            (1, 3): None,
            (1, 4): None,
            (2, 3): 3,
            (2, 4): 3,
            (3, 0): 1,
            (3, 1): None,
            (3, 2): 2,
            (4, 0): 1,
            (4, 1): None,
            (4, 2): 2,
        }

        for stab in correct_stabilizers:

            entangling_gates = [
                [Circuit("C" + stab.pauli[0], channels=[a_channels[0], d_channels[i]])]
                for i in range(len(stab.data_qubits))
            ]
            if padding_indices[stab.ancilla_qubits[0]] is not None:
                entangling_gates.insert(padding_indices[stab.ancilla_qubits[0]], [])

            syndrome_circuit = SyndromeCircuit(
                pauli=stab.pauli,
                name=stab.pauli + "_syndrome_extraction",
                circuit=Circuit(
                    name=stab.pauli + "_syndrome_extraction",
                    circuit=[reset, init_hadamard]
                    + entangling_gates
                    + [final_hadamard, measurement],
                ),
            )
            correct_syndrome_circuits.append(syndrome_circuit)

        # Generate HGP stabs and schedule
        rep_tanner = ClassicalTannerGraph(self.T_bitflip_rep)
        hgp_tanner = cartesian_product_tanner_graphs(rep_tanner, rep_tanner)
        stabilizers, syndrome_circuits = (
            generate_stabilizer_and_syndrome_circuits_from_algorithm(
                hgp_tanner, algorithm="cardinal"
            )
        )

        # Check that the generated stabilizers match the expected ones
        self.assertEqual(set(stabilizers), set(correct_stabilizers))

        # Check that the generated syndrome circuits match the expected ones
        self.assertEqual(syndrome_circuits, correct_syndrome_circuits)

        # EXAMPLE 2 - Shor code
        x_stabilizers = [
            Stabilizer(
                pauli="X" * 6,
                data_qubits=[(j + 3 * i,) for j in range(6)],
                ancilla_qubits=[(i + 9,)],
            )
            for i in range(2)
        ]

        z_stabilizers = [
            Stabilizer(
                pauli="Z" * 2,
                data_qubits=[(j + 3 * (i // 2) + (i % 2),) for j in range(2)],
                ancilla_qubits=[(i + 11,)],
            )
            for i in range(6)
        ]

        correct_stabilizers = x_stabilizers + z_stabilizers

        # Define the expected syndrome schedules
        correct_syndrome_circuits = []
        d_channels = [Channel(label=f"d{i}") for i in range(9)]
        a_channels = [Channel(label=f"a{i}") for i in range(1)]
        c_channels = [Channel(label=f"c{i}", type="classical") for i in range(1)]

        reset = [Circuit("Reset_0", channels=a_channels)]
        init_hadamard = [Circuit("H", channels=a_channels)]
        final_hadamard = [Circuit("H", channels=a_channels)]
        measurement = [Circuit("Measurement", channels=a_channels + c_channels)]

        # Define the expected syndrome extraction circuits

        for stab in correct_stabilizers:

            entangling_gates = [
                [Circuit("C" + stab.pauli[0], channels=[a_channels[0], d_channels[i]])]
                for i in range(len(stab.data_qubits))
            ]
            if stab.pauli == "X" * 6:
                entangling_gates += [()] * 2  # Add idling steps for X stabilizers
            else:
                entangling_gates = [()] * 6 + entangling_gates

            syndrome_circuit = SyndromeCircuit(
                pauli=stab.pauli,
                name=stab.pauli + "_syndrome_extraction",
                circuit=Circuit(
                    name=stab.pauli + "_syndrome_extraction",
                    circuit=[reset, init_hadamard]
                    + entangling_gates
                    + [final_hadamard, measurement],
                ),
            )
            correct_syndrome_circuits.append(syndrome_circuit)

        # Generate stabilizers and syndrome circuits using the coloration algorithm
        shor_tanner = TannerGraph(self.T_shor)
        stabilizers, syndrome_circuits = (
            generate_stabilizer_and_syndrome_circuits_from_algorithm(
                shor_tanner, algorithm="coloration"
            )
        )

        # Check that the generated stabilizers and circuits match the expected ones
        self.assertEqual(set(stabilizers), set(correct_stabilizers))
        self.assertEqual(syndrome_circuits, correct_syndrome_circuits)


if __name__ == "__main__":
    unittest.main()
