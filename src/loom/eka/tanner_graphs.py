"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

# pylint: disable=too-many-lines

from __future__ import annotations
from itertools import product
import numpy as np
import networkx as nx


from . import matrices  # pylint: disable=cyclic-import
from .stabilizer import Stabilizer


# pylint: disable=too-many-locals
def cartesian_product_tanner_graphs(
    t_graph_1: ClassicalTannerGraph, t_graph_2: ClassicalTannerGraph
) -> TannerGraph:
    """
    Computes the Cartesian product of two ClassicalTannerGraphs Gi(Vi,Ci,Ei), where Vi
    is the set of data nodes, Ci is the set of check nodes and Ei is the set of edges.
    Each ClassicalTannerGraph is associated with a classical code. See arXiv:2109.14609
    for a detailed description of the construction.

    The cartesian product yields a TannerGraph G(V,C,E) describing a quantum CSS code.
    The nodes of the product TannerGraph are labelled via two coordinates (v1,v2), each
    inherited from each classical code. The set of data nodes V is built out of the
    union of the nodes resulting from the product of the data nodes (v1,v2)  and
    of the check nodes (c1,c2) of each classical code. The set of check nodes C follows
    from the product of check and data nodes from each classical codes, i.e. (c1,v2) and
    (v1,c2). We assign X check nodes to be the first partition of check nodes and Z
    checks to be the second one. Edges are drawn between nodes (a1,a2) and (b1,b2), if
    either a2=b2 and G1 contains the edge (a1,b1) or a1=b1 and G2 contains the edge
    (a2,b2). The edges of product Tanner graphs are decorated with a cardinality label
    {`N`,`S`,`E`,`W`} according to the definitions introduced in arXiv:2109.14609.

    Note that we assign the first and second input ClassicalTannerGraphs to X and Z
    checks respectively, regardless of their associated `check_type` attribute.
    Furthermore, we impose a single integer labelling of the nodes, of the form `(i,)`
    in the input ClassicalTannerGraphs. If the any of the inputs does not satisfy this
    requirement, we build a relabelled graph using the `relabelled_graph()` method of
    the ClassicalTannerGraph class. While this constraint is fundamental from the
    cartesian product perspective, it is required in order to decorate the edges with
    cardinal labels.

    Parameters
    ----------
    t_graph_1: ClassicalTannerGraph
        Tanner graph of the first classical code.
    t_graph_2: ClassicalTannerGraph
        Tanner graph of the second classical code.

    Returns
    -------
    tanner_product: TannerGraph
       Tanner graph from cartesian product of t_graph_1 and t_graph_2, describing the full HGP code.
    """

    # Check inputs are of valid type
    if not all(
        isinstance(t_graph, ClassicalTannerGraph) for t_graph in [t_graph_1, t_graph_2]
    ):
        raise TypeError("Both inputs must be of type ClassicalTannerGraph.")

    # Ensure single integer tuple labelling of nodes
    # If not satisfied, generate relabelled graph
    if any(
        not isinstance(n, tuple) or (len(n) != 1) or (not isinstance(n[0], int))
        for n in t_graph_1.graph.nodes()
    ):
        t_graph_1 = t_graph_1.relabelled_graph()

    if any(
        not isinstance(n, tuple) or (len(n) != 1) or (not isinstance(n[0], int))
        for n in t_graph_2.graph.nodes()
    ):
        t_graph_2 = t_graph_2.relabelled_graph()

    # Initialize cartesian product graph
    graph_product = nx.Graph()

    # Extract size of node sets for both graphs
    len1 = len(t_graph_1.graph.nodes)
    len2 = len(t_graph_2.graph.nodes)

    # Add nodes to the product graph
    for t1_node, t2_node in product(
        t_graph_1.graph.nodes(data=True), t_graph_2.graph.nodes(data=True)
    ):
        # Nodes are indexed with a tuple, inherited from the indices in
        # the original codes
        node_pair = (t1_node[0][0], t2_node[0][0])
        label1 = t1_node[1]["label"]
        label2 = t2_node[1]["label"]

        # If both nodes are of the same type they are a data qubit
        if (label1 != "data" and label2 != "data") or (
            label1 == "data" and label2 == "data"
        ):
            graph_product.add_node(node_pair, label="data")

        # If nodes are different they correspond to a check node
        else:
            # We choose that if the nodes of the first code are checks, these
            # become X checks
            if label2 == "data":
                graph_product.add_node(node_pair, label="X")
            # If the nodes of the second code are checks, these become Z checks
            else:
                graph_product.add_node(node_pair, label="Z")

    # Add horizontal edges to the product graph, via edges inherited from the first code
    for (t1_node1, t1_node2), t2_node in product(
        t_graph_1.graph.edges(), t_graph_2.graph.nodes(data=False)
    ):
        # Define edges such that they follow the (check,data) order
        unordered_edge = ((t1_node1[0], t2_node[0]), (t1_node2[0], t2_node[0]))
        edge = (
            unordered_edge
            if graph_product.nodes[unordered_edge[1]]["label"] == "data"
            else unordered_edge[::-1]
        )

        # Compute their "label-distance"
        diff = edge[1][0] - edge[0][0]

        # Associate with E or W cardinality according to their label-distance
        is_east = diff % len1 <= len1 / 2
        cardinality = "E" if is_east else "W"

        # Add edge to the product
        graph_product.add_edge(*edge, cardinality=cardinality)

    # Add vertical edges to the product graph, via edges inherited from the second code
    for t1_node, (t2_node1, t2_node2) in product(
        t_graph_1.graph.nodes(data=False), t_graph_2.graph.edges()
    ):
        # Define edges such that they follow the (check,data) order
        unordered_edge = ((t1_node[0], t2_node1[0]), (t1_node[0], t2_node2[0]))
        edge = (
            unordered_edge
            if graph_product.nodes[unordered_edge[1]]["label"] == "data"
            else unordered_edge[::-1]
        )

        # Compute their "label-distance"
        diff = edge[1][1] - edge[0][1]

        # Associate with N or S cardinality according to their label-distance
        is_north = diff % len2 <= len2 / 2
        cardinality = "N" if is_north else "S"

        # Add edge to the product
        graph_product.add_edge(*edge, cardinality=cardinality)

    # Convert into TannerGraph object
    tanner_product = TannerGraph(graph_product)

    return tanner_product


def verify_css_code_stabilizers(
    stabilizers: list[Stabilizer] | tuple[Stabilizer],
) -> bool:
    """
    Verifies that the list of stabilizers define a valid CSS code.

    Parameters
    ----------
    stabilizers : list[Stabilizer] | tuple[Stabilizer]
        List of stabilizers defining a quantum CSS code.

    Returns
    -------
    valid: bool
        If True, the stabilizers define a valid CSS code.
    """

    # Check input
    if not isinstance(stabilizers, list) and not isinstance(stabilizers, tuple):
        raise TypeError("Input must be a list or tuple.")

    if len(stabilizers) == 0:
        raise ValueError("No stabilizers provided.")

    if not all(isinstance(stabilizer, Stabilizer) for stabilizer in stabilizers):
        raise TypeError("Input must be a list or tuple of Stabilizer objects.")

    # Verify all stabilizers are either of X or Z type
    valid = all(
        set(stabilizer.pauli) == {"X"} or set(stabilizer.pauli) == {"Z"}
        for stabilizer in stabilizers
    )

    # If all stabilizers are X and Z, verify that they all commute
    if valid:
        x_stabs = [stab for stab in stabilizers if set(stab.pauli) == {"X"}]
        z_stabs = [stab for stab in stabilizers if set(stab.pauli) == {"Z"}]
        for x_stab in x_stabs:
            for z_stab in z_stabs:
                if not x_stab.commutes_with(z_stab):
                    raise ValueError(
                        f"Input Stabilizers {x_stab} and {z_stab} do not commute."
                    )

    return valid


class ClassicalTannerGraph:
    """Classical Tanner Graph representation."""

    def __init__(
        self,
        input: nx.Graph | tuple[Stabilizer, ...] | matrices.ClassicalParityCheckMatrix,
        # pylint: disable=redefined-builtin
    ):
        """The Classical Tanner Graph class stores a bipartite graph representation of a
        classical error-correcting code. The graph is bipartite, with data nodes on one
        partition and check nodes on the other. The check nodes can be labelled with
        either 'X' or 'Z' if the code is to be related with quantum error properties,
        or it can be generically labelled as "check". If a networkx.Graph is given as
        input, the graph will go through a set of verifications to ensure its validity
        to represent the classical code. The input can also be a tuple of stabilizers,
        which need to be of the same time pauli type, and the graph will be generated
        from them. The graph can be converted into a list of stabilizers. The check type
        of the Tanner graph can be modified using the set_check_type method. Lastly, the
        class can also be instantiated from ClassicalParityCheckMatrix object, where the
        matrix is mapped to a Tanner graph and the check nodes are labeled as 'check'.

        Parameters
        ----------
        input : nx.Graph | tuple[Stabilizer,...] | ClassicalParityCheckMatrix
            Input graph, tuple of stabilizers, or ClassicalParityCheckMatrix to
            build the Classical Tanner graph.
        """
        self.check_type = None
        # Initialize graph based on input type
        if isinstance(input, nx.Graph):
            self.verify_input_graph(input)
            self.graph = input
        elif isinstance(input, tuple) and all(
            isinstance(item, Stabilizer) for item in input
        ):
            self.graph = self.generate_graph_from_stabilizers(input)
        elif isinstance(input, matrices.ClassicalParityCheckMatrix):
            self.graph = self.generate_graph_from_matrix(input)
        else:
            raise TypeError(
                "A networkx.Graph, a tuple of Stabilizer or a "
                "ClassicalParityCheckMatrix must be provided."
            )

        # Assign data nodes
        self.data_nodes = [
            n for n, attr in self.graph.nodes(data=True) if attr["label"] == "data"
        ]

        # Assign check nodes
        self.check_nodes = [
            n
            for n, attr in self.graph.nodes(data=True)
            if attr["label"] == self.check_type
        ]

    def verify_input_graph(self, graph: nx.Graph) -> None:
        """Verify the input graph is a faithful representation of a Classical
        Tanner graph.

        Parameters
        ----------
        graph : nx.Graph
            Input graph to be verified.
        """

        # Check graph is not empty
        if len(graph) == 0:
            raise ValueError("Input graph is empty. Please provide a non-empty graph.")

        # Extract labelling of nodes
        node_attributes = [attr for _, attr in graph.nodes(data=True)]

        # Check for unlabelled nodes
        if not all("label" in attr for attr in node_attributes):
            raise ValueError(
                "Missing node labels. All nodes should contain a 'label' "
                "attribute, with values 'data', 'X', 'Z' or 'check'."
            )

        # Check for invalid labels
        labels = {attr["label"] for attr in node_attributes}

        # There should be only two labels "data" and "X","Z" or "check" for check nodes
        if labels not in [{"data", "X"}, {"data", "Z"}, {"data", "check"}]:
            raise ValueError(
                "Invalid node labels in the input graph. Must be 'data' for data nodes"
                " and only 'X', 'Z' or 'check' for check nodes."
            )

        # Set check type based on left over label
        self.check_type = next(iter(labels - {"data"}))

        # Verify input graph is bipartite properties by checking connected components
        for component in nx.connected_components(graph):
            # Extract subcomponent
            sub_g = graph.subgraph(component)

            if not nx.algorithms.bipartite.is_bipartite(sub_g):
                raise ValueError("Graph is not bipartite.")

            # Verify partitions are correctly labelled
            part1, part2 = nx.algorithms.bipartite.sets(sub_g)

            # Ensure each partition contains only one type of node
            part1_labels = {graph.nodes[node]["label"] for node in part1}
            part2_labels = {graph.nodes[node]["label"] for node in part2}

            # Raise error if one partition contains more than one type of node
            if (len(part1_labels) > 1 and len(part2_labels) == 1) or (
                len(part2_labels) > 1 and len(part1_labels) == 1
            ):
                inter = next(iter(part1_labels & part2_labels))
                raise ValueError(
                    f"Graph contains invalid edges between" f" '{inter}' nodes."
                )
            # Raise error if both partitions contain more than one type of node
            if len(part1_labels) > 1 and len(part2_labels) > 1:
                raise ValueError(
                    f"Graph contains invalid edges between 'data' nodes and"
                    f" '{self.check_type}' nodes."
                )

    def generate_graph_from_stabilizers(
        self, stabilizers: tuple[Stabilizer]
    ) -> nx.Graph:
        """Generate a Classical Tanner graph from a tuple of stabilizers.

        Parameters
        ----------
        stabilizers : tuple[Stabilizer]
            Tuple of stabilizers to generate the Classical Tanner graph. All must be of
            the same pauli type.

        Returns
        -------
        t_graph : nx.Graph
            Classical Tanner graph, as a networkx object generated from the input
            stabilizers.
        """

        # Extract Pauli type from Stabilizers
        pauli_type = set(p for stab in stabilizers for p in stab.pauli)

        # Verify the pauli type is unique
        if len(pauli_type) > 1:
            raise ValueError(
                "Input stabilizers must be of the same type to define a classical"
                " Tanner graph."
            )

        # Assign check type based on the unique pauli type
        self.check_type = next(iter(pauli_type))

        # Verify that there is only a single ancilla per Stabilizer
        if any(len(stab.ancilla_qubits) != 1 for stab in stabilizers):
            raise ValueError("All Stabilizers must contain a single ancilla qubit.")

        # Verify that all ancillas are different
        check_qubits = {q for stab in stabilizers for q in stab.ancilla_qubits}
        if len(check_qubits) != len(stabilizers):
            raise ValueError("All ancilla qubits must be different.")

        # Extract data_qubits
        data_qubits = {q for stab in stabilizers for q in stab.data_qubits}

        # Initialize graph and add nodes
        t_graph = nx.Graph()
        t_graph.add_nodes_from(data_qubits, label="data")
        t_graph.add_nodes_from(check_qubits, label=self.check_type)

        # Generate edges
        edges = [
            (q, stab.ancilla_qubits[0])
            for stab in stabilizers
            for q in stab.data_qubits
        ]
        t_graph.add_edges_from(edges)

        return t_graph

    def to_stabilizers(self, pauli_type: str = None) -> list[Stabilizer]:
        """Convert the Classical Tanner graph to a list of stabilizers. Input pauli_type
        is required if check nodes have not been assigned a check_type. If check nodes
        have been assigned a check_type, the input pauli_type will override the
        assignment.

        Parameters
        ----------
        pauli_type : str | None
            Pauli type of the stabilizers, either 'X' or 'Z'. If None, the check_type
            of the Tanner graph will be used. Default is None.

        Returns
        -------
        stabilizers : list[Stabilizer]
            List of stabilizers generated from the Classical Tanner graph.
        """

        # Raise error if pauli_type input is missing and nodes do not have a pauli label
        if pauli_type is None and self.check_type == "check":
            raise ValueError(
                "Check nodes have not been assigned a pauli type. Please provide a"
                " pauli_type input."
            )
        # If no pauli_type is given, use the check_type
        if pauli_type is None:
            pauli_type = self.check_type
        # Ensure input pauli_type is valid
        elif pauli_type is not None and pauli_type not in ["X", "Z"]:
            raise ValueError("Pauli type must be either 'X' or 'Z'.")

        # Verify that all nodes can be converted into coordinates
        all_nodes = list(self.graph.nodes)
        if not all(
            isinstance(t, tuple)
            and len(t) == len(all_nodes[0])
            and all(isinstance(e, int) for e in t)
            for t in all_nodes
        ):
            raise ValueError(
                "Nodes are not tuples of equal size and cannot be "
                "converted to list of stabilizers. Consider using "
                "relabelled_graph() method, to generate a re-indexed graph and later"
                " convert to Stabilizer list."
            )

        # Generate stabilizers by checking the neighbors of each check node
        def generate_stabilizer(check_node):
            data_nodes = list(self.graph.neighbors(check_node))
            return Stabilizer(
                pauli=pauli_type * len(data_nodes),
                data_qubits=[data_node + (0,) for data_node in data_nodes],
                ancilla_qubits=[check_node + (1,)],
            )

        stabilizers = [generate_stabilizer(node) for node in self.check_nodes]

        return stabilizers

    def relabelled_graph(self) -> ClassicalTannerGraph:
        """Relabel a Classical Tanner graph to identify nodes with tuples of
        integers which can later be converted into Stabilizers. The nodes are a
        assigned a single-element tuple with an integer index. The data nodes are counted
        first, followed by the check nodes. The old node labels are stored in the node
        attributes under the key 'original_node'.

        Returns
        -------
        relabelled_tanner : ClassicalTannerGraph
            Relabelled Classical Tanner graph.
        """

        # Initialize graph
        t_graph = nx.Graph()

        # Add nodes
        data_nodes_info = {
            (i,): {"label": "data", "original_node": n}
            for i, n in enumerate(self.data_nodes)
        }
        data_nodes_dict = {v["original_node"]: k for k, v in data_nodes_info.items()}

        check_nodes_info = {
            (i + len(data_nodes_info),): {"label": self.check_type, "original_node": n}
            for i, n in enumerate(self.check_nodes)
        }
        check_nodes_dict = {v["original_node"]: k for k, v in check_nodes_info.items()}

        t_graph.add_nodes_from(data_nodes_info.items())
        t_graph.add_nodes_from(check_nodes_info.items())

        # Add edges
        edges = [
            (
                (data_nodes_dict[u], check_nodes_dict[v])
                if self.graph.nodes[u]["label"] == "data"
                else (data_nodes_dict[v], check_nodes_dict[u])
            )
            for u, v in self.graph.edges()
        ]
        t_graph.add_edges_from(edges)

        # Construct new relabelled Classical Tanner graph
        relabelled_tanner = ClassicalTannerGraph(t_graph)

        return relabelled_tanner

    def set_check_type(self, new_check_type: str) -> None:
        """Modify the check type of the Tanner graph.

        Parameters
        ----------
        new_check_type : str
            New check type to be assigned to the Tanner graph. Must be either 'X', 'Z'
            or 'check'.
        """

        # Ensure correct input
        if new_check_type not in ["X", "Z", "check"]:
            raise ValueError("Check type must be either 'X', 'Z' or 'check'.")

        # Create a dictionary of updated labels
        updated_labels = {
            n: new_check_type
            for n, attr in self.graph.nodes(data=True)
            if attr["label"] == self.check_type
        }

        # Apply the updates to the graph
        nx.set_node_attributes(self.graph, updated_labels, "label")

        # Update the check type
        self.check_type = new_check_type

    def generate_graph_from_matrix(
        self, h_matrix: matrices.ClassicalParityCheckMatrix
    ) -> nx.Graph:
        """Generate Tanner graph from a ClassicalParityCheckMatrix object.

        Parameters
        ----------
        h_matrix : matrices.ClassicalParityCheckMatrix
            Parity-check matrix to generate the Tanner graph.

        Returns
        -------
        g : nx.Graph
            Classical Tanner graph, as a networkx object generated from the input
            parity-check matrix.
        """

        # Initialize graph
        g = nx.Graph()

        # Extract the dimensions of the parity-check matrix
        n_rows, n_cols = h_matrix.matrix.shape

        # Add data qubit nodes from number of columns
        data_nodes = [(i,) for i in range(n_cols)]
        g.add_nodes_from(data_nodes, label="data")

        # Add check nodes from number of rows and set check_type
        check_nodes = [(i,) for i in range(n_cols, n_cols + n_rows)]
        g.add_nodes_from(check_nodes, label="check")
        self.check_type = "check"

        # Extract the edges from the indices associated with non-vanishing terms
        for i, row in enumerate(h_matrix.matrix):
            for j in np.where(row == 1)[0]:
                g.add_edge(check_nodes[i], data_nodes[j])

        return g

    def __eq__(self, other: ClassicalTannerGraph) -> bool:
        """Check if two Classical Tanner graphs are equal. Two graphs are equal if they
        have the same nodes and edges, and the same check type.

        Parameters
        ----------
        other : ClassicalTannerGraph
            Another Classical Tanner graph to compare with.

        Returns
        -------
        bool
            True if the Classical Tanner graphs are equal, False otherwise.
        """

        if not isinstance(other, ClassicalTannerGraph):
            raise TypeError(
                "Comparison is only supported between ClassicalTannerGraph objects."
            )

        def are_node_attributes_not_the_same():
            return set(self.data_nodes) != set(other.data_nodes) or set(
                self.check_nodes
            ) != set(other.check_nodes)

        def are_check_types_not_the_same():
            return self.check_type != other.check_type

        def are_nodes_not_the_same():
            return self.graph.nodes(data=True) != other.graph.nodes(data=True)

        # Check edges by sorting the tuples to ensure order does not matter
        def are_edges_not_the_same():
            return set(tuple(sorted(edge)) for edge in self.graph.edges()) != set(
                tuple(sorted(edge)) for edge in other.graph.edges()
            )

        # Check class attributes and graph properties
        return not (
            are_node_attributes_not_the_same()
            or are_check_types_not_the_same()
            or are_nodes_not_the_same()
            or are_edges_not_the_same()
        )


class TannerGraph:
    """A class representing a Tanner graph."""

    def __init__(
        self,
        input: nx.Graph | tuple[Stabilizer] | matrices.ParityCheckMatrix,
        # pylint: disable=redefined-builtin
    ):
        """
        The TannerGraph class stores a bipartite graph representation of a quantum
        CSS code. The graph is bipartite, with data nodes on one partition and check
        nodes on the other. The check nodes can be labelled with either 'X' or 'Z',
        associated with the type of stabilizers they represent and type of errors they
        check for, e.g. Z check nodes check for X errors and vice versa. If a
        networkx.Graph is given as input, the graph will go through a set of
        verifications to ensure its validity to represent the code. The input can also
        be a tuple of stabilizers, which need to satisfy the CSS condition, i.e.
        all must be either X- or Z-type, and the graph will be generated from them.

        Parameters
        ----------
        input : nx.Graph | tuple[Stabilizer] | ParityCheckMatrix
            Input graph or tuple of stabilizers to build the Tanner graph.
        """

        if isinstance(input, nx.Graph):
            self.verify_input_graph(input)
            self.graph = input

        elif isinstance(input, tuple) and all(
            isinstance(item, Stabilizer) for item in input
        ):
            self.graph = self.generate_graph_from_stabilizers(input)

        elif isinstance(input, matrices.ParityCheckMatrix):
            self.graph = self.generate_graph_from_matrix(input)

        else:
            raise TypeError(
                "A networkx.Graph, a tuple of Stabilizers or a ParityCheckMatrix must "
                "be provided."
            )

        # Assign data nodes
        self.data_nodes = [
            n for n, attr in self.graph.nodes(data=True) if attr["label"] == "data"
        ]

        # Assign X check nodes
        self.x_nodes = [
            n for n, attr in self.graph.nodes(data=True) if attr["label"] == "X"
        ]

        # Assign Z check nodes
        self.z_nodes = [
            n for n, attr in self.graph.nodes(data=True) if attr["label"] == "Z"
        ]

    def verify_input_graph(self, graph: nx.Graph) -> None:
        """
        Verify the input graph is bipartite and that nodes are correctly labelled.

        Parameters
        ----------
        graph : nx.Graph
            Input graph to be verified.
        """

        # Check graph is not empty
        if len(graph) == 0:
            raise ValueError("Input graph is empty. Please provide a non-empty graph.")

        # Verify labelling of nodes
        node_attributes = [attr for _, attr in graph.nodes(data=True)]

        # Check for unlabelled nodes
        if not all("label" in attr for attr in node_attributes):
            raise ValueError(
                "Missing node labels. All nodes should contain a 'label' "
                "attribute, with values 'X', 'Z' or 'data'."
            )

        # Check for invalid labels
        labels = {attr["label"] for attr in node_attributes}
        if not labels.issubset({"X", "Z", "data"}):
            raise ValueError(
                "Invalid node labels in the input graph. Must be 'X', 'Z', or 'data'."
            )

        # Verify input graph is bipartite properties by checking connected components
        for component in nx.connected_components(graph):

            # Extract subcomponent
            sub_g = graph.subgraph(component)
            if not nx.algorithms.bipartite.is_bipartite(sub_g):
                raise ValueError("Graph is not bipartite.")

            # Verify partitions are correctly labelled
            part1, part2 = nx.algorithms.bipartite.sets(sub_g)

            # Ensure each partition contains only one type of node
            part1_labels = {sub_g.nodes[node].get("label") for node in part1}
            part2_labels = {sub_g.nodes[node].get("label") for node in part2}

            if part1_labels == {"data"} and part2_labels.issubset({"X", "Z"}):
                _, check_part = part1, part2

            elif part2_labels == {"data"} and part1_labels.issubset({"X", "Z"}):
                check_part, _ = part1, part2
            else:
                raise ValueError(
                    "Graph contains invalid edges among data or check nodes."
                )

            # Ensure that X ad Z nodes only share even number or data neighbors
            # This ensures that stabilizers commute
            neighbor_sets = {n: set(graph.neighbors(n)) for n in check_part}
            x_nodes = [
                n for n, info in graph.nodes(data=True) if info.get("label") == "X"
            ]
            z_nodes = [
                n for n, info in graph.nodes(data=True) if info.get("label") == "Z"
            ]

            if any(
                len(neighbor_sets[x_node] & neighbor_sets[z_node]) % 2
                for x_node, z_node in product(x_nodes, z_nodes)
            ):
                raise ValueError(
                    "X and Z check nodes share an odd number of data qubits. "
                    "This results in non-commuting stabilizers and "
                    "Tanner graph does not represent a valid stabilizer code."
                )

    def generate_graph_from_matrix(
        self, h_matrix: matrices.ParityCheckMatrix
    ) -> nx.Graph:
        """
        Generate Tanner graph from an input parity-check matrix. Note that, given our
        definition of TannerGraph, we only allow for ParityCheckMatrix objects
        describing CSS codes.

        Parameters
        ----------
        h_matrix : ParityCheckMatrix
            Parity-check matrix to generate the Tanner graph. Must be a CSS code.

        Returns
        -------
        g : nx.Graph
            Tanner graph describing the code, as a networkx graph.
        """

        # Verify input matrix is CSS
        if not h_matrix.is_css:
            raise ValueError("Parity-check matrix does not define a CSS code.")

        hx_matrix, hz_matrix = h_matrix.get_components()

        # Initialize graph
        g = nx.Graph()

        # Generate nodes
        data_nodes = [(i,) for i in range(h_matrix.n_datas)]
        x_nodes = [
            (i,) for i in range(h_matrix.n_datas, h_matrix.n_datas + h_matrix.n_xstabs)
        ]
        z_nodes = [
            (i,)
            for i in range(
                h_matrix.n_datas + h_matrix.n_xstabs,
                h_matrix.n_datas + h_matrix.n_stabs,
            )
        ]

        g.add_nodes_from(data_nodes, label="data")
        g.add_nodes_from(x_nodes, label="X")
        g.add_nodes_from(z_nodes, label="Z")

        # Generate edges
        for i, row in enumerate(hx_matrix.matrix):
            for j in np.where(row == 1)[0]:
                g.add_edge(data_nodes[j], x_nodes[i])

        for i, row in enumerate(hz_matrix.matrix):
            for j in np.where(row == 1)[0]:
                g.add_edge(data_nodes[j], z_nodes[i])

        return g

    def generate_graph_from_stabilizers(
        self, stabilizers: tuple[Stabilizer, ...]
    ) -> nx.Graph:
        """
        Generate Tanner graph from a tuple of stabilizers.

        Parameters
        ----------
        stabilizers : tuple[Stabilizer,...]
            Stabilizers to generate the Tanner graph.
        """

        # Ensure input is not empty
        if len(stabilizers) == 0:
            raise ValueError("Input tuple of stabilizers is empty.")

        # Verify list of stabilizers define a CSS code
        if not verify_css_code_stabilizers(stabilizers):
            raise ValueError("Input stabilizers do not define a CSS code.")

        # Extract properties of datas and checks
        data_qubits = {qubit for stab in stabilizers for qubit in stab.data_qubits}

        x_qubits = {
            qubit
            for stab in stabilizers
            for qubit in stab.ancilla_qubits
            if set(stab.pauli) == {"X"}
        }

        z_qubits = {
            qubit
            for stab in stabilizers
            for qubit in stab.ancilla_qubits
            if set(stab.pauli) == {"Z"}
        }

        # Initialize graph
        t_graph = nx.Graph()

        t_graph.add_nodes_from(data_qubits, label="data")
        t_graph.add_nodes_from(x_qubits, label="X")
        t_graph.add_nodes_from(z_qubits, label="Z")

        # Generate edges
        edges = [
            (q, stab.ancilla_qubits[0])
            for stab in stabilizers
            for q in stab.data_qubits
        ]
        t_graph.add_edges_from(edges)

        return t_graph

    def to_stabilizers(self) -> list[Stabilizer]:
        """
        Convert Tanner graph to a list of stabilizers.
        """

        # Verify that nodes can be converted into coordinates
        all_nodes = list(self.graph.nodes)

        if not all(
            isinstance(t, tuple)
            and len(t) == len(all_nodes[0])
            and all(isinstance(e, int) for e in t)
            for t in all_nodes
        ):
            raise ValueError(
                "Nodes are not tuples of equal size and cannot be "
                "converted to list of stabilizers. Consider using "
                "relabel_graph() method, to re-index your graph and later"
                " convert to Stabilizer list."
            )

        stabilizers = []

        for x_node in self.x_nodes:
            data_nodes = list(self.graph.neighbors(x_node))
            stabilizer = Stabilizer(
                pauli="X" * len(data_nodes),
                data_qubits=[data_node + (0,) for data_node in data_nodes],
                ancilla_qubits=[x_node + (1,)],
            )
            stabilizers.append(stabilizer)

        for z_node in self.z_nodes:
            data_nodes = list(self.graph.neighbors(z_node))
            stabilizer = Stabilizer(
                pauli="Z" * len(data_nodes),
                data_qubits=[data_node + (0,) for data_node in data_nodes],
                ancilla_qubits=[z_node + (1,)],
            )
            stabilizers.append(stabilizer)

        return stabilizers

    def relabelled_graph(self) -> TannerGraph:
        """
        Relabel a Tanner graph to identify nodes with tuples of integers which can
        later be converted into Stabilizers. The nodes are a assigned a single-element
        tuple with an integer index. The data nodes are counted first, followed by the
        X check nodes and the Z check nodes. The old node labels are stored in the node
        attributes under the key 'original_node'.

        The relabelled graph is guaranteed to allow for conversion into Stabilizers
        through the `to_stabilizers()` method.

        Returns
        -------
        relabelled_tanner : TannerGraph
            Relabelled Tanner graph.
        """

        # Initialize graph
        t_graph = nx.Graph()

        # Add nodes
        data_nodes_info = {
            (i,): {"label": "data", "original_node": n}
            for i, n in enumerate(self.data_nodes)
        }
        data_nodes_dict = {v["original_node"]: k for k, v in data_nodes_info.items()}

        x_nodes_info = {
            (i + len(data_nodes_info),): {"label": "X", "original_node": n}
            for i, n in enumerate(self.x_nodes)
        }
        x_nodes_dict = {v["original_node"]: k for k, v in x_nodes_info.items()}

        z_nodes_info = {
            (i + len(data_nodes_info) + len(x_nodes_info),): {
                "label": "Z",
                "original_node": n,
            }
            for i, n in enumerate(self.z_nodes)
        }
        z_nodes_dict = {v["original_node"]: k for k, v in z_nodes_info.items()}

        t_graph.add_nodes_from(data_nodes_info.items())
        t_graph.add_nodes_from(x_nodes_info.items())
        t_graph.add_nodes_from(z_nodes_info.items())

        # Add edges
        for u, v in self.graph.edges():
            if self.graph.nodes[u]["label"] == "data":
                u, v = v, u
            if self.graph.nodes[u]["label"] == "X":
                t_graph.add_edge(data_nodes_dict[v], x_nodes_dict[u])
            else:
                t_graph.add_edge(data_nodes_dict[v], z_nodes_dict[u])

        # Construct new relabelled Tanner graph
        relabelled_tanner = TannerGraph(t_graph)

        return relabelled_tanner

    def get_components(
        self,
    ) -> tuple[ClassicalTannerGraph | None, ClassicalTannerGraph | None]:
        """
        Compute the X and Z components of the Tanner graph, associated with
        X and Z stabilizers, respectively.

        If the code contains only one type of stabilizer, the component of missing type
        will return None. The two of them cannot be None simultaneously.

        Returns
        -------
        t_graph_x, t_graph_z: tuple[ClassicalTannerGraph | None, ClassicalTannerGraph | None]
            Tuple containing the X and Z tanner graphs induced by the data nodes and the
            respective set of check nodes.
        """
        # Create subgraphs induced by the selected nodes and convert
        # into Classical Tanner Graphs
        if len(self.x_nodes) == 0:
            z_component = self.graph.copy().subgraph(self.data_nodes + self.z_nodes)
            t_graph_z = ClassicalTannerGraph(z_component)
            t_graph_x = None

        elif len(self.z_nodes) == 0:
            x_component = self.graph.copy().subgraph(self.data_nodes + self.x_nodes)
            t_graph_x = ClassicalTannerGraph(x_component)
            t_graph_z = None

        else:
            x_component = self.graph.copy().subgraph(self.data_nodes + self.x_nodes)
            z_component = self.graph.copy().subgraph(self.data_nodes + self.z_nodes)
            t_graph_x = ClassicalTannerGraph(x_component)
            t_graph_z = ClassicalTannerGraph(z_component)

        return t_graph_x, t_graph_z

    def __eq__(self, other: TannerGraph) -> bool:
        """
        Check if two Tanner graphs are equal. Two Tanner graphs are equal if they have
        the same graphs and same lists of data, x and z nodes.

        Parameters
        ----------
        other : TannerGraph
            TannerGraph to compare with.

        Returns
        -------
        bool
            True if the two Tanner graphs are equal, False otherwise.
        """
        if not isinstance(other, TannerGraph):
            raise TypeError("Comparison is only supported between TannerGraph objects.")

        def are_node_attributes_not_the_same():
            return (
                set(self.data_nodes) != set(other.data_nodes)
                or set(self.x_nodes) != set(other.x_nodes)
                or set(self.z_nodes) != set(other.z_nodes)
            )

        def are_nodes_not_the_same():
            return self.graph.nodes(data=True) != other.graph.nodes(data=True)

        # Check edges by sorting the tuples to ensure order does not matter
        def are_edges_not_the_same():
            return set(tuple(sorted(edge)) for edge in self.graph.edges()) != set(
                tuple(sorted(edge)) for edge in other.graph.edges()
            )

        return not (
            are_node_attributes_not_the_same()
            or are_nodes_not_the_same()
            or are_edges_not_the_same()
        )
