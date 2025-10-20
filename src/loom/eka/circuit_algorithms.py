"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

from __future__ import annotations

from .circuit import Channel, Circuit
from .stabilizer import Stabilizer
from .syndrome_circuit import SyndromeCircuit
from .tanner_graphs import TannerGraph
from .utilities.graph_matrix_utils import (
    minimum_edge_coloring,
    extract_subgraphs_from_edge_labels,
)


def coloration_circuit(
    t_graph: TannerGraph,
) -> dict[tuple[int, ...], list[tuple[int, ...]]]:
    """
    Computes the entangling layer order for a CSS code specified through a Tanner graph.
    This layering is obtained by the coloration circuit algorithm described in
    arXiv:2109.14609. In summary, the algorithm measures X and Z stabilizers in
    sequence, but optimizes the circuit depth for each type. This is done by extracting
    the X and Z subcomponents of the Tanner graph, and by obtaining a minimum edge
    coloring for each one. The sequence of entangling gates then corresponds to all the
    edges in each coloring layer, and then switching to the stabilizer type once the
    first component has been exhausted. Since the Tanner Graph is bipartite, this
    approach results in an upper bound of of max(deg(t_graph_x)) + max(deg(t_graph_z))
    for the total number of entangling layers, where t_graph_x and t_graph_z are the X
    and Z tanner subcomponents, respectively.

    The function returns a dictionary, where the keys correspond to each check nodes
    associated with a Stabilizer, and the values are the lists of data qubit channels
    in the precise order in which the entangling gates should be applied. Idling steps are
    added for Z(X) checks when the entangling gates between X(Z) ancillas and datas take
    place.

    Parameters
    ----------
    t_graph : TannerGraph
        Tanner graph associated with a CSS code.

    Returns
    -------
    check_to_data: dict[tuple[int,...],list[tuple[int,...]]]
        Dictionary mapping check nodes to lists of data nodes, properly ordered for
        entangling operations.
    """

    # Initialize list containing CNOT actions
    check_to_data = {node: [] for node in t_graph.x_nodes} | {
        node: [] for node in t_graph.z_nodes
    }

    # Extract subgraphs according to cardinality
    t_graph_x, t_graph_z = t_graph.get_components()

    # Loop over both subcomponents
    for t_card, t_card_opp in [(t_graph_x, t_graph_z), (t_graph_z, t_graph_x)]:

        # Obtain minimum edge coloring for the subgraph
        coloring = minimum_edge_coloring(t_card.graph)

        # Apply entangling gates for each color layer
        for _, colored_edges in coloring.items():

            # Store colored checks to identify idling ones later
            colored_checks = set()

            # Simultaneously apply all gates supported edges of t_card with same color
            for edge in colored_edges:
                if t_card.graph.nodes[edge[1]]["label"] == "data":
                    check_node, data_node = edge
                else:
                    check_node, data_node = edge[::-1]
                check_to_data[check_node].append(data_node)
                colored_checks.add(check_node)

            # Add idling step for qubits not participating in the layer
            # These include a subset of the local check type and those of opposite type
            idle_nodes = list(t_card_opp.check_nodes) + [
                n for n in t_card.check_nodes if n not in colored_checks
            ]
            for node in idle_nodes:
                check_to_data[node].append(())

    return check_to_data


# pylint: disable=too-many-locals
def cardinal_circuit(
    t_graph: TannerGraph,
) -> dict[tuple[int, int], list[tuple[int, int]]]:
    """
    Computes the entangling layer order for a HGP code specified through a Tanner graph
    with cardinal edge labelling. This layering is obtained by the cardinal circuit
    algorithm described in arXiv:2109.14609. Importantly, the ordering over which the
    cardinalities are examined is CRUCIAL for commutation relations. Here it has been
    fixed to ["E", "N", "S", "W"]. Only a subset of permutations from this order will
    ensure proper commutation of stabilizers.

    The function returns a dictionary, where the keys correspond to each check nodes
    associated with a Stabilizer, and the values are the lists of data qubit channels
    in the precise order in which the entangling gates should be applied. Idling steps are
    added when ancillas do not participate in a given time step.

    Parameters
    ----------
    t_graph : TannerGraph
        Tanner graph associated with the corresponding HGP code.

    Returns
    -------
    check_to_data: dict[tuple[int,int],list[tuple[int,int]]]
        Dictionary mapping check nodes to lists of data nodes, properly ordered for
        entangling operations.
    """

    # Check that input Tanner Graph contains proper edge labels
    attribute_names = {
        key for _, _, attr in t_graph.graph.edges(data=True) for key in attr
    }
    if "cardinality" not in attribute_names:
        raise ValueError("All edges most have a 'cardinality' attribute.")

    edge_labels = {label for _, _, label in t_graph.graph.edges(data="cardinality")}
    if not all(label in ["E", "N", "S", "W"] for label in edge_labels):
        raise ValueError("Cardinality labels should be either 'E', 'W', 'N' or 'S'.")

    # Initialize list containing CNOT actions
    check_to_data = {node: [] for node in t_graph.x_nodes} | {
        node: [] for node in t_graph.z_nodes
    }

    # Extract subgraphs according to cardinality
    t_sub_dict = extract_subgraphs_from_edge_labels(t_graph.graph, "cardinality")

    # Loop over cardinalities in specific valid order
    for cardinality in ["E", "N", "S", "W"]:

        # Extract subgraph associated with cardinality
        t_card = t_sub_dict[cardinality]

        # Obtain minimum edge coloring for the cardinal subgraph
        coloring = minimum_edge_coloring(t_card)

        # Apply entangling gates for each color layer
        for _, colored_edges in coloring.items():

            # Simultaneously apply all gates supported edges of t_card with same color
            for edge in colored_edges:
                if t_card.nodes[edge[1]]["label"] == "data":
                    check_node, data_node = edge
                else:
                    check_node, data_node = edge[::-1]
                check_to_data[check_node].append(data_node)

            # Add idling step for qubits not participating in the layer
            active_nodes = set(node for edge in colored_edges for node in edge)
            idle_nodes = set(t_graph.x_nodes + t_graph.z_nodes) - active_nodes
            for node in idle_nodes:
                check_to_data[node].append(())

    return check_to_data


def generate_stabilizer_and_syndrome_circuits_from_algorithm(
    t_graph: TannerGraph,
    algorithm: str,
) -> tuple[list[Stabilizer], list[SyndromeCircuit]]:
    """
    Generates stabilizers and syndrome circuits for a Tanner graph t_graph using the
    a supported circuit algorithm such as the cardinal or the coloration algorithms.

    Parameters
    ----------
    t_graph : TannerGraph
        Tanner graph associated with the code.

    algorithm : str
        Algorithm to be used to find the entangling schedule.

    Returns
    -------
    tuple[list[Stabilizer], list[SyndromeCircuit]]
        A list of Stabilizer objects describing the code and alist of associated
        SyndromeCircuit objects.
    """

    match algorithm:
        case "cardinal":
            entangling_schedule = cardinal_circuit(t_graph)
        case "coloration":
            entangling_schedule = coloration_circuit(t_graph)
        case _:
            raise NotImplementedError(
                f"The scheduling algorithm {algorithm} is not implemented."
                " Must be 'cardinal' or 'coloration."
            )

    # Leverage a circuit algorithm schedule to generate stabilizers and circuits
    syndrome_circuits = []
    stabilizers = []

    # Loop over check nodes of the Tanner graph
    for node in t_graph.x_nodes + t_graph.z_nodes:
        pauli = t_graph.graph.nodes[node]["label"] * t_graph.graph.degree(node)

        # Compute stabilizer associated with check node
        stabilizer = Stabilizer(
            pauli=pauli,
            data_qubits=[q for q in entangling_schedule[node] if q],
            ancilla_qubits=[node],
        )
        stabilizers.append(stabilizer)

        # Compute syndrome circuit associated with check node
        syndrome_circuit = extract_syndrome_circuit(pauli, entangling_schedule[node])
        syndrome_circuits.append(syndrome_circuit)

    return stabilizers, syndrome_circuits


# pylint: disable=too-many-locals
def extract_syndrome_circuit(
    pauli: str, entangling_order: list[tuple[int, ...]]
) -> SyndromeCircuit:
    """
    Generate the syndrome extraction circuit for a given stabilizer.

    Parameters
    ----------
    pauli : str
        The Pauli string defining the stabilizer.
    entangling_order : list[tuple[int,...]]
        The order in which data qubits are entangled with the ancilla qubit.

    Returns
    -------
    SyndromeCircuit
        The syndrome extraction circuit for the given stabilizer.
    """
    # Extract parameters
    pauli_type = next(iter(pauli))
    name = f"{pauli}_syndrome_extraction"
    weight = len(pauli)

    # Define channels
    data_channels = [Channel(type="quantum", label=f"d{i}") for i in range(weight)]
    cbit_channel = Channel(type="classical", label="c0")
    ancilla_channel = Channel(type="quantum", label="a0")

    # Define Hadamard gates
    hadamard1 = [Circuit("H", channels=[ancilla_channel])]
    hadamard2 = [Circuit("H", channels=[ancilla_channel])]

    # Define mapping between gate action and data qubit inside stabilizer

    entangle_ancilla = []
    q_counter = 0

    for qubit in entangling_order:
        if qubit:
            engtangling_gate = Circuit(
                f"C{pauli_type}",
                channels=[ancilla_channel, data_channels[q_counter]],
            )
            entangle_ancilla.append([engtangling_gate])
            q_counter += 1
        else:
            entangle_ancilla.append([])

    # Add ancilla measurement and reset
    measurement = [Circuit("Measurement", channels=[ancilla_channel, cbit_channel])]
    reset = [Circuit("Reset_0", channels=[ancilla_channel])]

    # Compose circuit operations as a list
    circuit_list = [reset, hadamard1] + entangle_ancilla + [hadamard2, measurement]

    # Build the syndrome extraction circuit
    syndrome_circuit = SyndromeCircuit(
        pauli=pauli_type * weight,
        name=name,
        circuit=Circuit(
            name=name,
            circuit=circuit_list,
            channels=data_channels + [ancilla_channel, cbit_channel],
        ),
    )

    return syndrome_circuit
