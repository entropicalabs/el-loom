"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

import numpy as np
import networkx as nx


def binary_gaussian_elimination(matrix: np.ndarray) -> np.ndarray:
    """
    Performs binary gaussian elimination on an input matrix.

    Algorithm reference:
    https://www.cs.umd.edu/~gasarch/TOPICS/factoring/fastgauss.pdf

    Implementation:
    https://gist.github.com/popcornell/bc29d1b7ba37d824335ab7b6280f7fec
    """
    matrix = matrix.copy()

    nrows, ncols = matrix.shape

    # Deal with trivial cases
    if nrows == 0 or ncols == 0:
        return matrix

    i = 0
    # Each iteration of this loop will ensure that column j does not contain any 1s
    # from row i downwards.
    # If possible, row i will contain the only 1 in column j. This is possible if there
    # is a row with index pivot_row >= i that has a 1 in column j.
    for j in range(ncols):
        # Find index of first `1` in column j, from row i-th onwards.
        pivot_row = np.argmax(matrix[i:, j]) + i

        # If the pivot is 0, skip this column
        # The reason is that in this column j no row below the current row has a 1
        # in it (thus the pivot row has a zero in column j), so we can't use this
        # column to eliminate 1s in other rows.
        if matrix[pivot_row, j] == 0:
            continue

        # Swap pivot with row i
        matrix[[pivot_row, i]] = matrix[[i, pivot_row]]

        # XOR the pivot row that we just relocated into the i-th row into all other rows
        # to make all other entries in the column j 0
        ## Target row i, from column j onwards
        target_row = matrix[i, j:]

        target_column = np.copy(matrix[:, j])
        target_column[i] = 0  # don't XOR the pivot row with itself

        # The array with which we will XOR the matrix
        # Every row in column j that is 1 will be XORed with the target row
        xor_mask = np.outer(target_column, target_row)

        # XOR the matrix
        matrix[:, j:] = matrix[:, j:] ^ xor_mask

        # Increment row index
        i += 1
        # If we reach the last row, break the loop
        if i >= nrows:
            break

    return matrix


def find_maximum_matching(graph: nx.Graph) -> dict[int, int]:
    """Obtains maximum matching of a bipartite graph.

    Parameters
    ----------
    graph : nx.Graph
        Networkx Graph from which to extract the maximum matching. The graph needs to
        be bipartite.

    Returns
    ------
    matching: dict[int,int]
        Matching as dictionary whose values and keys contain the matching connections
        between nodes.
    """

    # Ensure the graph is bipartite
    if not nx.algorithms.bipartite.is_bipartite(graph):
        raise ValueError("Graph is not bipartite.")

    # Dictionary to hold the matching results
    matching = {}

    # Iterate over connected components
    for component in nx.connected_components(graph):
        # Extract the subgraph for the current component
        subgraph = graph.subgraph(component)

        # Find the bipartite sets for the component
        _, top_nodes = nx.algorithms.bipartite.sets(subgraph)

        # Find the maximum matching for the component using Hopcroft-Karp algorithm
        local_matching = nx.algorithms.bipartite.maximum_matching(
            subgraph, top_nodes=top_nodes
        )

        # Add to the overall matching dictionary
        matching.update(local_matching)

    return matching


def minimum_edge_coloring(graph: nx.Graph) -> dict[int, list[tuple[int, int]]]:
    """
    Computes the minimum edge coloring of a bipartite graph ``graph``. The chromatic
    index for all bipartite graphs is equal to the maximum degree of the graph. In the
    context of leveraging this algorithm for circuit construction, this is equivalent to
    distributing more gates into a single layer.

    Parameters
    ----------
    graph : nx.Graph
        A bipartite graph.

    Returns
    -------
    coloring: dict[int,list]
        Minimum edge coloring as a dictionary where keys are color indices and values
        the list of edges.
    """

    # Get the maximum degree of the graph, corresponding to the number of colors needed
    d = max(dict(graph.degree()).values())

    # Initialize coloring dictionary
    coloring = {i: [] for i in range(d)}
    incident_color_tracker = {node: set() for node in graph.nodes()}

    sorted_edges = [tuple(sorted(edge)) for edge in graph.edges]

    # Assign colors, ranging from 0 to d-1
    for edge in sorted_edges:
        u, v = edge
        used_colors = incident_color_tracker[u] | incident_color_tracker[v]
        color = next(i for i in range(d) if i not in used_colors)

        coloring[color].append(edge)
        incident_color_tracker[u].add(color)
        incident_color_tracker[v].add(color)

    return coloring


def extract_subgraphs_from_edge_labels(
    graph: nx.Graph, label_attribute: str = "cardinality"
) -> dict[str, nx.Graph]:
    """
    Extract subgraphs of the Tanner Graphs induced by the edges associated with a
    particular attribute. This function is mainly used for obtaining the ones associated
    with the cardinality decorators in the cardinal circuit constructor function.

    Parameters
    ----------
    graph : nx.Graph
        Tanner Graph from HGP code whose edges are decorated according to the different
        attributes.

    label_attribute: str
        Edge attribute name. Defaults to cardinality.

    Returns
    -------
    subgraphs_dict: dict[str,nx.Graph]
        Dictionary containing the value of each attribute as a key and the associated
        induced subgraph as a value.
    """

    # Check input
    if not isinstance(label_attribute, str):
        raise TypeError("Label attribute must be a string.")

    attribute_names = {key for _, _, attr in graph.edges(data=True) for key in attr}
    if label_attribute not in attribute_names:
        raise ValueError(f"Edge attribute {label_attribute} not present in graph.")

    # Check for any unlabelled edges w.r.t. to label_attribute input
    unlabelled_edges = [
        (a, b)
        for a, b, info in graph.edges(data=True)
        if info.get(label_attribute) is None
    ]
    if len(unlabelled_edges) > 0:
        raise ValueError(f"Edges {unlabelled_edges} do not contain input label.")

    # Extract unique edge labels
    attribute_values = set(nx.get_edge_attributes(graph, label_attribute).values())

    # Initialize subgraph dictionary
    subgraphs_dict = {}

    # Create subgraphs for each label
    for value in attribute_values:
        # Filter edges with the current label
        edges_with_value = [
            (u, v) for u, v, d in graph.edges(data=True) if d[label_attribute] == value
        ]

        # Create a subgraph with those edges
        subgraph = graph.edge_subgraph(edges_with_value).copy()

        # Store the subgraph in the dictionary
        subgraphs_dict[value] = subgraph

    return subgraphs_dict


def cardinality_distribution(t_graph: nx.Graph) -> dict[str, list]:
    """
    Extracts the edges in a HGP Tanner Graph associated with each cardinality {N,E,S,W}.

    Parameters
    ----------
    t_graph: nx.Graph
        Tanner graph associated with HGP code.

    Returns
    -------
    edge_dict: dict[str,list]
        A dictionary whose keys are the different cardinalities and the values
        all the edges associated with it.

    """

    # Check input
    attribute_names = {key for _, _, attr in t_graph.edges(data=True) for key in attr}

    if len(attribute_names - {"cardinality"}) > 0:
        raise ValueError(
            f"Only allowed attribute name is 'cardinality', but input "
            f"contains: {attribute_names-{'cardinality'}}."
        )

    edge_labels = set(nx.get_edge_attributes(t_graph, "cardinality").values())

    if len(edge_labels - {"E", "N", "S", "W"}) > 0:
        raise ValueError(
            f"Only cardinal values 'E','N','S','W' are allowed attributes, "
            f"but input contains invalid ones: {edge_labels-{'E', 'N', 'S', 'W'}}."
        )

    # Check for any unlabelled edges
    unlabelled_edges = {(a, b) for a, b, info in t_graph.edges(data=True) if info == {}}

    if len(unlabelled_edges) > 0:
        raise ValueError(f"Edges {unlabelled_edges} are not labeled.")

    # Initialize dictionary
    edge_dict = {}

    # Extract all the edges associated with each cardinality
    for label in edge_labels:
        edge_dict[label] = [
            (u, v) for u, v, d in t_graph.edges(data=True) if d["cardinality"] == label
        ]

    return edge_dict


def verify_css_code_condition(hx: np.ndarray, hz: np.ndarray) -> bool:
    """
    Verifies that the parity-check matrices define a valid CSS code, by checking that
    the X and Z check matrices are orthogonal, i.e. that the stabilizers commute.

    Parameters
    ----------
    hx : np.ndarray
        X parity-check matrix.
    hz : np.ndarray
        Z parity-check matrix.

    Returns
    -------
    valid: bool
        If True, the parity-check matrices define a valid CSS code.
    """

    # Check input
    for h in [hx, hz]:
        if not np.all(np.isin(h, [0, 1])):
            raise ValueError("Parity-check matrix contains non-binary elements.")

        if not np.any(h):
            raise ValueError("Parity-check matrix is empty.")

    # Compute the symplectic product of the parity-check matrices
    product = np.dot(hx, hz.T) % 2

    # Specify validity as a boolean
    valid = not bool(product.any())

    return valid
