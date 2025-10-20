"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

# Utilities used by cliffordsim tests
import numpy as np

from loom.cliffordsim.operations import (
    Hadamard,
    Phase,
    PhaseInv,
    CNOT,
    CZ,
    CY,
    SWAP,
    X,
    Y,
    Z,
    GateOperation,
)

sq_gates_hermitian = [
    Hadamard,
    X,
    Y,
    Z,
]

sq_gates_non_hermitian = [
    Phase,
    PhaseInv,
]

two_q_gates = [
    CNOT,
    CZ,
    CY,
    SWAP,
]


def random_gate(
    nqubits: int,
    seed: int | None = None,
    chance_of_2q_gate: float = 0.5,
    include_non_hermitian=True,
) -> GateOperation:
    """Returns a random (single or two) qubit gate operation.

    Parameters
    ----------
    nqubits : int
        The number of qubits in the register.
    seed : int | None, optional
        _description_, by default None
    chance_of_2q_gate : float, optional
        The chance that a 2qubit gate is selected, by default 0.5
    include_non_hermitian : bool, optional
        Whether to include non-hermitian operations, by default True

    Returns
    -------
    GateOperation
        _description_
    """

    rand_gen = np.random.default_rng(seed=seed)

    apply_two_q_gate = rand_gen.random() < chance_of_2q_gate

    if apply_two_q_gate:
        qubs = np.arange(nqubits)
        rand_gen.shuffle(qubs)
        q0, q1 = tuple(qubs[0:2])

        return rand_gen.choice(two_q_gates)(q0, q1)

    sq_gates = sq_gates_hermitian.copy()
    if include_non_hermitian:
        sq_gates += sq_gates_non_hermitian

    return rand_gen.choice(sq_gates)(rand_gen.integers(0, nqubits))


def random_list_of_gates(
    nqubits: int,
    ngates: int,
    seed: int | None = None,
    chance_of_2q_gate: float = 0.5,
    include_non_hermitian=True,
) -> list[GateOperation]:
    """Returns a list of randomly selected gate operations.

    Parameters
    ----------
    nqubits : int
        The number of qubits in the register.
    ngates : int
        _description_
    seed : int | None, optional
        _description_, by default None
    chance_of_2q_gate : float, optional
        The chance that a 2qubit gate is selected, by default 0.5
    include_non_hermitian : bool, optional
        Whether to include non-hermitian operations, by default True

    Returns
    -------
    list[GateOperation]
        _description_
    """
    rand_gen = np.random.default_rng(seed)
    gate_list = [
        random_gate(
            nqubits=nqubits,
            seed=rand_gen.integers(10**5),
            chance_of_2q_gate=chance_of_2q_gate,
            include_non_hermitian=include_non_hermitian,
        )
        for _ in range(ngates)
    ]

    return gate_list
