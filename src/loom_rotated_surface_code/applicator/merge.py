"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

from itertools import product

from loom.eka import Block, Circuit, PauliOperator, Stabilizer
from loom.eka.utilities import Orientation
from loom.eka.operations import (
    Merge,
    MeasureBlockSyndromes,
    LogicalMeasurement,
)
from loom.interpreter import InterpretationStep, Cbit, Syndrome
from loom.interpreter.applicator import generate_syndromes, measureblocksyndromes

from ..code_factory import RotatedSurfaceCode


# pylint: disable=too-many-lines
def merge_consistency_check(  # pylint: disable=too-many-branches
    interpretation_step: InterpretationStep,
    operation: Merge,
) -> tuple[RotatedSurfaceCode, RotatedSurfaceCode]:
    """
    Check if the blocks are compatible for merging. Also re-orders the blocks so
    that block1 is the left or top block and block2 is the right or bottom block. This
    is required for the rest of the workflow to function properly.

    Parameters
    ----------
    interpretation_step : InterpretationStep
        InterpretationStep containing the blocks to merge.
    operation : Merge
        Descriptor of the merge operation to perform.

    Returns
    -------
    tuple[RotatedSurfaceCode, RotatedSurfaceCode]
        The blocks to merge, ordered so that block1 is the left or top block and
        block2 is the right or bottom block.

    Raises
    ------
    ValueError
        If the blocks are not compatible for merging. The blocks are compatible if:
        (1) They do not overlap (this should already be enforced).
        (2) The blocks' upper left corner qubits are aligned (either horizontally or
        vertically).
        (3) The blocks have the same size in the direction normal to the merge.
        (4) The boundaries to be merged are of the same type.
        (5) The alternate pattern of stabilizers is preserved. I.e. the blocks have the
        same upper left 4-body stabilizer if the distance is even, different stabilizer
        if the distance is odd.
        (6) There needs to be at least one row/column of data qubits between the two
        blocks.
        (7) The blocks have a single logical operator.
        Also raises a ValueError if the orientation specified in the operation does not
        match the default orientation.
    """

    name1, name2 = operation.input_blocks_name
    block1 = interpretation_step.get_block(name1)
    block2 = interpretation_step.get_block(name2)
    # 1 - Check that the blocks are of the right type
    if not isinstance(block1, RotatedSurfaceCode) or not isinstance(
        block2, RotatedSurfaceCode
    ):
        raise TypeError(
            f"The merge operation is not supported for "
            f"{tuple(set(type(block1), type(block2)))} blocks."
        )
    context_str = (
        # pylint: disable=protected-access
        f"Operation {operation.__class__.__name__} on {operation._inputs} failed:\n"
    )

    # 2 - Check that the blocks do not overlap
    if len(set(block1.data_qubits + block2.data_qubits)) < len(
        block1.data_qubits
    ) + len(block2.data_qubits):
        raise ValueError(context_str + "The blocks overlap.")

    # TODO this can be replaced by a more elaborate scheme to merge two blocks
    # 3 - Check that the blocks' upper left corners are aligned
    # Compute the number of non zero coordinates in the vector that links the upper left
    # corners of the two blocks.
    link_vector = [
        coord2 - coord1
        for coord1, coord2 in zip(
            block1.upper_left_qubit, block2.upper_left_qubit, strict=True
        )
    ]
    blocks_are_aligned = sum(coord != 0 for coord in link_vector) == 1
    index_nonzero = next((i for i, coord in enumerate(link_vector) if coord != 0), None)

    if not blocks_are_aligned:
        raise ValueError(
            context_str + "The blocks' upper left corners are not aligned."
        )

    # Ensure that block1 is left or top and block2 is right or bottom
    if link_vector[index_nonzero] < 0:
        block1, block2 = block2, block1

    # Find the direction of the merge
    if index_nonzero == 0:
        default_orientation = Orientation.HORIZONTAL
        merge_is_horizontal = True
    elif index_nonzero == 1:
        default_orientation = Orientation.VERTICAL
        merge_is_horizontal = False
    else:
        raise ValueError(
            context_str
            + "The blocks corners have same x and y coordinates but differ in another "
            "way."
        )

    orientation = operation.orientation or default_orientation
    if orientation != default_orientation:
        raise ValueError(
            context_str + "The orientation specified in the operation does not "
            "match the default orientation."
        )

    # 4 - Check that the blocks have the same size in the direction normal to the merge
    if merge_is_horizontal:
        if block1.size[1] != block2.size[1]:
            raise ValueError(
                context_str
                + "The blocks have different sizes in the vertical direction."
            )
    else:
        if block1.size[0] != block2.size[0]:
            raise ValueError(
                context_str
                + "The blocks have different sizes in the horizontal direction."
            )

    # 5 - Check that the boundaries to be merged are of the same type
    if merge_is_horizontal:
        if block1.boundary_type("right") != block2.boundary_type("left"):
            raise ValueError(
                context_str + "The boundaries to be merged are of different types."
            )
    else:
        if block1.boundary_type("bottom") != block2.boundary_type("top"):
            raise ValueError(
                context_str + "The boundaries to be merged are of different types."
            )

    # 6 - Check that the alternate pattern of stabilizers is preserved
    if merge_is_horizontal:
        distance = block2.upper_left_qubit[0] - block1.upper_left_qubit[0]
    else:
        distance = block2.upper_left_qubit[1] - block1.upper_left_qubit[1]
    #  If distance is even, upper left 4-body stabilizers should be the same type
    #  If distance is odd, upper left 4-body stabilizers should be different types
    if (
        distance % 2 == 0
        and (
            block1.upper_left_4body_stabilizer.pauli
            != block2.upper_left_4body_stabilizer.pauli
        )
    ) or (
        distance % 2 != 0
        and (
            block1.upper_left_4body_stabilizer.pauli
            == block2.upper_left_4body_stabilizer.pauli
        )
    ):
        raise ValueError(
            context_str + "The alternate pattern of stabilizers is not preserved."
        )

    # 7 - Check that there is at least one row/column of data qubits between the two
    # blocks
    if merge_is_horizontal:
        if block1.upper_left_qubit[0] + block1.size[0] == block2.upper_left_qubit[0]:
            raise ValueError(
                context_str
                + "There is no column of data qubits between the two blocks."
            )
    else:
        if block1.upper_left_qubit[1] + block1.size[1] == block2.upper_left_qubit[1]:
            raise ValueError(
                context_str + "There is no row of data qubits between the two blocks."
            )

    return (block1, block2)


def create_merge_circuit(
    interpretation_step: InterpretationStep,
    blocks: tuple[RotatedSurfaceCode, RotatedSurfaceCode],
    operation: Merge,
    qubits_to_reset: tuple[tuple[int, ...], ...],
    boundary_type: str,
) -> Circuit:
    """Create the circuit for the merge operation.

    Parameters
    ----------
    interpretation_step : InterpretationStep
        Interpretation step containing the blocks to merge.
    blocks : tuple[RotatedSurfaceCode, RotatedSurfaceCode]
        Blocks to merge.
    operation : Merge
        Descriptor of the merge operation to perform.
    qubits_to_reset : tuple[tuple[int, ...], ...]
        Qubits to reset during the merge operation.
    boundary_type : str
        Type of boundary to merge.

    Returns
    -------
    Circuit
        Circuit for the merge operation.
    """
    # B) - CIRCUIT
    #    B.1) Find which basis the data qubits have to be reset in
    # If the boundary type is X the stabilizers are Z type, they need to be reset in |0>
    # If the boundary type is Z the stabilizers are X type, they need to be reset in |+>
    reset_state = "0" if boundary_type == "X" else "+"
    #    B.2) Create a reset circuit for every reset data qubit
    reset_circuit_seq = [
        [
            Circuit(
                f"Reset_{reset_state}",
                channels=[interpretation_step.get_channel_MUT(q)],
            )
            for q in qubits_to_reset
        ]
    ]

    merge_circuit = Circuit(
        name=(
            f"Merge {blocks[0].unique_label} and {blocks[1].unique_label} into "
            f"{operation.output_block_name}"
        ),
        circuit=reset_circuit_seq,
    )

    return merge_circuit


def find_merge_stabilizer_to_circuit_mappings(
    blocks: tuple[RotatedSurfaceCode, RotatedSurfaceCode],
    new_stabs_left: list[Stabilizer],
    new_stabs_right: list[Stabilizer],
    new_stabs_top: list[Stabilizer],
    new_stabs_bottom: list[Stabilizer],
    new_bulk_stabilizers: list[Stabilizer],
    old_stabs_to_lengthen: list[Stabilizer],
) -> dict[str, str]:
    """Finds the mapping between the stabilizers in the new block and the associated
    syndrome circuits.

    Parameters
    ----------
    blocks : tuple[RotatedSurfaceCode, RotatedSurfaceCode]
        Blocks to merge.
    new_stabs_left : list[Stabilizer]
        New stabilizers that form a part of the left boundary of the new block. This is
        an empty list for horizontal merges.
    new_stabs_right : list[Stabilizer]
        New stabilizers that form a part of the right boundary of the new block. This is
        an empty list for horizontal merges.
    new_stabs_top : list[Stabilizer]
        New stabilizers that form a part of the top boundary of the new block. This is
        an empty list for vertical merges.
    new_stabs_bottom : list[Stabilizer]
        New stabilizers that form a part of the bottom boundary of the new block. This
        is an empty list for vertical merges.
    new_bulk_stabilizers : list[Stabilizer]
        New stabilizers that form a part of the bulk of the new block.
    old_stabs_to_lengthen : list[Stabilizer]
        Old stabilizers that are gonna be replaced by new stabilizers.

    Returns
    -------
    dict[str, str]
        New mapping of stabilizers to syndrome circuits for the block.
    """
    block1, block2 = blocks
    # We only use one set of SyndromeCircuits for the new block
    # Create the mapping from block2 syndrome circuits to the new syndrome_circuits list
    block2_to_block1_synd_circ_map = {
        block2_synd_circ.uuid: block1_synd_circ.uuid
        for block1_synd_circ in block1.syndrome_circuits
        for block2_synd_circ in block2.syndrome_circuits
        if block1_synd_circ.name == block2_synd_circ.name
    }
    # Ensure that the id corresponds to a stabilizer that is not removed
    conserved_stabs_id = [
        stab.uuid
        for stab in block1.stabilizers + block2.stabilizers
        if stab not in old_stabs_to_lengthen
    ]
    # Use the stabilizer to circuit mapping of the first block (apart from removed
    # stabilizers)
    new_stab_to_circuit = {
        stab_id: synd_circ_id
        for (stab_id, synd_circ_id) in block1.stabilizer_to_circuit.items()
        if stab_id in conserved_stabs_id
    }
    # Create a new mapping to the syndrome circuits for the new block for the
    # stabilizers of block2 that are not gonna be replaced
    new_stab_to_circuit.update(
        {
            stab_id: block2_to_block1_synd_circ_map[synd_circ_id]
            for (stab_id, synd_circ_id) in block2.stabilizer_to_circuit.items()
            if stab_id in conserved_stabs_id
        }
    )

    # Left stabilizers
    for stab in new_stabs_left:
        new_stab_to_circuit[stab.uuid] = next(
            syndrome_circuit.uuid
            for syndrome_circuit in block1.syndrome_circuits + block2.syndrome_circuits
            if syndrome_circuit.name == f"left-{stab.pauli.lower()}"
        )
    # Right stabilizers
    for stab in new_stabs_right:
        new_stab_to_circuit[stab.uuid] = next(
            syndrome_circuit.uuid
            for syndrome_circuit in block1.syndrome_circuits + block2.syndrome_circuits
            if syndrome_circuit.name == f"right-{stab.pauli.lower()}"
        )
    # Top stabilizers
    for stab in new_stabs_top:
        new_stab_to_circuit[stab.uuid] = next(
            syndrome_circuit.uuid
            for syndrome_circuit in block1.syndrome_circuits + block2.syndrome_circuits
            if syndrome_circuit.name == f"top-{stab.pauli.lower()}"
        )
    # Bottom stabilizers
    for stab in new_stabs_bottom:
        new_stab_to_circuit[stab.uuid] = next(
            syndrome_circuit.uuid
            for syndrome_circuit in block1.syndrome_circuits + block2.syndrome_circuits
            if syndrome_circuit.name == f"bottom-{stab.pauli.lower()}"
        )
    # Bulk stabilizers
    for stab in new_bulk_stabilizers:
        new_stab_to_circuit[stab.uuid] = next(
            syndrome_circuit.uuid
            for syndrome_circuit in block1.syndrome_circuits + block2.syndrome_circuits
            if syndrome_circuit.name == stab.pauli.lower()
        )

    return new_stab_to_circuit


def create_merge_2_body_stabilizers(
    blocks: tuple[RotatedSurfaceCode, RotatedSurfaceCode],
    merge_orientation: Orientation,
    merge_distance: int,
    filling_upper_left_qubit: tuple[int, int],
    pauli: str,
) -> tuple[list[Stabilizer], list[Stabilizer]]:
    """
    Create the new 2-body stabilizers located at the merged boundaries.

    Parameters
    ----------
    blocks : tuple[RotatedSurfaceCode, RotatedSurfaceCode]
        blocks to merge.
    merge_orientation : Orientation
        Orientation of the merge.
    merge_distance : int
        Distance between the two blocks in the direction of the merge. A distance m
        means that there are m-1 rows/columns of data qubits between the two blocks
        to merge.
    filling_upper_left_qubit : tuple[int, int]
        Upper left qubit of the filling region.
    pauli : str
        Pauli string of the new stabilizers.

    Returns
    -------
    tuple[list[Stabilizer], list[Stabilizer]]
        New left and right boundary stabilizers if the merge is horizontal, new top and
        bottom boundary stabilizers if the merge is vertical.
    """
    block1, _ = blocks
    is_merge_horizontal = merge_orientation == Orientation.HORIZONTAL

    # Index in block.size corresponding to the merge direction, 0 for horizontal, 1 for
    # vertical
    size_index = 0 if is_merge_horizontal else 1

    # The two boundaries merged both have a 2 body stabilizer in the first row/column if
    # the distance normal to the merge direction is odd, e.g. for a 3x3 block,
    # weight_2_stab_is_first_row=True and a vertical merge, the additional left boundary
    # will have a stabilizer in the first row but the right boundary does not.
    boundaries_have_same_weight_2_stab_is_first = block1.size[not size_index] % 2 == 0

    if is_merge_horizontal:
        block1_weight_2_stab_is_first_column = not block1.weight_2_stab_is_first_row
        # If the size of block1 is even, the filling's weight_2_stab_is_first_column is
        # the same as block1's else it's opposite
        filling_weight_2_stab_is_first_top_column = (
            block1_weight_2_stab_is_first_column ^ ((block1.size[size_index] % 2) == 0)
        )
        filling_weight_2_stab_is_first_bot_column = (
            filling_weight_2_stab_is_first_top_column
            if boundaries_have_same_weight_2_stab_is_first
            else not filling_weight_2_stab_is_first_top_column
        )
        n_stabs_top = merge_distance // 2 + (
            filling_weight_2_stab_is_first_top_column and (merge_distance % 2)
        )
        n_stabs_bottom = merge_distance // 2 + (
            filling_weight_2_stab_is_first_bot_column and (merge_distance % 2)
        )
        new_stabs_top = RotatedSurfaceCode.generate_weight2_stabs(
            pauli=pauli,
            initial_position=(
                filling_upper_left_qubit[0]
                + (not filling_weight_2_stab_is_first_top_column),
                filling_upper_left_qubit[1],
            ),
            num_stabs=n_stabs_top,
            orientation=Orientation.HORIZONTAL,
            is_bottom_or_right=False,
        )
        new_stabs_bottom = RotatedSurfaceCode.generate_weight2_stabs(
            pauli=pauli,
            initial_position=(
                filling_upper_left_qubit[0]
                + (not filling_weight_2_stab_is_first_bot_column),
                filling_upper_left_qubit[1] + block1.size[1] - 1,
            ),
            num_stabs=n_stabs_bottom,
            orientation=Orientation.HORIZONTAL,
            is_bottom_or_right=True,
        )

        return new_stabs_top, new_stabs_bottom

    # else
    # If the size of block 1 is even, the filling's weight_2_stab_is_first_row is
    # the same as block 1 else it's opposite
    filling_weight_2_stab_is_first_left_row = block1.weight_2_stab_is_first_row ^ (
        (block1.size[size_index] % 2) == 0
    )
    filling_weight_2_stab_is_first_right_row = (
        filling_weight_2_stab_is_first_left_row
        if boundaries_have_same_weight_2_stab_is_first
        else not filling_weight_2_stab_is_first_left_row
    )
    n_stabs_left = merge_distance // 2 + (
        filling_weight_2_stab_is_first_left_row and (merge_distance % 2)
    )
    n_stabs_right = merge_distance // 2 + (
        filling_weight_2_stab_is_first_right_row and (merge_distance % 2)
    )
    new_stabs_left = RotatedSurfaceCode.generate_weight2_stabs(
        pauli=pauli,
        initial_position=(
            filling_upper_left_qubit[0],
            filling_upper_left_qubit[1] + (not filling_weight_2_stab_is_first_left_row),
        ),
        num_stabs=n_stabs_left,
        orientation=Orientation.VERTICAL,
        is_bottom_or_right=False,
    )
    new_stabs_right = RotatedSurfaceCode.generate_weight2_stabs(
        pauli=pauli,
        initial_position=(
            filling_upper_left_qubit[0] + block1.size[0] - 1,
            filling_upper_left_qubit[1]
            + (not filling_weight_2_stab_is_first_right_row),
        ),
        num_stabs=n_stabs_right,
        orientation=Orientation.VERTICAL,
        is_bottom_or_right=True,
    )

    return new_stabs_left, new_stabs_right


def merge_stabilizers(  # pylint: disable=too-many-locals
    blocks: tuple[RotatedSurfaceCode, RotatedSurfaceCode],
    merge_is_horizontal: bool,
) -> tuple[list[Stabilizer], list[Stabilizer], list[Stabilizer], dict[str, str]]:
    """Merge the two initial blocks stabilizers into a single lists of stabilizers, the
    list of old stabilizers that are gonna be replaced and the new stabilizers that will
    replace them.

    Parameters
    ----------
    blocks : tuple[RotatedSurfaceCode, RotatedSurfaceCode]
        Blocks to merge. By construction blocks[0] should be the left or top block and
        blocks[1] should be the right or bottom block.
    merge_is_horizontal : bool
        True if the merge is horizontal, False if the merge is vertical.

    Returns
    -------
    tuple[list[Stabilizer], list[Stabilizer], list[Stabilizer], dict[str, str]]
        List of stabilizers forming the new block, list of old stabilizers to be
        replaced, list of new stabilizers which will replace them and map from
        stabilizer to the syndrome circuits.
    """
    # C) - STABILIZERS
    #    C.1) Create the new 4-body stabilizers located between the two initial blocks
    block1, block2 = blocks
    weight_4_x_schedule = block1.weight_4_x_schedule
    weight_4_z_schedule = block1.weight_4_z_schedule

    # Find the qubit located at the upper left corner of the filling
    # block1 is always the left or top block and block2 is always the right or bottom
    # block
    boundary_block_1 = (
        block1.boundary_qubits("right")
        if merge_is_horizontal
        else block1.boundary_qubits("bottom")
    )
    boundary_block_2 = (
        block2.boundary_qubits("left")
        if merge_is_horizontal
        else block2.boundary_qubits("top")
    )

    # The "filling" is the set of new 4 body stabilizers and 2-body stabilizers
    upper_left_qubit_filling = min(boundary_block_1, key=lambda x: x[0] + x[1])
    dx = (
        block2.upper_left_qubit[0] - upper_left_qubit_filling[0] + 1
        if merge_is_horizontal
        else block1.size[0]
    )
    dz = (
        block2.upper_left_qubit[1] - upper_left_qubit_filling[1] + 1
        if not merge_is_horizontal
        else block1.size[1]
    )

    adjacent_stab = next(
        stab
        for stab in block1.stabilizers
        if upper_left_qubit_filling in stab.data_qubits and len(stab.pauli) == 4
    )
    upper_left_4_body_pauli = "XXXX" if adjacent_stab.pauli[0] == "Z" else "ZZZZ"
    new_upleft_4_body_stabs = RotatedSurfaceCode.generate_weight4_stabs(
        pauli=upper_left_4_body_pauli,
        schedule=(
            weight_4_x_schedule
            if upper_left_4_body_pauli == "XXXX"
            else weight_4_z_schedule
        ),
        start_in_top_left_corner=True,
        dx=dx,
        dz=dz,
        initial_position=upper_left_qubit_filling,
    )
    new_rest_4_body_stabs = RotatedSurfaceCode.generate_weight4_stabs(
        pauli="XXXX" if upper_left_4_body_pauli == "ZZZZ" else "ZZZZ",
        schedule=(
            weight_4_x_schedule
            if upper_left_4_body_pauli != "XXXX"
            else weight_4_z_schedule
        ),
        start_in_top_left_corner=False,
        dx=dx,
        dz=dz,
        initial_position=upper_left_qubit_filling,
    )
    new_4_body_stabs = new_upleft_4_body_stabs + new_rest_4_body_stabs

    #    C.2) Create the new 2-body stabilizers
    # new_top_left_is_xxxx = upper_left_4_body_pauli == "XXXX"
    stab_left_right_is_x = (
        block1.x_boundary == Orientation.HORIZONTAL
    )  # The logical operators are aligned

    stab_left_right = "XX" if stab_left_right_is_x else "ZZ"
    stab_top_bottom = "ZZ" if stab_left_right_is_x else "XX"

    if merge_is_horizontal:
        stabs_top, stabs_bottom = create_merge_2_body_stabilizers(
            blocks=blocks,
            merge_orientation=Orientation.HORIZONTAL,
            merge_distance=dx - 1,  # dx is the number of qubits used in the filling
            filling_upper_left_qubit=upper_left_qubit_filling,
            pauli=stab_top_bottom,
        )
        stabs_left, stabs_right = [], []
    else:
        stabs_top, stabs_bottom = [], []
        stabs_left, stabs_right = create_merge_2_body_stabilizers(
            blocks=blocks,
            merge_orientation=Orientation.VERTICAL,
            merge_distance=dz - 1,  # dz is the number of qubits used in the filling
            filling_upper_left_qubit=upper_left_qubit_filling,
            pauli=stab_left_right,
        )

    new_2_body_stabs = stabs_top + stabs_bottom + stabs_left + stabs_right

    #    C.3) Find the 2-body stabilizers which weight should be increased
    old_stabs_to_lengthen = [
        stab
        for stab in block1.stabilizers + block2.stabilizers
        if set(stab.data_qubits).issubset(set(boundary_block_1 + boundary_block_2))
        and len(stab.data_qubits) == 2
    ]
    #    C.4) Find the associated 4-body stabilizers
    #    There is a 1-to-1 correspondence between old_stabs_to_lengthen and
    #    new_stabs_increased_weight
    # NOTE: the order in the product matters, we want to match the order
    # old_stabs_to_lengthen to new_stabs_increased_weight
    new_stabs_increased_weight = [
        stab
        for (old_stab, stab) in product(old_stabs_to_lengthen, new_4_body_stabs)
        if set(old_stab.data_qubits).issubset(set(stab.data_qubits))
        and old_stab.pauli in stab.pauli
    ]

    #    C.5) Create the new `stabilizer_to_circuit` mapping
    new_stab_to_circ = find_merge_stabilizer_to_circuit_mappings(
        blocks,
        stabs_left,
        stabs_right,
        stabs_top,
        stabs_bottom,
        new_4_body_stabs,
        old_stabs_to_lengthen,
    )

    #    C.6) Create a single set of stabilizers for the new block
    new_stabilizers = new_4_body_stabs + new_2_body_stabs

    new_block_stabilizers = [
        stab
        for stab in block1.stabilizers + block2.stabilizers
        if stab not in old_stabs_to_lengthen
    ] + new_stabilizers

    return (
        new_block_stabilizers,
        old_stabs_to_lengthen,
        new_stabs_increased_weight,
        new_stab_to_circ,
    )


def find_data_qubits_between(
    qubit1: tuple[int, ...], qubit2: tuple[int, ...]
) -> list[tuple[int, ...]]:
    """
    Generate all qubits between two given qubits coordinates (not inclusive).

    Parameters
    ----------
    qubit1: tuple[int, ...]:
        The first qubit (x1, y1, ...).
    qubit2: tuple[int, ...]:
        The second qubit (x2, y2, ...).

    Returns
    -------
    list[tuple[int, ...]]
        A list of qubits between the two qubits.
    """
    x1, y1 = qubit1[:2]
    x2, y2 = qubit2[:2]

    if x1 == x2:
        # Vertical alignment
        y_start, y_end = sorted([y1, y2])
        qubits_between = [(x1, y, 0) for y in range(y_start + 1, y_end)]
    elif y1 == y2:
        # Horizontal alignment
        x_start, x_end = sorted([x1, x2])
        qubits_between = [(x, y1, 0) for x in range(x_start + 1, x_end)]
    else:
        raise ValueError("The qubits are not aligned horizontally or vertically.")

    return qubits_between


def merge_logical_operators(
    interpretation_step: InterpretationStep,
    blocks: tuple[RotatedSurfaceCode, RotatedSurfaceCode],
    qubits_to_reset: tuple[tuple[int, ...]],
    merge_is_horizontal: bool,
) -> tuple[
    InterpretationStep, PauliOperator, tuple[Cbit, ...], PauliOperator, tuple[Cbit, ...]
]:
    """Generate the new logical operators from the two blocks to merge. Note that they
    will be pushed to the upper left corner of the new block, similarly to the default
    choice of logical operators. This function also retrieves the lates measurements of
    the equivalent stabilizers and returns them.

    Parameters
    ----------
    interpretation_step : InterpretationStep
        Interpretation step containing the blocks to merge.
    blocks : tuple[RotatedSurfaceCode, RotatedSurfaceCode]
        Initial blocks to merge. By convention, the first block must be the one on the
        left or top. The second block must be the one on the right or bottom
        respectively.
    qubits_to_reset : tuple[tuple[int, ...]]
        Qubits to measure between the two blocks.
    merge_is_horizontal : bool
        Whether the merge is horizontal or vertical.

    Returns
    -------
    tuple[
        InterpretationStep,
        PauliOperator,
        tuple[Cbit, ...],
        PauliOperator,
        tuple[Cbit, ...]
    ]
        The updated interpretation step, the new logical X operator, the measurements
        coming from equivalent X stabilizers, the new logical Z operator and the
        measurements coming from equivalent Z stabilizers.
    """
    # D) LOGICAL OPERATORS
    block1, block2 = blocks
    x_log_is_horizontal = block1.x_boundary == Orientation.HORIZONTAL

    # The logical operator aligned with the merge will be modified
    if x_log_is_horizontal == merge_is_horizontal:
        pauli_to_be_merged = "X"
        initial_log_ops_to_be_merged = (
            block1.logical_x_operators[0],
            block2.logical_x_operators[0],
        )
        log_ops_untouched = (
            block1.logical_z_operators[0],
            block2.logical_z_operators[0],
        )
    else:
        pauli_to_be_merged = "Z"
        initial_log_ops_to_be_merged = (
            block1.logical_z_operators[0],
            block2.logical_z_operators[0],
        )
        log_ops_untouched = (
            block1.logical_x_operators[0],
            block2.logical_x_operators[0],
        )

    #    D.1) Align the 2 initial operators with stabilizer products
    aligned_logical_block_1, extra_stabs1 = (
        block1.get_shifted_equivalent_logical_operator(
            initial_log_ops_to_be_merged[0],
            block1.upper_left_qubit,
        )
    )
    aligned_logical_block_2, extra_stabs2 = (
        block2.get_shifted_equivalent_logical_operator(
            initial_log_ops_to_be_merged[1],
            block2.upper_left_qubit,
        )
    )

    #    D.2) Retrieve the cbits coming from the latest equivalent stabilizer
    #    measurements.
    cbits_block_1 = interpretation_step.retrieve_cbits_from_stabilizers(
        extra_stabs1, block1
    )
    cbits_block_2 = interpretation_step.retrieve_cbits_from_stabilizers(
        extra_stabs2, block2
    )

    #    D.3) Create new logical operators using qubits to measure
    # We assume the logical operators are on a line and located between the end of the
    # first logical operator and the start of the second one
    qubits_in_between = find_data_qubits_between(
        max(aligned_logical_block_1.data_qubits, key=lambda x: x[0] + x[1]),
        min(aligned_logical_block_2.data_qubits, key=lambda x: x[0] + x[1]),
    )
    if not set(qubits_in_between).issubset(set(qubits_to_reset)):
        raise RuntimeError("The qubits chosen are not included in the reset qubits.")

    qubits_new_logical = (
        aligned_logical_block_1.data_qubits
        + aligned_logical_block_2.data_qubits
        + tuple(qubits_in_between)
    )

    new_logical = PauliOperator(
        pauli=pauli_to_be_merged * len(qubits_new_logical),
        data_qubits=qubits_new_logical,
    )
    untouched_logical = log_ops_untouched[0]
    new_log_x, new_log_z = (
        (new_logical, untouched_logical)
        if new_logical.pauli[0] == "X"
        else (untouched_logical, new_logical)
    )

    #    D.4) Update `logical_x/z_evolution`
    if pauli_to_be_merged == "X":
        interpretation_step.logical_x_evolution[new_logical.uuid] = (
            initial_log_ops_to_be_merged[0].uuid,
            initial_log_ops_to_be_merged[1].uuid,
        ) + tuple(stab.uuid for stab in extra_stabs1 + extra_stabs2)
        cbits_x = cbits_block_1 + cbits_block_2
        cbits_z = ()
    else:
        interpretation_step.logical_z_evolution[new_logical.uuid] = (
            initial_log_ops_to_be_merged[0].uuid,
            initial_log_ops_to_be_merged[1].uuid,
        ) + tuple(stab.uuid for stab in extra_stabs1 + extra_stabs2)
        cbits_x = ()
        cbits_z = cbits_block_1 + cbits_block_2

    return interpretation_step, new_log_x, cbits_x, new_log_z, cbits_z


def create_syndromes(  # pylint: disable= too-many-arguments
    interpretation_step: InterpretationStep,
    qubits_to_reset: tuple[tuple[int, int, int], ...],
    reset_type: str,
    new_stabilizers: tuple[Stabilizer, ...],
    new_stabs_increased_weight: tuple[Stabilizer, ...],
    merged_block: Block,
) -> tuple[Syndrome, ...]:
    """Creates the new syndromes for the stabilizers that are reset.

    Parameters
    ----------
    interpretation_step : InterpretationStep
        Interpretation step to which we add the new syndromes.
    qubits_to_reset : tuple[tuple[int, int, int], ...]
        Tuple of qubits that are reset during the merge
    reset_type:
        Pauli type of the reset, this gives information on which stabilizer is reset in
        a deterministic state.
    new_stabilizers : tuple[Stabilizer, ...]
        Stabilizers created by the merge operation.
    new_stabs_increased_weight:
        Stabilizers that grew from a previous 2-body of the initial blocks.
    merged_block_id : str
        UUID of the newly merged block.

    Returns
    -------
    tuple[Syndrome, ...]
        Reference syndromes of the reset stabilizers.
    """
    reset_stabilizers = tuple(
        stab
        for stab in new_stabilizers
        if (
            all(q in qubits_to_reset for q in stab.data_qubits)
            and (stab not in new_stabs_increased_weight)
            and set(stab.pauli) == {reset_type}
        )
    )
    syndromes = generate_syndromes(
        interpretation_step=interpretation_step,
        stabilizers=reset_stabilizers,
        block=merged_block,
        stab_measurements=tuple(() for _ in reset_stabilizers),
    )
    return syndromes


def merge(  # pylint: disable=line-too-long, too-many-locals
    interpretation_step: InterpretationStep,
    operation: Merge,
    same_timeslice: bool,
    debug_mode: bool,
) -> InterpretationStep:
    """
    Merge two blocks specified in the Merge operation.

    The algorithm is the following:

    - A.) DATA QUBITS
    
        - A.1) Find data qubits to be measured in between the two blocks, they will be \
        merged into a single new block
        
    - B.) CIRCUIT
    
        - B.1) Create classical channels for all data qubit measurements
        - B.2) Create a measurement circuit for every measured data qubit
        - B.3) Append the measurement circuits to the InterpretationStep circuit. \
        If needed, apply a basis change
            
    - C.) - STABILIZERS
    
        - C.1) Create the new 4-body stabilizers located between the two initial blocks
        - C.2) Create the new 2-body stabilizers located at the merged boundaries
        - C.3) Find the 2-body stabilizers which weight should be increased
        - C.4) Find the associated 4-body stabilizers
        - C.5) Create the new ``stabilizer_to_circuit`` mapping
        - C.6) Create a single set of stabilizers for the new block
        - C.7) Update ``stabilizer_evolution`` and ``stabilizer_updates`` for the \
        stabilizers which have been increased in weight
        
    - D.) LOGICAL OPERATORS
    
        - D.1) Align the 2 initial operators with stabilizer products
        - D.2) Retrieve the cbits coming from the latest equivalent stabilizer \
        measurements.
        - D.3) Create new logical operators using qubits to measure
        - D.4) Update ``logical_x/z_evolution``
        - D.5) Update ``logical_x/z_updates``
        
    - E.) NEW BLOCK AND NEW INTERPRETATION STEP
    
        - E.1) Create the new block
        - E.2) Update the block history
        
    - F.) SYNDROMES
    
        - F.1) Generate syndromes for newly created stabilizers

    - G) JOINT OBSERVABLES

        - G.1) Obtain the cbits for joint observable and store them in the
               interpretation_step

    Parameters
    ----------
    interpretation_step : InterpretationStep
        Input interpretation step.
    operation : Merge
        Descriptor of the merge operation to perform.
    same_timeslice : bool
        Flag indicating whether the operation is part of the same timestep as the
        previous operation.
    debug_mode : bool
        Flag indicating whether the interpretation should be done in debug mode.
        Activating debug mode will enable commutation validation for Block.

    Returns
    -------
    InterpretationStep
        Modified interpretation step after the merge operation.
    """
    # Consistency checks
    block1, block2 = merge_consistency_check(interpretation_step, operation)

    # A) - DATA QUBITS
    #    Find data qubits to be measured in between the two blocks, they will be
    #    merged into a single new block
    link_vector = [
        coord2 - coord1
        for coord1, coord2 in zip(
            block1.upper_left_qubit, block2.upper_left_qubit, strict=True
        )
    ]
    index_nonzero = next((i for i, coord in enumerate(link_vector) if coord != 0))
    # Ensure that block1 is left or top and block2 is right or bottom
    if link_vector[index_nonzero] < 0:
        block1, block2 = block2, block1

    merge_is_horizontal = not bool(index_nonzero)

    if merge_is_horizontal:
        boundary_qubits_block1 = block1.boundary_qubits("right")
        _ = block2.boundary_qubits("left")
        distance = block2.upper_left_qubit[0] - (
            block1.upper_left_qubit[0] + block1.size[0]
        )
    else:
        boundary_qubits_block1 = block1.boundary_qubits("bottom")
        _ = block2.boundary_qubits("top")
        distance = block2.upper_left_qubit[1] - (
            block1.upper_left_qubit[1] + block1.size[1]
        )

    # Create a square of qubits to reset between the two blocks
    qubits_to_reset = [
        (
            qubit[0] + i * merge_is_horizontal,
            qubit[1] + i * (not merge_is_horizontal),
            0,
        )
        for qubit in boundary_qubits_block1
        for i in range(1, distance + 1)
    ]

    # B) - CIRCUIT
    #    B.1) Create classical channels for all data qubit measurements
    #    B.2) Create a measurement circuit for every measured data qubit
    boundary_type = (
        block1.boundary_type("right")
        if merge_is_horizontal
        else block1.boundary_type("bottom")
    )
    merge_circuit = create_merge_circuit(
        interpretation_step,
        (block1, block2),
        operation,
        qubits_to_reset,
        boundary_type,
    )
    #    B.3) Append the measurement circuits to the InterpretationStep circuit
    #       If needed, apply a basis change
    interpretation_step.append_circuit_MUT(merge_circuit, same_timeslice)

    # C) - STABILIZERS
    #    C.1) Create the new 4-body stabilizers located between the two initial blocks
    #    C.2) Create the new 2-body stabilizers located at the merged boundaries
    #    C.3) Find the 2-body stabilizers which weight should be increased
    #    C.4) Find the associated 4-body stabilizers
    #    C.5) Create the new `stabilizer_to_circuit` mapping
    #    C.6) Create a single set of stabilizers for the new block
    (
        new_block_stabilizers,
        old_stabs_to_lengthen,
        new_stabs_increased_weight,
        new_stabilizer_to_circuit,
    ) = merge_stabilizers((block1, block2), merge_is_horizontal)

    #    C.7) Update `stabilizer_evolution` and `stabilizer_updates` for the
    #         stabilizers which have been increased in weight
    # Stabilizer evolution: Update the stabilizer evolution dictionary
    stab_map_weight2_to_weight4 = {
        new_stab.uuid: (stab.uuid,)
        for new_stab, stab in zip(
            new_stabs_increased_weight, old_stabs_to_lengthen, strict=True
        )
    }
    interpretation_step.stabilizer_evolution.update(stab_map_weight2_to_weight4)
    # Stabilizer updates: Since we use a reset, there is no update to the stabilizers

    # D) LOGICAL OPERATORS
    #    D.1) Align the 2 initial operators with stabilizer products
    #    D.2) Retrieve the cbits coming from the latest equivalent stabilizer
    #         measurements.
    #    D.3) Create new logical operators using qubits to measure
    #    D.4) Update `logical_x/z_evolution`
    interpretation_step, new_log_x, cbits_x, new_log_z, cbits_z = (
        merge_logical_operators(
            interpretation_step,
            (block1, block2),
            qubits_to_reset,
            merge_is_horizontal,
        )
    )

    #    D.5) Update `logical_x/z_updates`
    # Inherit the updates from previous operators (only if the operator has changed),
    # adding the Cbits coming from moving logical operators
    interpretation_step.update_logical_operator_updates_MUT(
        operator_type="X",
        logical_operator_id=new_log_x.uuid,
        new_updates=cbits_x,
        inherit_updates=(
            new_log_x.uuid
            not in (
                block1.logical_x_operators[0].uuid,
                block2.logical_x_operators[0].uuid,
            )
        ),
    )
    interpretation_step.update_logical_operator_updates_MUT(
        operator_type="Z",
        logical_operator_id=new_log_z.uuid,
        new_updates=cbits_z,
        inherit_updates=(
            new_log_z.uuid
            not in (
                block1.logical_z_operators[0].uuid,
                block2.logical_z_operators[0].uuid,
            )
        ),
    )

    # E) NEW BLOCK AND NEW INTERPRETATION STEP
    #    E.1) Create the new block
    merged_block = RotatedSurfaceCode(
        stabilizers=list(new_block_stabilizers),
        logical_x_operators=[new_log_x],
        logical_z_operators=[new_log_z],
        unique_label=operation.output_block_name,
        syndrome_circuits=block1.syndrome_circuits,
        stabilizer_to_circuit=new_stabilizer_to_circuit,
        skip_validation=not debug_mode,
    )
    #    E.2) Update the block history
    # Update only the blocks that are involved in the merge
    interpretation_step.update_block_history_and_evolution_MUT(
        new_blocks=(merged_block,),
        old_blocks=(block1, block2),
    )

    # F) SYNDROMES
    #    F.1) Generate syndromes for newly created stabilizers
    merge_syndromes = create_syndromes(
        interpretation_step=interpretation_step,
        qubits_to_reset=qubits_to_reset,
        reset_type=("X" if boundary_type == "Z" else "Z"),
        new_stabilizers=new_block_stabilizers,
        new_stabs_increased_weight=new_stabs_increased_weight,
        merged_block=merged_block,
    )
    interpretation_step.append_syndromes_MUT(merge_syndromes)

    # G) JOINT OBSERVABLE

    # Measure blocks once to obtain the cbits for the stabilizers in between
    interpretation_step = measureblocksyndromes(
        interpretation_step,
        MeasureBlockSyndromes(merged_block.unique_label, 1),
        same_timeslice=False,
        debug_mode=debug_mode,
    )

    new_log_op, old_log_op_2, log_updates = (
        (
            merged_block.logical_x_operators[0],
            block2.logical_x_operators[0],
            interpretation_step.logical_x_operator_updates,
        )
        if boundary_type == "X"
        else (
            merged_block.logical_z_operators[0],
            block2.logical_z_operators[0],
            interpretation_step.logical_z_operator_updates,
        )
    )

    # Obtain stabilizers between the logical operators of block1 and block2.
    # Note that by default, the new logical operator is equal to that of block1.
    _, stabs_between = merged_block.get_shifted_equivalent_logical_operator(
        new_log_op, min(old_log_op_2.data_qubits, key=lambda x: x[0] + x[1])
    )

    # Retrieve cbits associated with `stabs_between` from the latest stabilizer
    # measurements.
    stabs_between_cbits = interpretation_step.retrieve_cbits_from_stabilizers(
        stabs_between, merged_block
    )

    log_corrections = log_updates.get(new_log_op.uuid, ()) + log_updates.get(
        old_log_op_2.uuid, ()
    )

    joint_observable = stabs_between_cbits + log_corrections

    interpretation_step.logical_measurements[
        LogicalMeasurement(
            blocks=(block1.unique_label, block2.unique_label),
            observable=boundary_type * 2,
        )
    ] = joint_observable

    return interpretation_step
