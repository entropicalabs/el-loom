"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

# pylint: disable=duplicate-code, too-many-lines
import unittest
from copy import deepcopy
from itertools import product

from loom.eka import Circuit, Channel, Lattice, Eka, Stabilizer, PauliOperator
from loom.eka.utilities import Orientation
from loom.eka.operations import Merge
from loom.interpreter import InterpretationStep, interpret_eka, Syndrome

from loom_rotated_surface_code.code_factory import RotatedSurfaceCode
from loom_rotated_surface_code.applicator.merge import (
    merge_consistency_check,
    create_merge_circuit,
    find_merge_stabilizer_to_circuit_mappings,
    create_merge_2_body_stabilizers,
    merge_stabilizers,
    find_data_qubits_between,
    merge_logical_operators,
    create_syndromes,
    merge,
)

# pylint: disable=too-many-lines


class TestRotatedSurfaceCodeMerge(unittest.TestCase):
    """
    Test the applicator for the merge operation of RotatedSurfaceCode blocks.
    """

    def setUp(self):
        self.lattice = Lattice.square_2d((10, 10))
        self.block_1 = RotatedSurfaceCode.create(
            dx=3,
            dz=3,
            lattice=self.lattice,
            unique_label="standard1",
        )
        self.block_2 = RotatedSurfaceCode.create(
            dx=3,
            dz=3,
            lattice=self.lattice,
            unique_label="standard2",
        ).shift((4, 0))
        self.base_step = InterpretationStep(
            block_history=((self.block_1, self.block_2),),
            # Add syndromes manually (required for displacement of logical operators)
            syndromes=tuple(
                Syndrome(
                    stabilizer=stab.uuid,
                    measurements=((f"c_{stab.ancilla_qubits[0]}", 0),),
                    block=block.uuid,
                    round=5,
                    corrections=(),
                )
                for block in (self.block_1, self.block_2)
                for stab in block.stabilizers
            ),
        )

    def test_merge_consistency_check(self):
        """
        Test that the merge_consistency_check function raises all the expected errors.
        """
        # 1 - Test blocks that overlap
        block1 = RotatedSurfaceCode.create(
            dx=3,
            dz=3,
            lattice=self.lattice,
            unique_label="q1",
        )
        # The first column of block2 overlaps with the last column of block1
        block2 = RotatedSurfaceCode.create(
            dx=3,
            dz=3,
            lattice=self.lattice,
            unique_label="q2",
        ).shift((2, 0))
        input_step = InterpretationStep(
            block_history=((block1, block2),),
        )

        horizontal_merge = Merge(["q1", "q2"], "q3")

        with self.assertRaises(ValueError) as cm:
            _ = merge_consistency_check(input_step, horizontal_merge)
        self.assertIn("The blocks overlap", str(cm.exception))

        # 2 - Test blocks which corners are not aligned
        block1 = RotatedSurfaceCode.create(
            dx=3,
            dz=3,
            lattice=self.lattice,
            unique_label="q1",
        )
        # The second block is shifted by (4,1) which means that the corners are not
        # aligned
        block2 = RotatedSurfaceCode.create(
            dx=3,
            dz=3,
            position=(4, 1),
            lattice=self.lattice,
            unique_label="q2",
        )
        input_step = InterpretationStep(
            block_history=((block1, block2),),
        )

        horizontal_merge = Merge(["q1", "q2"], "q3")

        with self.assertRaises(ValueError) as cm:
            _ = merge_consistency_check(input_step, horizontal_merge)
        self.assertIn(
            "The blocks' upper left corners are not aligned.", str(cm.exception)
        )

        # 3 - Test blocks that do not have the same size in the direction of merge
        block1 = RotatedSurfaceCode.create(
            dx=3,
            dz=3,
            lattice=self.lattice,
            unique_label="q1",
        )
        block2 = RotatedSurfaceCode.create(
            dx=3,
            dz=4,
            lattice=self.lattice,
            unique_label="q2",
        ).shift((4, 0))
        input_step = InterpretationStep(
            block_history=((block1, block2),),
        )

        horizontal_merge = Merge(["q1", "q2"], "q3")

        with self.assertRaises(ValueError) as cm:
            _ = merge_consistency_check(input_step, horizontal_merge)
        self.assertIn(
            "The blocks have different sizes in the vertical direction.",
            str(cm.exception),
        )

        # 4 - Test blocks that do not have the same boundary types
        block1 = RotatedSurfaceCode.create(
            dx=3,
            dz=3,
            lattice=self.lattice,
            unique_label="q1",
        )
        block2 = RotatedSurfaceCode.create(
            dx=3,
            dz=3,
            position=(4, 0),
            x_boundary=Orientation.VERTICAL,
            lattice=self.lattice,
            unique_label="q2",
        )
        input_step = InterpretationStep(
            block_history=((block1, block2),),
        )

        horizontal_merge = Merge(["q1", "q2"], "q3")
        with self.assertRaises(ValueError) as cm:
            _ = merge_consistency_check(input_step, horizontal_merge)
        self.assertIn(
            "The boundaries to be merged are of different types.", str(cm.exception)
        )

        # 5 - Test blocks that would not preserve the alternate pattern of 4-body
        # stabilizers
        block1 = RotatedSurfaceCode.create(
            dx=3,
            dz=3,
            lattice=self.lattice,
            unique_label="q1",
        )
        block2 = RotatedSurfaceCode.create(
            dx=3,
            dz=3,
            position=(4, 0),
            weight_2_stab_is_first_row=False,
            lattice=self.lattice,
            unique_label="q2",
        )
        input_step = InterpretationStep(
            block_history=((block1, block2),),
        )
        horizontal_merge = Merge(["q1", "q2"], "q3")

        with self.assertRaises(ValueError) as cm:
            _ = merge_consistency_check(input_step, horizontal_merge)
        self.assertIn(
            "The alternate pattern of stabilizers is not preserved.",
            str(cm.exception),
        )

        # 6 - Test blocks that do not have one row/column of data qubits between them
        # 6a - Test for a horizontal merge
        block1 = RotatedSurfaceCode.create(
            dx=3,
            dz=3,
            lattice=self.lattice,
            unique_label="q1",
        )
        block2 = RotatedSurfaceCode.create(
            dx=3,
            dz=3,
            position=(3, 0),
            weight_2_stab_is_first_row=False,
            lattice=self.lattice,
            unique_label="q2",
        )
        input_step = InterpretationStep(
            block_history=((block1, block2),),
        )
        horizontal_merge = Merge(["q1", "q2"], "q3")

        with self.assertRaises(ValueError) as cm:
            _ = merge_consistency_check(input_step, horizontal_merge)
        self.assertIn(
            "There is no column of data qubits between the two blocks.",
            str(cm.exception),
        )
        # 6b - Test for a vertical merge
        block2 = RotatedSurfaceCode.create(
            dx=3,
            dz=3,
            position=(0, 3),
            weight_2_stab_is_first_row=False,
            lattice=self.lattice,
            unique_label="q2",
        )
        input_step = InterpretationStep(
            block_history=((block1, block2),),
        )
        vertical_merge = Merge(["q1", "q2"], "q3")

        with self.assertRaises(ValueError) as cm:
            _ = merge_consistency_check(input_step, vertical_merge)
        self.assertIn(
            "There is no row of data qubits between the two blocks.",
            str(cm.exception),
        )

    def test_create_merge_circuit(self):
        """
        Test that the circuit created is the right one.
        """
        # 1 - Test a merge with a Z boundary

        blocks = (self.block_1, self.block_2)
        qubits_to_measure = [(3, 0, 0), (3, 1, 0), (3, 2, 0)]
        merge_op = Merge(["standard1", "standard2"], "merged")

        new_base_step = deepcopy(self.base_step)
        circuit = create_merge_circuit(
            new_base_step,
            blocks,
            merge_op,
            qubits_to_measure,
            boundary_type=self.block_1.boundary_type("right"),
        )
        # Note that this configuration requires Hadamard gates
        qubit_channels = [Channel("quantum", label=str(q)) for q in qubits_to_measure]
        expected_circuit = Circuit(
            name="merge standard1 and standard2 into merged",
            circuit=[
                [Circuit(name="Reset_+", channels=[q]) for q in qubit_channels],
            ],
        )
        self.assertEqual(circuit, expected_circuit)

        # 2 - Test a merge with an X boundary
        block1 = RotatedSurfaceCode.create(
            dx=3,
            dz=3,
            lattice=self.lattice,
            unique_label="q1",
            x_boundary=Orientation.VERTICAL,
        )
        block2 = RotatedSurfaceCode.create(
            dx=3,
            dz=3,
            lattice=self.lattice,
            unique_label="q2",
            x_boundary=Orientation.VERTICAL,
        ).shift((4, 0))
        merge_op = Merge(["q1", "q2"], "q3")
        qubits_to_measure = [(3, 0, 0), (3, 1, 0), (3, 2, 0)]

        new_base_step = deepcopy(self.base_step)
        circuit = create_merge_circuit(
            new_base_step,
            (block1, block2),
            merge_op,
            qubits_to_measure,
            boundary_type=block1.boundary_type("right"),
        )
        # Note that this configuration does not require Hadamard gates
        qubit_channels = [Channel("quantum", label=str(q)) for q in qubits_to_measure]
        expected_circuit = Circuit(
            name="merge q1 and q2 into q3",
            circuit=[
                [Circuit(name="Reset_0", channels=[q]) for q in qubit_channels],
            ],
        )
        self.assertEqual(circuit, expected_circuit)

    def test_find_stabilizer_to_circuit_mappings(self):
        """
        Test that the stabilizer to circuit mapping is correctly generated.
        """
        blocks = (self.block_1, self.block_2)
        # We only use one of them they should both contain the same information
        syndrome_circuits = self.block_1.syndrome_circuits
        # Remap to the syndrome circuit of the first block
        block2_to_block1_synd_circ_map = {
            block2_synd_circ.uuid: block1_synd_circ.uuid
            for block1_synd_circ in self.block_1.syndrome_circuits
            for block2_synd_circ in self.block_2.syndrome_circuits
            if block1_synd_circ.name == block2_synd_circ.name
        }

        # Find the removed stabilizers
        right_boundary_block1 = [
            stab
            for stab in self.block_1.stabilizers
            if set(stab.data_qubits).issubset(
                set(self.block_1.boundary_qubits("right"))
            )
        ]
        left_boundary_block2 = [
            stab
            for stab in self.block_2.stabilizers
            if set(stab.data_qubits).issubset(
                set(self.block_2.boundary_qubits("left"))  # pylint: disable=no-member
            )
        ]
        old_2body_to_lengthen = right_boundary_block1 + left_boundary_block2
        old_2body_ids = [stab.uuid for stab in old_2body_to_lengthen]

        # Define the new stabilizers to test the function
        new_bulk = [
            Stabilizer(
                pauli="XXXX",
                data_qubits=((3, 1, 0), (2, 1, 0), (3, 2, 0), (2, 2, 0)),
                ancilla_qubits=((3, 2, 1),),
            ),
            Stabilizer(
                pauli="XXXX",
                data_qubits=((4, 0, 0), (3, 0, 0), (4, 1, 0), (3, 1, 0)),
                ancilla_qubits=((4, 1, 1),),
            ),
            Stabilizer(
                pauli="ZZZZ",
                data_qubits=((3, 0, 0), (3, 1, 0), (2, 0, 0), (2, 1, 0)),
                ancilla_qubits=((3, 1, 1),),
            ),
            Stabilizer(
                pauli="ZZZZ",
                data_qubits=((4, 1, 0), (4, 2, 0), (3, 1, 0), (3, 2, 0)),
                ancilla_qubits=((4, 2, 1),),
            ),
        ]
        new_left_stabs = []
        new_right_stabs = []
        new_top_stabs = [
            Stabilizer("ZZ", ((4, 0, 0), (3, 0, 0)), ancilla_qubits=((4, 0, 1),))
        ]
        new_bottom_stabs = [
            Stabilizer("ZZ", ((3, 2, 0), (2, 2, 0)), ancilla_qubits=((3, 3, 1),))
        ]
        stabilizer_to_circuit = find_merge_stabilizer_to_circuit_mappings(
            blocks,
            new_left_stabs,
            new_right_stabs,
            new_top_stabs,
            new_bottom_stabs,
            new_bulk,
            old_2body_to_lengthen,
        )

        expected_stabilizer_to_circuit = (
            {
                stab.uuid: synd_circuit.uuid
                for synd_circuit in syndrome_circuits
                for stab in new_bulk
                if synd_circuit.name == stab.pauli.lower()
            }
            | {
                stab.uuid: synd_circuit.uuid
                for synd_circuit in syndrome_circuits
                for stab in new_top_stabs
                if synd_circuit.name == f"top-{stab.pauli.lower()}"
            }
            | {
                stab.uuid: synd_circuit.uuid
                for synd_circuit in syndrome_circuits
                for stab in new_bottom_stabs
                if synd_circuit.name == f"bottom-{stab.pauli.lower()}"
            }
            | {  # The first block mapping is conserved apart from removed stabs
                stab_id: synd_circ_id
                for stab_id, synd_circ_id in self.block_1.stabilizer_to_circuit.items()
                if stab_id not in old_2body_ids
            }
            | {  # The second block has to be remapped
                stab_id: block2_to_block1_synd_circ_map[synd_circ_id]
                for stab_id, synd_circ_id in self.block_2.stabilizer_to_circuit.items()
                if stab_id not in old_2body_ids
            }
        )

        self.assertEqual(stabilizer_to_circuit, expected_stabilizer_to_circuit)

    def test_create_merge_2_body_stabilizers(self):
        "Test that the 2-body stabilizers are correctly created."

        lattice = Lattice.square_2d((20, 20))

        expected_stabs = [
            (  # 1:  d=3 - or=Orientation.VERTICAL - m_dist1 - pos=(0, 0) - pauli=XX
                [Stabilizer("XX", ((0, 2, 0), (0, 3, 0)), ancilla_qubits=((0, 3, 1),))],
                [],
            ),
            (  # 2: d=3 - or=Orientation.VERTICAL - m_dist1 - pos=(0, 0) - pauli=ZZ
                [Stabilizer("ZZ", ((0, 2, 0), (0, 3, 0)), ancilla_qubits=((0, 3, 1),))],
                [],
            ),
            (  # 3: d=3 - or=Orientation.VERTICAL - m_dist1 - pos=(2, 3) - pauli=XX
                [Stabilizer("XX", ((2, 5, 0), (2, 6, 0)), ancilla_qubits=((2, 6, 1),))],
                [],
            ),
            (  # 4: d=3 - or=Orientation.VERTICAL - m_dist1 - pos=(2, 3) - pauli=ZZ
                [Stabilizer("ZZ", ((2, 5, 0), (2, 6, 0)), ancilla_qubits=((2, 6, 1),))],
                [],
            ),
            (  # 5: d=3 - or=Orientation.VERTICAL - m_dist4 - pos=(0, 0) - pauli=XX
                [
                    Stabilizer(
                        "XX", ((0, 2, 0), (0, 3, 0)), ancilla_qubits=((0, 3, 1),)
                    ),
                    Stabilizer(
                        "XX", ((0, 4, 0), (0, 5, 0)), ancilla_qubits=((0, 5, 1),)
                    ),
                ],
                [
                    Stabilizer(
                        "XX", ((2, 3, 0), (2, 4, 0)), ancilla_qubits=((3, 4, 1),)
                    ),
                    Stabilizer(
                        "XX", ((2, 5, 0), (2, 6, 0)), ancilla_qubits=((3, 6, 1),)
                    ),
                ],
            ),
            (  # 6: d=3 - or=Orientation.VERTICAL - m_dist4 - pos=(0, 0) - pauli=ZZ
                [
                    Stabilizer(
                        "ZZ", ((0, 2, 0), (0, 3, 0)), ancilla_qubits=((0, 3, 1),)
                    ),
                    Stabilizer(
                        "ZZ", ((0, 4, 0), (0, 5, 0)), ancilla_qubits=((0, 5, 1),)
                    ),
                ],
                [
                    Stabilizer(
                        "ZZ", ((2, 3, 0), (2, 4, 0)), ancilla_qubits=((3, 4, 1),)
                    ),
                    Stabilizer(
                        "ZZ", ((2, 5, 0), (2, 6, 0)), ancilla_qubits=((3, 6, 1),)
                    ),
                ],
            ),
            (  # 7: d=3 - or=Orientation.VERTICAL - m_dist4 - pos=(2, 3) - pauli=XX
                [
                    Stabilizer(
                        "XX", ((2, 5, 0), (2, 6, 0)), ancilla_qubits=((2, 6, 1),)
                    ),
                    Stabilizer(
                        "XX", ((2, 7, 0), (2, 8, 0)), ancilla_qubits=((2, 8, 1),)
                    ),
                ],
                [
                    Stabilizer(
                        "XX", ((4, 6, 0), (4, 7, 0)), ancilla_qubits=((5, 7, 1),)
                    ),
                    Stabilizer(
                        "XX", ((4, 8, 0), (4, 9, 0)), ancilla_qubits=((5, 9, 1),)
                    ),
                ],
            ),
            (  # 8: d=3 - or=Orientation.VERTICAL - m_dist4 - pos=(2, 3) - pauli=ZZ
                [
                    Stabilizer(
                        "ZZ", ((2, 5, 0), (2, 6, 0)), ancilla_qubits=((2, 6, 1),)
                    ),
                    Stabilizer(
                        "ZZ", ((2, 7, 0), (2, 8, 0)), ancilla_qubits=((2, 8, 1),)
                    ),
                ],
                [
                    Stabilizer(
                        "ZZ", ((4, 6, 0), (4, 7, 0)), ancilla_qubits=((5, 7, 1),)
                    ),
                    Stabilizer(
                        "ZZ", ((4, 8, 0), (4, 9, 0)), ancilla_qubits=((5, 9, 1),)
                    ),
                ],
            ),
            (  # 9: d=3 - or=Orientation.HORIZONTAL - m_dist1 - pos=(0, 0) - pauli=XX
                [],
                [Stabilizer("XX", ((3, 2, 0), (2, 2, 0)), ancilla_qubits=((3, 3, 1),))],
            ),
            (  # 10: d=3 - or=Orientation.HORIZONTAL - m_dist1 - pos=(0, 0) - pauli=ZZ
                [],
                [Stabilizer("ZZ", ((3, 2, 0), (2, 2, 0)), ancilla_qubits=((3, 3, 1),))],
            ),
            (  # 11 - d=3 - or=Orientation.HORIZONTAL - m_dist1 - pos=(2, 3) - pauli=XX
                [],
                [Stabilizer("XX", ((5, 5, 0), (4, 5, 0)), ancilla_qubits=((5, 6, 1),))],
            ),
            (  # 12 - d=3 - or=Orientation.HORIZONTAL - m_dist1 - pos=(2, 3) - pauli=ZZ
                [],
                [Stabilizer("ZZ", ((5, 5, 0), (4, 5, 0)), ancilla_qubits=((5, 6, 1),))],
            ),
            (  # 13 - d=3 - or=Orientation.HORIZONTAL - m_dist4 - pos=(0, 0) - pauli=XX
                [
                    Stabilizer(
                        "XX", ((4, 0, 0), (3, 0, 0)), ancilla_qubits=((4, 0, 1),)
                    ),
                    Stabilizer(
                        "XX", ((6, 0, 0), (5, 0, 0)), ancilla_qubits=((6, 0, 1),)
                    ),
                ],
                [
                    Stabilizer(
                        "XX", ((3, 2, 0), (2, 2, 0)), ancilla_qubits=((3, 3, 1),)
                    ),
                    Stabilizer(
                        "XX", ((5, 2, 0), (4, 2, 0)), ancilla_qubits=((5, 3, 1),)
                    ),
                ],
            ),
            (  # 14 - d=3 - or=Orientation.HORIZONTAL - m_dist4 - pos=(0, 0) - pauli=ZZ
                [
                    Stabilizer(
                        "ZZ", ((4, 0, 0), (3, 0, 0)), ancilla_qubits=((4, 0, 1),)
                    ),
                    Stabilizer(
                        "ZZ", ((6, 0, 0), (5, 0, 0)), ancilla_qubits=((6, 0, 1),)
                    ),
                ],
                [
                    Stabilizer(
                        "ZZ", ((3, 2, 0), (2, 2, 0)), ancilla_qubits=((3, 3, 1),)
                    ),
                    Stabilizer(
                        "ZZ", ((5, 2, 0), (4, 2, 0)), ancilla_qubits=((5, 3, 1),)
                    ),
                ],
            ),
            (  # 15 - d=3 - or=Orientation.HORIZONTAL - m_dist4 - pos=(2, 3) - pauli=XX
                [
                    Stabilizer(
                        "XX", ((6, 3, 0), (5, 3, 0)), ancilla_qubits=((6, 3, 1),)
                    ),
                    Stabilizer(
                        "XX", ((8, 3, 0), (7, 3, 0)), ancilla_qubits=((8, 3, 1),)
                    ),
                ],
                [
                    Stabilizer(
                        "XX", ((5, 5, 0), (4, 5, 0)), ancilla_qubits=((5, 6, 1),)
                    ),
                    Stabilizer(
                        "XX", ((7, 5, 0), (6, 5, 0)), ancilla_qubits=((7, 6, 1),)
                    ),
                ],
            ),
            #  16 - d=3 - or=Orientation.HORIZONTAL - m_dist4 - pos=(2, 3) - pauli=ZZ
            (
                [
                    Stabilizer(
                        "ZZ", ((6, 3, 0), (5, 3, 0)), ancilla_qubits=((6, 3, 1),)
                    ),
                    Stabilizer(
                        "ZZ", ((8, 3, 0), (7, 3, 0)), ancilla_qubits=((8, 3, 1),)
                    ),
                ],
                [
                    Stabilizer(
                        "ZZ", ((5, 5, 0), (4, 5, 0)), ancilla_qubits=((5, 6, 1),)
                    ),
                    Stabilizer(
                        "ZZ", ((7, 5, 0), (6, 5, 0)), ancilla_qubits=((7, 6, 1),)
                    ),
                ],
            ),
            #  17 - d=4 - or=Orientation.VERTICAL - m_dist1 - pos=(0, 0) - pauli=XX
            ([], []),
            # 18 - d=4 - or=Orientation.VERTICAL - m_dist1 - pos=(0, 0) - pauli=ZZ
            ([], []),
            # 19 - d=4 - or=Orientation.VERTICAL - m_dist1 - pos=(2, 3) - pauli=XX
            ([], []),
            # 20 - d=4 - or=Orientation.VERTICAL - m_dist1 - pos=(2, 3) - pauli=ZZ
            ([], []),
            (  # 21 - d=4 - or=Orientation.VERTICAL - m_dist4 - pos=(0, 0) - pauli=XX
                [
                    Stabilizer(
                        "XX", ((0, 4, 0), (0, 5, 0)), ancilla_qubits=((0, 5, 1),)
                    ),
                    Stabilizer(
                        "XX", ((0, 6, 0), (0, 7, 0)), ancilla_qubits=((0, 7, 1),)
                    ),
                ],
                [
                    Stabilizer(
                        "XX", ((3, 4, 0), (3, 5, 0)), ancilla_qubits=((4, 5, 1),)
                    ),
                    Stabilizer(
                        "XX", ((3, 6, 0), (3, 7, 0)), ancilla_qubits=((4, 7, 1),)
                    ),
                ],
            ),
            (  # 22 - d=4 - or=Orientation.VERTICAL - m_dist4 - pos=(0, 0) - pauli=ZZ
                [
                    Stabilizer(
                        "ZZ", ((0, 4, 0), (0, 5, 0)), ancilla_qubits=((0, 5, 1),)
                    ),
                    Stabilizer(
                        "ZZ", ((0, 6, 0), (0, 7, 0)), ancilla_qubits=((0, 7, 1),)
                    ),
                ],
                [
                    Stabilizer(
                        "ZZ", ((3, 4, 0), (3, 5, 0)), ancilla_qubits=((4, 5, 1),)
                    ),
                    Stabilizer(
                        "ZZ", ((3, 6, 0), (3, 7, 0)), ancilla_qubits=((4, 7, 1),)
                    ),
                ],
            ),
            (  # 23 - d=4 - or=Orientation.VERTICAL - m_dist4 - pos=(2, 3) - pauli=XX
                [
                    Stabilizer(
                        "XX", ((2, 7, 0), (2, 8, 0)), ancilla_qubits=((2, 8, 1),)
                    ),
                    Stabilizer(
                        "XX", ((2, 9, 0), (2, 10, 0)), ancilla_qubits=((2, 10, 1),)
                    ),
                ],
                [
                    Stabilizer(
                        "XX", ((5, 7, 0), (5, 8, 0)), ancilla_qubits=((6, 8, 1),)
                    ),
                    Stabilizer(
                        "XX", ((5, 9, 0), (5, 10, 0)), ancilla_qubits=((6, 10, 1),)
                    ),
                ],
            ),
            (  # 24 - d=4 - or=Orientation.VERTICAL - m_dist4 - pos=(2, 3) - pauli=ZZ
                [
                    Stabilizer(
                        "ZZ", ((2, 7, 0), (2, 8, 0)), ancilla_qubits=((2, 8, 1),)
                    ),
                    Stabilizer(
                        "ZZ", ((2, 9, 0), (2, 10, 0)), ancilla_qubits=((2, 10, 1),)
                    ),
                ],
                [
                    Stabilizer(
                        "ZZ", ((5, 7, 0), (5, 8, 0)), ancilla_qubits=((6, 8, 1),)
                    ),
                    Stabilizer(
                        "ZZ", ((5, 9, 0), (5, 10, 0)), ancilla_qubits=((6, 10, 1),)
                    ),
                ],
            ),
            (  # 25 - d=4 - or=Orientation.HORIZONTAL - m_dist1 - pos=(0, 0) - pauli=XX
                [Stabilizer("XX", ((4, 0, 0), (3, 0, 0)), ancilla_qubits=((4, 0, 1),))],
                [Stabilizer("XX", ((4, 3, 0), (3, 3, 0)), ancilla_qubits=((4, 4, 1),))],
            ),
            (  # 26 - d=4 - or=Orientation.HORIZONTAL - m_dist1 - pos=(0, 0) - pauli=ZZ
                [Stabilizer("ZZ", ((4, 0, 0), (3, 0, 0)), ancilla_qubits=((4, 0, 1),))],
                [Stabilizer("ZZ", ((4, 3, 0), (3, 3, 0)), ancilla_qubits=((4, 4, 1),))],
            ),
            (  # 27 - d=4 - or=Orientation.HORIZONTAL - m_dist1 - pos=(2, 3) - pauli=XX
                [Stabilizer("XX", ((6, 3, 0), (5, 3, 0)), ancilla_qubits=((6, 3, 1),))],
                [Stabilizer("XX", ((6, 6, 0), (5, 6, 0)), ancilla_qubits=((6, 7, 1),))],
            ),
            (  # 28 - d=4 - or=Orientation.HORIZONTAL - m_dist1 - pos=(2, 3) - pauli=ZZ
                [Stabilizer("ZZ", ((6, 3, 0), (5, 3, 0)), ancilla_qubits=((6, 3, 1),))],
                [Stabilizer("ZZ", ((6, 6, 0), (5, 6, 0)), ancilla_qubits=((6, 7, 1),))],
            ),
            (  # 29 - d=4 - or=Orientation.HORIZONTAL - m_dist4 - pos=(0, 0) - pauli=XX
                [
                    Stabilizer(
                        "XX", ((4, 0, 0), (3, 0, 0)), ancilla_qubits=((4, 0, 1),)
                    ),
                    Stabilizer(
                        "XX", ((6, 0, 0), (5, 0, 0)), ancilla_qubits=((6, 0, 1),)
                    ),
                ],
                [
                    Stabilizer(
                        "XX", ((4, 3, 0), (3, 3, 0)), ancilla_qubits=((4, 4, 1),)
                    ),
                    Stabilizer(
                        "XX", ((6, 3, 0), (5, 3, 0)), ancilla_qubits=((6, 4, 1),)
                    ),
                ],
            ),
            (  # 30 - d=4 - or=Orientation.HORIZONTAL - m_dist4 - pos=(0, 0) - pauli=ZZ
                [
                    Stabilizer(
                        "ZZ", ((4, 0, 0), (3, 0, 0)), ancilla_qubits=((4, 0, 1),)
                    ),
                    Stabilizer(
                        "ZZ", ((6, 0, 0), (5, 0, 0)), ancilla_qubits=((6, 0, 1),)
                    ),
                ],
                [
                    Stabilizer(
                        "ZZ", ((4, 3, 0), (3, 3, 0)), ancilla_qubits=((4, 4, 1),)
                    ),
                    Stabilizer(
                        "ZZ", ((6, 3, 0), (5, 3, 0)), ancilla_qubits=((6, 4, 1),)
                    ),
                ],
            ),
            (  # 31 - d=4 - or=Orientation.HORIZONTAL - m_dist4 - pos=(2, 3) - pauli=XX
                [
                    Stabilizer(
                        "XX", ((6, 3, 0), (5, 3, 0)), ancilla_qubits=((6, 3, 1),)
                    ),
                    Stabilizer(
                        "XX", ((8, 3, 0), (7, 3, 0)), ancilla_qubits=((8, 3, 1),)
                    ),
                ],
                [
                    Stabilizer(
                        "XX", ((6, 6, 0), (5, 6, 0)), ancilla_qubits=((6, 7, 1),)
                    ),
                    Stabilizer(
                        "XX", ((8, 6, 0), (7, 6, 0)), ancilla_qubits=((8, 7, 1),)
                    ),
                ],
            ),
            (
                [
                    Stabilizer(
                        "ZZ", ((6, 3, 0), (5, 3, 0)), ancilla_qubits=((6, 3, 1),)
                    ),
                    Stabilizer(
                        "ZZ", ((8, 3, 0), (7, 3, 0)), ancilla_qubits=((8, 3, 1),)
                    ),
                ],
                [
                    Stabilizer(
                        "ZZ", ((6, 6, 0), (5, 6, 0)), ancilla_qubits=((6, 7, 1),)
                    ),
                    Stabilizer(
                        "ZZ",
                        ((8, 6, 0), (7, 6, 0)),
                        ancilla_qubits=((8, 7, 1),),
                    ),
                ],
            ),
        ]

        for i, (
            block_distance,
            orientation,
            merge_distance,
            position,
            pauli,
        ) in enumerate(
            product(
                [3, 4],
                [Orientation.VERTICAL, Orientation.HORIZONTAL],
                [1, 4],
                [(0, 0), (2, 3)],
                ["XX", "ZZ"],
            )
        ):
            is_horizontal = orientation == Orientation.HORIZONTAL
            upper_left_filling = (
                position[0] + (block_distance - 1) * is_horizontal,
                position[1] + (block_distance - 1) * (not is_horizontal),
            )
            block_1 = RotatedSurfaceCode.create(
                dx=block_distance,
                dz=block_distance,
                lattice=lattice,
                unique_label="q1",
                position=position,
            )
            block2_position = (
                position[0] + (block_distance + merge_distance) * (is_horizontal),
                position[1] + (block_distance + merge_distance) * (not is_horizontal),
            )
            # The second block has weight_2_stab_is_first_row if the distance between
            # the two blocks is even (for block 1 it's alway True)
            block_2 = RotatedSurfaceCode.create(
                dx=block_distance,
                dz=block_distance,
                lattice=lattice,
                unique_label="q2",
                position=block2_position,
                weight_2_stab_is_first_row=(
                    not ((block_distance + merge_distance) % 2)
                ),
            )
            resulting_stabs = create_merge_2_body_stabilizers(
                blocks=(block_1, block_2),
                merge_orientation=orientation,
                merge_distance=merge_distance,
                filling_upper_left_qubit=upper_left_filling,
                pauli=pauli,
            )

            # Compare the resulting stabilizers
            self.assertEqual(resulting_stabs, expected_stabs[i])

    def test_merge_stabilizers(self):
        """
        Test that the stabilizers are merged correctly.
        """
        # 1 - Test a merge with a Z boundary
        blocks = (self.block_1, self.block_2)
        new_block_stabs, old_2body_to_lengthen, new_4body_replacing, _ = (
            merge_stabilizers(blocks, merge_is_horizontal=True)
        )

        mock_block = RotatedSurfaceCode.create(
            dx=7,
            dz=3,
            lattice=self.lattice,
            unique_label="mock",
        )
        expected_stabs = mock_block.stabilizers
        # Compare the new block stabilizrs
        self.assertEqual(set(new_block_stabs), set(expected_stabs))

        expected_2body_to_lengthen = [
            Stabilizer("XX", ((2, 1, 0), (2, 2, 0)), ancilla_qubits=((3, 2, 1),)),
            Stabilizer("XX", ((4, 0, 0), (4, 1, 0)), ancilla_qubits=((4, 1, 1),)),
        ]
        # Compare the 2-body stabilizers that are modified, order matters
        self.assertEqual(old_2body_to_lengthen, expected_2body_to_lengthen)

        expected_4body_replacing = [
            Stabilizer(
                pauli="XXXX",
                data_qubits=((3, 1, 0), (3, 2, 0), (2, 1, 0), (2, 2, 0)),
                ancilla_qubits=((3, 2, 1),),
            ),
            Stabilizer(
                pauli="XXXX",
                data_qubits=((4, 0, 0), (4, 1, 0), (3, 0, 0), (3, 1, 0)),
                ancilla_qubits=((4, 1, 1),),
            ),
        ]
        # Compare the resulting 4-body stabilizers, order matters
        self.assertEqual(new_4body_replacing, expected_4body_replacing)

        # 2 - Test a merge with a Z boundary
        block_1 = RotatedSurfaceCode.create(
            dx=3,
            dz=3,
            lattice=self.lattice,
            unique_label="q1",
            x_boundary=Orientation.VERTICAL,
        )
        block_2 = RotatedSurfaceCode.create(
            dx=3,
            dz=3,
            lattice=self.lattice,
            unique_label="q1",
            position=(4, 0),
            x_boundary=Orientation.VERTICAL,
        )
        blocks = (block_1, block_2)
        new_block_stabs, old_2body_to_lengthen, new_4body_replacing, _ = (
            merge_stabilizers(blocks, merge_is_horizontal=True)
        )

        mock_block = RotatedSurfaceCode.create(
            dx=7,
            dz=3,
            lattice=self.lattice,
            unique_label="mock",
            x_boundary=Orientation.VERTICAL,  # This means default X schedule is "Z"
        )
        expected_stabs = mock_block.stabilizers
        # Compare the new block stabilizers
        self.assertEqual(set(new_block_stabs), set(expected_stabs))

        expected_2body_to_lengthen = [
            Stabilizer("ZZ", ((2, 1, 0), (2, 2, 0)), ancilla_qubits=((3, 2, 1),)),
            Stabilizer("ZZ", ((4, 0, 0), (4, 1, 0)), ancilla_qubits=((4, 1, 1),)),
        ]
        # Compare the 2-body stabilizers that are modified, order matters
        self.assertEqual(old_2body_to_lengthen, expected_2body_to_lengthen)

        # IMPORTANT: the schedule should be the Z-schedule here (different than the
        # X-schedule) of the previous test
        expected_4body_replacing = [
            Stabilizer(
                pauli="ZZZZ",
                data_qubits=((3, 1, 0), (3, 2, 0), (2, 1, 0), (2, 2, 0)),
                ancilla_qubits=((3, 2, 1),),
            ),
            Stabilizer(
                pauli="ZZZZ",
                data_qubits=((4, 0, 0), (4, 1, 0), (3, 0, 0), (3, 1, 0)),
                ancilla_qubits=((4, 1, 1),),
            ),
        ]
        # Compare the resulting 4-body stabilizers, order matters
        self.assertEqual(new_4body_replacing, expected_4body_replacing)

    def test_find_data_qubits_between(self):
        """
        Test that the function returns the right qubits.
        """
        # 1 - Test a horizontal alignment
        qubits = find_data_qubits_between((0, 0, 0), (0, 2, 0))
        expected_qubits = [(0, 1, 0)]
        self.assertEqual(qubits, expected_qubits)

        # 2 - Test a vertical alignment
        qubits = find_data_qubits_between((0, 0, 0), (5, 0, 0))
        expected_qubits = [(i, 0, 0) for i in range(1, 5)]
        self.assertEqual(qubits, expected_qubits)

        # 3 - Test the case where the two qubits are the same
        qubits = find_data_qubits_between((0, 0, 0), (0, 0, 0))
        expected_qubits = []
        self.assertEqual(qubits, expected_qubits)

        # 4 - Test the case where the two qubits are not aligned
        with self.assertRaises(ValueError) as cm:
            find_data_qubits_between((0, 0, 0), (1, 1, 0))
        self.assertIn(
            "The qubits are not aligned horizontally or vertically.", str(cm.exception)
        )

    def test_merge_logical_operators(self):
        """
        Test that the logical operators are merged correctly.
        """
        # 1 - Test for a standard merge operation
        base_step = deepcopy(self.base_step)
        new_step, new_x, cbits_x, new_z, cbits_z = merge_logical_operators(
            base_step,
            (self.block_1, self.block_2),
            qubits_to_reset=((3, 0, 0), (3, 1, 0), (3, 2, 0)),
            merge_is_horizontal=True,
        )

        # Test the generated logical operators
        expected_log_x = PauliOperator("XXXXXXX", tuple((i, 0, 0) for i in range(7)))
        expected_log_z = PauliOperator("ZZZ", tuple((0, i, 0) for i in range(3)))
        self.assertEqual(new_x, expected_log_x)
        self.assertEqual(new_z, expected_log_z)
        # Test the logical evolution was recorded
        expected_x_evol = {
            new_x.uuid: (
                self.block_1.logical_x_operators[0].uuid,
                self.block_2.logical_x_operators[0].uuid,
            )
        }
        self.assertEqual(new_step.logical_x_evolution, expected_x_evol)
        self.assertNotIn(
            new_z.uuid, new_step.logical_z_evolution.keys()
        )  # No Z evolution

        # 2 - Test for a merge where the logical operators are moved
        rsc1 = RotatedSurfaceCode.create(
            dx=3,
            dz=3,
            lattice=self.lattice,
            unique_label="q1",
            logical_x_operator=PauliOperator("XXX", tuple((i, 1, 0) for i in range(3))),
            logical_z_operator=PauliOperator("ZZZ", tuple((1, i, 0) for i in range(3))),
        )
        rsc2 = RotatedSurfaceCode.create(
            dx=3,
            dz=3,
            lattice=self.lattice,
            unique_label="q2",
            position=(4, 0),
        )  # Use default logical operators for this block
        base_step = InterpretationStep(
            block_history=((rsc1, rsc2),),
            syndromes=tuple(
                Syndrome(
                    stabilizer=stab.uuid,
                    measurements=((f"c_{stab.ancilla_qubits[0]}", 0),),
                    block=block.uuid,
                    round=5,
                    corrections=(),
                )
                for block in (rsc1, rsc2)
                for stab in block.stabilizers
            ),
        )
        new_step, new_x, cbits_x, new_z, cbits_z = merge_logical_operators(
            base_step,
            (rsc1, rsc2),
            qubits_to_reset=((3, 0, 0), (3, 1, 0), (3, 2, 0)),
            merge_is_horizontal=True,
        )
        # Test the logical operators
        expected_log_x = PauliOperator("XXXXXXX", tuple((i, 0, 0) for i in range(7)))
        # This time, Z is taken from block 1, which is not in the leftmost position
        expected_log_z = PauliOperator("ZZZ", tuple((1, i, 0) for i in range(3)))
        self.assertEqual(new_x, expected_log_x)
        self.assertEqual(new_z, expected_log_z)
        # Test the logical evolution was recorded
        weight_4_eq_stab = next(
            stab
            for stab in rsc1.stabilizers
            if ((2, 0, 0), (2, 1, 0), (1, 0, 0), (1, 1, 0)) == stab.data_qubits
        )
        weight_2_eq_stab = next(
            stab
            for stab in rsc1.stabilizers
            if ((0, 0, 0), (0, 1, 0)) == stab.data_qubits
        )
        expected_x_evol = {
            new_x.uuid: (
                rsc1.logical_x_operators[0].uuid,
                rsc2.logical_x_operators[0].uuid,
                weight_4_eq_stab.uuid,
                weight_2_eq_stab.uuid,
            )
        }
        self.assertEqual(new_step.logical_x_evolution, expected_x_evol)
        self.assertNotIn(
            new_z.uuid, new_step.logical_z_evolution.keys()
        )  # No Z evolution

        # Test for the value of the given Cbits
        expected_cbits = {
            "X": tuple(
                (f"c_{stab.ancilla_qubits[0]}", 0)
                for stab in (weight_4_eq_stab, weight_2_eq_stab)
            ),
            "Z": (),
        }
        self.assertEqual(cbits_x, expected_cbits["X"])
        self.assertEqual(cbits_z, expected_cbits["Z"])

    def test_create_syndromes(self):  # pylint: disable=too-many-locals
        """Test the generation of syndromes for multiple configurations of merge"""
        for merge_orientation, merge_distance, x_boundary in product(
            [Orientation.HORIZONTAL, Orientation.VERTICAL],
            [2, 3, 4, 5],
            [Orientation.HORIZONTAL, Orientation.VERTICAL],
        ):
            is_horizontal = merge_orientation == Orientation.HORIZONTAL
            block_1 = RotatedSurfaceCode.create(
                dx=3,
                dz=3,
                lattice=self.lattice,
                x_boundary=x_boundary,
            )
            block_2 = RotatedSurfaceCode.create(
                dx=3,
                dz=3,
                lattice=self.lattice,
                x_boundary=x_boundary,
                unique_label="q2",
                weight_2_stab_is_first_row=(merge_distance % 2 == 0),
                position=(
                    block_1.upper_left_qubit[0]
                    + (block_1.size[0] + merge_distance - 1) * is_horizontal,
                    block_1.upper_left_qubit[1]
                    + (block_1.size[1] + merge_distance - 1) * (not is_horizontal),
                ),
            )
            # Create the merged block for reference
            merged_block = RotatedSurfaceCode.create(
                dx=block_1.size[0]
                + (block_2.size[0] + merge_distance - 1) * is_horizontal,
                dz=block_1.size[1]
                + (block_2.size[1] + merge_distance - 1) * (not is_horizontal),
                lattice=self.lattice,
                x_boundary=x_boundary,
            )
            # Reset the qubits that are in between the two blocks
            qubits_to_reset = tuple(
                q
                for q in merged_block.qubits
                if q not in block_1.qubits + block_2.qubits
            )

            # The reset preserves the logical that is merged (if the merge is horizontal
            # the horizontal logical operator determines the type of reset)
            reset_type = "X" if x_boundary == merge_orientation else "Z"
            new_stabilizers = tuple(
                stab
                for stab in merged_block.stabilizers
                if stab not in (block_1.stabilizers + block_2.stabilizers)
            )
            new_stabs_increased_weight = tuple(
                stab
                for stab in new_stabilizers
                for q in stab.ancilla_qubits
                if q in (block_1.ancilla_qubits + block_2.ancilla_qubits)
            )
            interpretation_step = InterpretationStep(
                block_history=((block_1, block_2),)
            )
            # Generate the merge syndromes
            syndromes = create_syndromes(
                interpretation_step,
                qubits_to_reset,
                reset_type,
                new_stabilizers,
                new_stabs_increased_weight,
                merged_block,
            )
            # Generate the expected syndromes
            stabs_reset = tuple(
                stab
                for stab in new_stabilizers
                if (
                    set(stab.pauli) == {reset_type}
                    and all(
                        q not in (block_1.qubits + block_2.qubits)
                        for q in stab.ancilla_qubits
                    )
                )
            )
            expected_syndromes = tuple(
                Syndrome(stab.uuid, (), merged_block.uuid, 0) for stab in stabs_reset
            )
            self.assertEqual(expected_syndromes, syndromes)

    def test_merge(self):
        """
        Test a standard merge workflow.
        """
        lattice = Lattice.square_2d((10, 10))
        initial_step = deepcopy(self.base_step)
        block_1 = RotatedSurfaceCode.create(
            dx=3,
            dz=3,
            lattice=lattice,
            unique_label="standard1",
        )
        block_2 = RotatedSurfaceCode.create(
            dx=3,
            dz=3,
            position=(4, 0),
            lattice=lattice,
            unique_label="standard2",
        )
        expected_new_block = RotatedSurfaceCode.create(
            dx=7,
            dz=3,
            lattice=lattice,
            unique_label="merged",
        )
        initial_step.block_history = ((block_1, block_2),)
        initial_step.logical_x_operator_updates = {
            block_1.logical_x_operators[0].uuid: (("dummy_X_left", 0),),
            block_2.logical_x_operators[0].uuid: (("dummy_X_right", 0),),
        }
        initial_step.logical_z_operator_updates = {
            block_1.logical_z_operators[0].uuid: (("dummy_Z_left", 0),),
            block_2.logical_z_operators[0].uuid: (("dummy_Z_right", 0),),
        }
        merge_op = Merge(["standard2", "standard1"], "merged")
        interpretation_step = merge(
            initial_step, merge_op, same_timeslice=False, debug_mode=True
        )

        # Check that the block is correctly generated
        merged_block = interpretation_step.get_block("merged")
        self.assertEqual(merged_block, expected_new_block)

        # Check that the block history is correctly updated
        self.assertEqual(
            interpretation_step.block_history,
            ((block_1, block_2), (merged_block,)),
        )

        # Check that the circuit is correctly generated
        qubits_to_measure = [(3, 0, 0), (3, 1, 0), (3, 2, 0)]
        qubit_channels = [Channel("quantum", label=str(q)) for q in qubits_to_measure]
        expected_circuit = Circuit(
            name="merge standard1 and standard2 into merged",
            circuit=[
                [Circuit(name="Reset_+", channels=[q]) for q in qubit_channels],
            ],
        )
        self.assertEqual(
            interpretation_step.intermediate_circuit_sequence[0][0], expected_circuit
        )

        # Check that the stabilizer_evolution is correctly generated
        initial_2body = [
            stab
            for stab in block_1.stabilizers + block_2.stabilizers
            if stab.data_qubits in [((2, 1, 0), (2, 2, 0)), ((4, 0, 0), (4, 1, 0))]
        ]
        final_4body = [
            stab
            for stab in merged_block.stabilizers
            if stab.data_qubits
            in [
                ((3, 1, 0), (3, 2, 0), (2, 1, 0), (2, 2, 0)),
                ((4, 0, 0), (4, 1, 0), (3, 0, 0), (3, 1, 0)),
            ]
        ]
        expected_stabilizer_evolution = {
            final.uuid: (initial.uuid,)  # 1-to-1 mapping (not the general case)
            for final, initial in zip(final_4body, initial_2body, strict=True)
        }
        self.assertEqual(
            interpretation_step.stabilizer_evolution, expected_stabilizer_evolution
        )

        # Check that the stabilizer_update is correctly generated and is empty
        self.assertEqual(interpretation_step.stabilizer_updates, {})

        # Check that the logical evolution is correctly propagated
        expected_x_evol = {
            merged_block.logical_x_operators[0].uuid: (
                block_1.logical_x_operators[0].uuid,
                block_2.logical_x_operators[0].uuid,
            )
        }
        self.assertEqual(interpretation_step.logical_x_evolution, expected_x_evol)
        self.assertNotIn(
            merged_block.logical_z_operators[0].uuid,
            interpretation_step.logical_z_evolution.keys(),
        )  # No Z evolution

        # Check that the logical update is correctly generated
        self.assertEqual(
            interpretation_step.logical_x_operator_updates[
                merged_block.logical_x_operators[0].uuid
            ],
            (
                ("dummy_X_left", 0),
                ("dummy_X_right", 0),
            ),
        )
        self.assertEqual(
            interpretation_step.logical_z_operator_updates[
                merged_block.logical_z_operators[0].uuid
            ],
            (("dummy_Z_left", 0),),
        )

        # Check that there are no new syndromes generated because there are no new
        # deterministic stabilizers
        self.assertEqual(interpretation_step.syndromes, initial_step.syndromes)

    def test_all_merge_configurations(self):
        """Test all configurations of blocks for the merge operation."""
        base_position = (1, 2)
        # q1_w2stab is the boolean for weight_2stab_is_first_row fo the first block
        for (
            merge_orientation,
            block_size,
            n_rows_between_blocks,
            block_order,
            q1_w2stab,
        ) in product(
            [Orientation.VERTICAL, Orientation.HORIZONTAL],  # Merge orientation
            [3, 4],  # Even and odd dimensions
            [4, 5],  # Even and odd separation of blocks
            [1, -1],  # Change the order in which we feed the blocks in (same result)
            [True, False],  # Different values of weight_2_stab_is_first_row for block 1
        ):
            is_horizontal = merge_orientation == Orientation.HORIZONTAL
            lattice = Lattice.square_2d(
                lattice_size=(
                    base_position[0]
                    + block_size
                    + (block_size + n_rows_between_blocks + 1) * is_horizontal
                    + 1,
                    base_position[1]
                    + block_size
                    + (block_size + n_rows_between_blocks + 1) * (not is_horizontal)
                    + 1,
                )
            )
            # The second block has
            q2_w2stab = (
                q1_w2stab
                if ((block_size + n_rows_between_blocks) % 2 == 0)
                else not q1_w2stab
            )

            q1 = RotatedSurfaceCode.create(
                dx=block_size,
                dz=block_size,
                position=base_position,
                lattice=lattice,
                unique_label="q1",
                weight_2_stab_is_first_row=q1_w2stab,
            )
            q2 = RotatedSurfaceCode.create(
                dx=block_size,
                dz=block_size,
                position=(
                    base_position[0]
                    + (block_size + n_rows_between_blocks) * is_horizontal,
                    base_position[1]
                    + (block_size + n_rows_between_blocks) * (not is_horizontal),
                ),
                lattice=lattice,
                unique_label="q2",
                weight_2_stab_is_first_row=q2_w2stab,
            )
            eka = Eka(
                lattice=lattice,
                blocks=[q1, q2][::block_order],
                operations=[
                    [Merge(("q1", "q2"), "q1_q2")],
                ],
            )
            step_1 = interpret_eka(eka)

            expected_final_block = RotatedSurfaceCode.create(
                dx=block_size + (n_rows_between_blocks + block_size) * is_horizontal,
                dz=block_size
                + (n_rows_between_blocks + block_size) * (not is_horizontal),
                position=base_position,
                lattice=lattice,
                unique_label="q1_q2",
                weight_2_stab_is_first_row=q1_w2stab,
            )

            self.assertEqual(
                step_1.get_block("q1_q2"),
                expected_final_block,
            )


if __name__ == "__main__":
    unittest.main()
