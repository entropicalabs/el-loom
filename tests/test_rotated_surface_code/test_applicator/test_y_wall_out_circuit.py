"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

# pylint: disable=duplicate-code
import unittest

from loom.eka import Circuit, Channel, Lattice
from loom.eka.utilities import Orientation, Direction, DiagonalDirection
from loom.interpreter import InterpretationStep, Syndrome

from loom_rotated_surface_code.code_factory import RotatedSurfaceCode
from loom_rotated_surface_code.applicator.move_corners import move_corners
from loom_rotated_surface_code.applicator.move_block import (
    direction_to_coord,
    DetailedSchedule,
)
from loom_rotated_surface_code.applicator.y_wall_out import y_wall_out
from loom_rotated_surface_code.applicator.y_wall_out_circuit import (
    map_stabilizer_schedule,
)

# pylint: disable=duplicate-code


class TestRotatedSurfaceCodeYWallOut(unittest.TestCase):
    """
    Class for Tests of the Y wall out operation circuit generation.
    """

    def setUp(self):
        self.square_2d_lattice = Lattice.square_2d((10, 20))

        # MAKE A VERTICAL BLOCK
        # distance: 3, top-left bulk stabilizer: Z
        self.big_block_v3z = RotatedSurfaceCode.create(
            dx=3,
            dz=6,
            lattice=self.square_2d_lattice,
            unique_label="q1",
            weight_2_stab_is_first_row=False,
            x_boundary=Orientation.VERTICAL,
        )
        # Get the twisted block v3z by moving the topological corner instead
        self.base_step = InterpretationStep(
            block_history=((self.big_block_v3z,),),
            syndromes=tuple(
                Syndrome(
                    stabilizer=stab.uuid,
                    measurements=((f"c_{stab.ancilla_qubits[0]}", 0),),
                    block=self.big_block_v3z.unique_label,
                    round=0,
                    corrections=[],
                )
                for stab in self.big_block_v3z.stabilizers
            ),
        )
        int_step = move_corners(
            interpretation_step=self.base_step,
            block=self.big_block_v3z,
            corner_args=(((0, 5, 0), Direction.TOP, 2),),
            same_timeslice=False,
            debug_mode=True,
        )
        self.twisted_rsc_block_v3z = int_step.get_block(self.big_block_v3z.unique_label)

        # The transformation of the block is as follows:
        #            X                                    X
        #    *(0,0) --- (1,0) --- (2,0)*          *(0,0) --- (1,0) --- (2,0)*
        #       |         |         |                |         |         |
        #       |    Z    |    X    |  Z             |    Z    |    X    |  Z
        #       |         |         |                |         |         |
        #     (0,1) --- (1,1) --- (2,1)            (0,1) --- (1,1) --- (2,1)
        #       |         |         |                |         |         |
        #    Z  |    X    |    Z    |             Z  |    X    |    Z    |
        #       |         |         |                |         |         |
        #     (0,2) --- (1,2) --- (2,2)     ->     (0,2) --- (1,2) --- (2,2)*
        #       |         |         |                |         |         |
        #       |    Z    |    X    |  Z             |    Z    |    X    |
        #       |         |         |                |         |         |
        #    *(0,3) --- (1,3) --- (2,3)            (0,3) --- (1,3) --- (2,3)
        #       |         |         |                |         |         |
        #       |    X    |    Z    |             Z  |    X    |    Z    |  X
        #       |         |         |                |         |         |
        #     (0,4) --- (1,4) --- (2,4)            (0,4) --- (1,4) --- (2,4)*
        #       |         |         |                     Z
        #    X  |    Z    |    X    |  Z
        #       |         |         |
        #     (0,5) --- (1,5) --- (2,5)*
        #            X

        self.qubit_channels = {
            q: Channel("quantum", f"{q}")
            for q in (
                # The block qubits
                list(self.twisted_rsc_block_v3z.qubits)
                # The data qubits on the right of the block
                + [(3, row, 0) for row in range(5)]
            )
            # The ancilla qubits on the right of the block
            + [(3, row, 1) for row in range(5)]
            # The ancilla qubits on the left of the block
            + [(0, row, 1) for row in range(5)]
        }

    def test_y_wall_out_circuit(self):
        """Test the circuit of the y_wall_out function for a specific case."""
        base_int_step = InterpretationStep(
            block_history=((self.twisted_rsc_block_v3z,),),
            syndromes=[
                Syndrome(
                    stabilizer=stab.uuid,
                    measurements=[("mock_register", i)],
                    block=self.twisted_rsc_block_v3z.uuid,
                    round=-1,
                )
                for i, stab in enumerate(self.twisted_rsc_block_v3z.stabilizers)
            ],
        )

        # Get the output circuit sequence
        interpreted_eka = y_wall_out(
            base_int_step,
            self.twisted_rsc_block_v3z,
            3,
            Orientation.HORIZONTAL,
            True,
        )
        output_circuit_seq = interpreted_eka.intermediate_circuit_sequence
        output_block = interpreted_eka.get_block(
            self.twisted_rsc_block_v3z.unique_label
        )

        # Define the expected circuit sequence
        expected_circuit_seq = [
            (self.wall_measurement_circuit(), self.hadamard_circuit()),
            (self.set_qubits_in_right_basis_circuit(),),
            (self.swap_qec_cnots_circuit(),),
            (
                self.measure_stabilizers_operation_circuit(),
                self.teleportation_finalization_circuit(),
            ),
            (self.initialization_circuit_final(),),
            (self.swap_qec_cnots_circuit_final(),),
            (
                self.measure_stabilizers_operation_circuit_final(),
                self.teleportation_finalization_circuit_final(),
            ),
        ]

        # Compare the output circuit sequence with the expected one
        self.assertEqual(
            Circuit(
                "output_circ",
                Circuit.construct_padded_circuit_time_sequence(output_circuit_seq),
            ),
            Circuit(
                "expected_circ",
                Circuit.construct_padded_circuit_time_sequence(expected_circuit_seq),
            ),
        )

        # Assert that the first syndromes of the block are measured correctly
        for stab in output_block.stabilizers:
            # Find round 0 Syndrome that measure the stabilizer
            first_syndrome = next(
                synd
                for synd in interpreted_eka.syndromes
                if synd.stabilizer == stab.uuid and synd.round == 0
            )

            # check that the measurement is from one of the data qubits of the
            # stabilizer
            self.assertEqual(len(first_syndrome.measurements), 1)

            # The function for a given direction set, finds the data qubit that is
            # in that mixed direction from the ancilla qubit
            def data_qubit_from_directions(stab, dirs):
                return tuple(
                    map(
                        sum,
                        zip(
                            stab.ancilla_qubits[0],
                            direction_to_coord(dirs, 1),
                            strict=True,
                        ),
                    )
                )

            possible_regs = [
                f"c_{data_qubit_from_directions(stab, dirs)}"
                for dirs in [
                    # Due to the recombination of the block, the data qubit measured
                    # is going to be either BOTTOM-RIGHT for the idling side or
                    # TOP-RIGHT for the Hadamard side
                    DiagonalDirection.BOTTOM_RIGHT,
                    DiagonalDirection.TOP_RIGHT,
                ]
            ]
            self.assertIn(first_syndrome.measurements[0][0], possible_regs)

        # Assert that there are no trivial detectors in the block, i.e. dependent on the
        # same syndrome
        for det in interpreted_eka.detectors:
            self.assertEqual(len(set(det.syndromes)), len(det.syndromes))

    def wall_measurement_circuit(self) -> Circuit:
        """Obtain the circuit that measures the qubits of the wall in the Y basis."""
        qubits_to_measure = [(0, 3, 0), (1, 3, 0), (2, 3, 0)]

        return Circuit(
            "wall_measurement",
            [
                [
                    Circuit(
                        "measure_y",
                        channels=[self.qubit_channels[q], Channel("classical")],
                    )
                    for q in qubits_to_measure
                ]
            ],
        )

    def hadamard_circuit(self) -> Circuit:
        """Obtain the circuit that applies the transversal Hadamard gate to the qubits
        beyond the wall."""
        qubits_to_had = [
            (0, 4, 0),
            (0, 5, 0),
            (1, 4, 0),
            (1, 5, 0),
            (2, 4, 0),
            (2, 5, 0),
        ]

        return Circuit(
            "hadamard beyond the wall",
            [[Circuit("h", channels=[self.qubit_channels[q]]) for q in qubits_to_had]],
        )

    def set_qubits_in_right_basis_circuit(self) -> Circuit:
        """Obtain the circuit that sets some of the qubits in the X basis."""

        qubits_to_set_in_x_basis = [
            # Wall data qubits
            (2, 3, 0),
            # Hadamard side data qubits
            (3, 4, 0),
            # Had side ancillas
            (2, 4, 1),
            # Had side teleportation qubit
            (3, 5, 1),
            (1, 5, 1),
            # Idle side ancillas
            (2, 1, 1),
            (2, 3, 1),
            (3, 2, 1),
            # Idle side teleportation qubits
            (1, 2, 1),
        ]
        qubits_to_set_in_z_basis = [
            # Wall data qubits
            (1, 3, 0),
            # Idle side data qubit
            (3, 1, 0),
            # Had side ancillas
            (2, 5, 1),
            (1, 4, 1),
            (3, 4, 1),
            (1, 3, 1),
            # Idle side ancillas
            (2, 2, 1),
            (3, 3, 1),
            (3, 1, 1),
            (1, 1, 1),
        ]

        return Circuit(
            "set_qubits_in_right_basis",
            [
                [
                    Circuit("reset_+", channels=[self.qubit_channels[q]])
                    for q in qubits_to_set_in_x_basis
                ]
                + [
                    Circuit("reset_0", channels=[self.qubit_channels[q]])
                    for q in qubits_to_set_in_z_basis
                ]
            ],
        )

    def swap_qec_cnots_circuit(self) -> Circuit:
        """
        Obtain the circuit that applies the CNOT gates to merge together the idling
        and Hadamard side of the block. This is for the case where the block is
        vertical of distance 3 and the top-left bulk stabilizer is Z.

        For clarity of the test, the CNOTs of the steps 1-4 are defined in the original
        block position and then shifted by a half step towards the bottom right.
        For example, the CNOT ((1, 0, 1), (0, 0, 0)) is shifted to
        ((1, 0, 0), (1, 1, 1)).
        """
        # Define the vector to find the bottom right ancilla/data from its top left
        # data/ancilla
        dq_to_anc_vec_br = direction_to_coord(DiagonalDirection.BOTTOM_RIGHT, 0)
        anc_to_dq_vec_br = direction_to_coord(DiagonalDirection.BOTTOM_RIGHT, 1)

        # DEFINE THE CNOT GATES FOR EACH TIME SLICE

        # TIME SLICE 0
        cnot_slice_0 = [
            # Hadamard side CNOTS (towards top-right)
            ((0, 4, 0), (1, 4, 1)),
            ((1, 5, 1), (0, 5, 0)),
            ((2, 4, 1), (1, 4, 0)),
            ((1, 5, 0), (2, 5, 1)),
            ((2, 4, 0), (3, 4, 1)),
            ((3, 5, 1), (2, 5, 0)),
            # Idle side CNOTS (towards bottom-right)
            ((0, 2, 0), (1, 3, 1)),
            ((1, 2, 1), (0, 1, 0)),
            ((0, 0, 0), (1, 1, 1)),
            ((2, 3, 1), (1, 2, 0)),
            ((1, 1, 0), (2, 2, 1)),
            ((2, 1, 1), (1, 0, 0)),
            ((2, 2, 0), (3, 3, 1)),
            ((3, 2, 1), (2, 1, 0)),
            ((2, 0, 0), (3, 1, 1)),
        ]

        # TIME SLICE 1 (define on)
        cnot_slice_1_original = [
            ((1, 0, 1), (0, 0, 0)),
            ((1, 0, 0), (1, 1, 1)),
            ((1, 2, 1), (0, 2, 0)),
            ((2, 1, 1), (1, 1, 0)),
            ((0, 4, 0), (0, 4, 1)),
            ((0, 1, 0), (0, 2, 1)),
            ((2, 1, 0), (2, 2, 1)),
            ((2, 3, 1), (2, 3, 0)),
            ((1, 3, 0), (1, 3, 1)),
        ]

        # TIME SLICE 2
        cnot_slice_2_original = [
            ((2, 1, 0), (3, 1, 1)),
            ((0, 1, 0), (1, 1, 1)),
            ((1, 2, 1), (1, 1, 0)),
            ((2, 1, 1), (2, 0, 0)),
            ((0, 4, 0), (1, 5, 1)),
            ((3, 4, 1), (2, 3, 0)),
            ((1, 4, 1), (0, 3, 0)),
            ((1, 3, 0), (2, 4, 1)),
            ((2, 3, 1), (2, 2, 0)),
            ((1, 2, 0), (1, 3, 1)),
        ]

        # TIME SLICE 3
        cnot_slice_3_original = [
            ((2, 0, 0), (3, 1, 1)),
            ((0, 0, 0), (1, 1, 1)),
            ((1, 2, 1), (0, 1, 0)),
            ((2, 1, 1), (1, 0, 0)),
            ((1, 4, 1), (1, 4, 0)),
            ((2, 4, 0), (2, 4, 1)),
            ((1, 2, 0), (2, 2, 1)),
            ((2, 3, 1), (1, 3, 0)),
            ((0, 3, 0), (1, 3, 1)),
        ]
        # TIME SLICE 4
        cnot_slice_4_original = [
            ((3, 4, 1), (2, 4, 0)),
            ((1, 4, 1), (0, 4, 0)),
            ((1, 4, 0), (2, 4, 1)),
            ((1, 1, 0), (2, 2, 1)),
            ((2, 3, 1), (1, 2, 0)),
            ((0, 2, 0), (1, 3, 1)),
        ]

        # Function to map the cnot pairs to the bottom right position
        def map_cnot_pair(cnot_pair):
            return tuple(
                tuple(
                    map(
                        sum,
                        zip(
                            q,
                            dq_to_anc_vec_br if q[2] == 0 else anc_to_dq_vec_br,
                            strict=True,
                        ),
                    )
                )
                for q in cnot_pair
            )

        # Map the cnot pairs to the bottom right position
        cnot_slices_1to4 = [
            [map_cnot_pair(cnot_pair) for cnot_pair in cnot_slice]
            for cnot_slice in [
                cnot_slice_1_original,
                cnot_slice_2_original,
                cnot_slice_3_original,
                cnot_slice_4_original,
            ]
        ]

        cnots = [cnot_slice_0] + cnot_slices_1to4

        return Circuit(
            "swap_qec_cnots",
            circuit=[
                [
                    Circuit(
                        "cx",
                        channels=[self.qubit_channels[q] for q in qubit_pair],
                    )
                    for qubit_pair in cnot_slice
                ]
                for cnot_slice in cnots
            ],
        )

    def teleportation_finalization_circuit(self) -> Circuit:
        """Obtain the circuit that finalizes the teleportation of qubits while merging
        the idling and Hadamard side of the block."""
        teleportation_info = [
            ((2, 0, 0), "measure_x"),
            ((0, 0, 0), "measure_x"),
            ((0, 1, 0), "measure_z"),
            ((2, 5, 0), "measure_z"),
            ((0, 5, 0), "measure_z"),
        ]

        teleportation_circ_seq = [[]]
        for dq, meas_op in teleportation_info:
            c_channel = Channel("classical")
            teleportation_circ_seq[0].append(
                Circuit(
                    meas_op,
                    channels=[self.qubit_channels[dq], c_channel],
                )
            )

        return Circuit(
            "teleportation",
            teleportation_circ_seq,
        )

    def measure_stabilizers_operation_circuit(self) -> Circuit:
        """Obtain the circuit that measures the stabilizers of the block. Note that
        it's only the final measurement operations."""
        mops_list = [
            ("measure_x", (1, 0, 0)),
            ("measure_z", (0, 2, 0)),
            ("measure_z", (3, 1, 0)),
            ("measure_z", (1, 1, 0)),
            ("measure_z", (2, 2, 0)),
            ("measure_x", (1, 2, 0)),
            ("measure_x", (2, 1, 0)),
            ("measure_z", (1, 5, 0)),
            ("measure_x", (3, 4, 0)),
            ("measure_z", (0, 4, 0)),
            ("measure_x", (1, 4, 0)),
            ("measure_x", (2, 3, 0)),
            ("measure_z", (1, 3, 0)),
            ("measure_z", (2, 4, 0)),
        ]

        classical_channels = [Channel("classical", f"c_{dq}_0") for _, dq in mops_list]

        return Circuit(
            "measure_stabilizers",
            [
                [
                    Circuit(
                        op,
                        channels=[self.qubit_channels[q], c_chan],
                    )
                    for (op, q), c_chan in zip(
                        mops_list, classical_channels, strict=True
                    )
                ],
            ],
        )

    def initialization_circuit_final(self) -> Circuit:
        """Obtain the circuit that initializes the qubits in the X basis after the
        merging of the idling and Hadamard side of the block to move the block to the
        final position."""
        qubits_to_init_in_x = [
            # This is going to be the ancilla after the moving of the block for a
            # boundary X stabilizer
            (1, 0, 1),
            # Ancillas of the block in the initial position
            # but data qubits after the moving of the block
            (0, 1, 0),
            (1, 0, 0),
            (2, 3, 0),
            (0, 3, 0),
            (1, 2, 0),
            # The final two are teleportation qubits
            (2, 1, 0),
            (1, 4, 0),
        ]

        qubits_to_init_in_z = [
            # These are going to be the ancillas after the moving of the block for a
            (0, 2, 1),
            (0, 4, 1),
            # Ancillas of the block in the initial position
            # but data qubits after the moving of the block
            (2, 0, 0),
            (0, 0, 0),
            (1, 1, 0),
            (0, 4, 0),
            (0, 2, 0),
            (1, 3, 0),
            # The final two are teleportation qubits
            (2, 2, 0),
            (2, 4, 0),
        ]

        return Circuit(
            name="Second initialization for swap_qec",
            circuit=[
                [
                    Circuit("reset_+", channels=[self.qubit_channels[q]])
                    for q in qubits_to_init_in_x
                ]
                + [
                    Circuit("reset_0", channels=[self.qubit_channels[q]])
                    for q in qubits_to_init_in_z
                ]
            ],
        )

    def swap_qec_cnots_circuit_final(self) -> Circuit:
        """Obtain the circuit that applies the CNOT gates to move the block to its final
        position.
        """
        cnots = [
            # TIMESLICE 0 (15 elements since it's 15 data qubits being moved)
            [
                ((1, 1, 1), (0, 0, 0)),
                ((0, 1, 0), (1, 2, 1)),
                ((1, 3, 1), (0, 2, 0)),
                ((0, 3, 0), (1, 4, 1)),
                ((1, 5, 1), (0, 4, 0)),
                ((1, 0, 0), (2, 1, 1)),
                ((2, 2, 1), (1, 1, 0)),
                ((1, 2, 0), (2, 3, 1)),
                ((2, 4, 1), (1, 3, 0)),
                ((1, 4, 0), (2, 5, 1)),
                ((3, 1, 1), (2, 0, 0)),
                ((2, 1, 0), (3, 2, 1)),
                ((3, 3, 1), (2, 2, 0)),
                ((2, 3, 0), (3, 4, 1)),
                ((3, 5, 1), (2, 4, 0)),
            ],
            # TIMESLICE 1
            [
                ((2, 1, 0), (3, 1, 1)),
                ((0, 1, 0), (1, 1, 1)),
                ((1, 2, 0), (2, 2, 1)),
                ((1, 2, 1), (1, 1, 0)),
                ((2, 1, 1), (2, 0, 0)),
                ((1, 4, 0), (1, 5, 1)),
                ((1, 4, 1), (1, 3, 0)),
                ((2, 3, 1), (2, 2, 0)),
                ((0, 3, 0), (1, 3, 1)),
                ((2, 3, 0), (2, 4, 1)),
            ],
            # TIMESLICE 2
            [
                ((1, 0, 1), (0, 0, 0)),
                ((0, 1, 0), (0, 2, 1)),
                ((1, 0, 0), (1, 1, 1)),
                ((2, 1, 0), (2, 2, 1)),
                ((1, 2, 1), (0, 2, 0)),
                ((2, 1, 1), (1, 1, 0)),
                ((3, 4, 1), (2, 4, 0)),
                ((0, 3, 0), (0, 4, 1)),
                ((1, 4, 1), (0, 4, 0)),
                ((2, 3, 1), (1, 3, 0)),
                ((1, 2, 0), (1, 3, 1)),
                ((1, 4, 0), (2, 4, 1)),
            ],
            # TIMESLICE 3
            [
                ((1, 0, 1), (1, 0, 0)),
                ((0, 2, 0), (0, 2, 1)),
                ((1, 1, 0), (1, 1, 1)),
                ((2, 2, 0), (2, 2, 1)),
                ((1, 2, 1), (1, 2, 0)),
                ((2, 1, 1), (2, 1, 0)),
                ((0, 4, 0), (0, 4, 1)),
                ((1, 4, 1), (1, 4, 0)),
                ((2, 3, 1), (2, 3, 0)),
                ((1, 3, 0), (1, 3, 1)),
                ((2, 4, 0), (2, 4, 1)),
            ],
        ]

        return Circuit(
            "swap_qec_cnots_final",
            circuit=[
                [
                    Circuit(
                        "cx",
                        channels=[self.qubit_channels[q] for q in qubit_pair],
                    )
                    for qubit_pair in cnot_slice
                ]
                for cnot_slice in cnots
            ],
        )

    def teleportation_finalization_circuit_final(self) -> Circuit:
        """
        Obtain the circuit that finalizes the teleportation of qubits while moving the
        block to its final position.
        """
        teleportation_info = [
            ((3, 3, 1), "measure_x"),
            ((3, 2, 1), "measure_z"),
            ((2, 5, 1), "measure_z"),
            ((3, 5, 1), "measure_x"),
        ]

        teleportation_circ_seq = [[]]
        for dq, meas_op in teleportation_info:
            c_channel = Channel("classical")
            teleportation_circ_seq[0].append(
                Circuit(
                    meas_op,
                    channels=[self.qubit_channels[dq], c_channel],
                )
            )
        return Circuit("teleportation_final", teleportation_circ_seq)

    def measure_stabilizers_operation_circuit_final(self) -> Circuit:
        """
        Obtain the circuit that measures the stabilizers of the block in its final
        position.
        """
        mops_list = [
            ("measure_X", (1, 0, 1)),
            ("measure_Z", (0, 2, 1)),
            ("measure_Z", (3, 1, 1)),
            ("measure_Z", (1, 1, 1)),
            ("measure_Z", (2, 2, 1)),
            ("measure_X", (1, 2, 1)),
            ("measure_X", (2, 1, 1)),
            ("measure_Z", (1, 5, 1)),
            ("measure_X", (3, 4, 1)),
            ("measure_Z", (0, 4, 1)),
            ("measure_X", (1, 4, 1)),
            ("measure_X", (2, 3, 1)),
            ("measure_Z", (1, 3, 1)),
            ("measure_Z", (2, 4, 1)),
        ]

        classical_channels = [Channel("classical", f"c_{dq}_1") for _, dq in mops_list]

        return Circuit(
            "measure_stabilizers_final",
            [
                [
                    Circuit(
                        op,
                        channels=[self.qubit_channels[q], c_chan],
                    )
                    for (op, q), c_chan in zip(
                        mops_list, classical_channels, strict=True
                    )
                ],
            ],
        )

    def test_map_stabilizer_schedule(self):
        """ "
        Test the mapping of stabilizer schedules based on block orientation and
        top-left bulk stabilizer type.
        """
        # Take 8 stabilizers
        stabilizers = self.twisted_rsc_block_v3z.stabilizers[:8]

        # Use the 8 different schedules on them
        stab_to_detailed_sched = {
            stabilizers[0]: DetailedSchedule.N1,
            stabilizers[1]: DetailedSchedule.N2,
            stabilizers[2]: DetailedSchedule.N3,
            stabilizers[3]: DetailedSchedule.N4,
            stabilizers[4]: DetailedSchedule.Z1,
            stabilizers[5]: DetailedSchedule.Z2,
            stabilizers[6]: DetailedSchedule.Z3,
            stabilizers[7]: DetailedSchedule.Z4,
        }

        # Test for (is_top_left_bulk_stab_x=False, is_block_horizontal=False)
        self.assertEqual(
            map_stabilizer_schedule(
                is_top_left_bulk_stab_x=False,
                is_block_horizontal=False,
                stab_schedule_dict_default=stab_to_detailed_sched,
            ),
            stab_to_detailed_sched,
        )

        # Test for (is_top_left_bulk_stab_x=True, is_block_horizontal=False)
        self.assertEqual(
            map_stabilizer_schedule(
                is_top_left_bulk_stab_x=True,
                is_block_horizontal=False,
                stab_schedule_dict_default=stab_to_detailed_sched,
            ),
            {
                stab: det_sched.invert_vertically()
                for stab, det_sched in stab_to_detailed_sched.items()
            },
        )

        # Test for (is_top_left_bulk_stab_x=True, is_block_horizontal=True)
        self.assertEqual(
            map_stabilizer_schedule(
                is_top_left_bulk_stab_x=True,
                is_block_horizontal=True,
                stab_schedule_dict_default=stab_to_detailed_sched,
            ),
            {
                stab: det_sched.rotate_ccw_90()
                for stab, det_sched in stab_to_detailed_sched.items()
            },
        )

        # Test for (is_top_left_bulk_stab_x=False, is_block_horizontal=True)
        self.assertEqual(
            map_stabilizer_schedule(
                is_top_left_bulk_stab_x=False,
                is_block_horizontal=True,
                stab_schedule_dict_default=stab_to_detailed_sched,
            ),
            {
                stab: det_sched.invert_vertically().rotate_ccw_90()
                for stab, det_sched in stab_to_detailed_sched.items()
            },
        )
