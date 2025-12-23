"""
Copyright 2024 Entropica Labs Pte Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""

# pylint: disable=duplicate-code
import unittest

from loom.eka import Circuit, Channel, Lattice
from loom.eka.utilities import Orientation, Direction
from loom.interpreter import InterpretationStep, Syndrome

from loom_rotated_surface_code.code_factory import RotatedSurfaceCode
from loom_rotated_surface_code.applicator.move_corners import move_corners
from loom_rotated_surface_code.applicator.y_wall_out import y_wall_out

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
        self.base_step = InterpretationStep.create(
            [self.big_block_v3z],
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
        base_int_step = InterpretationStep.create(
            [self.twisted_rsc_block_v3z],
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
            False,
            True,
        )
        output_circuit_seq = interpreted_eka.intermediate_circuit_sequence

        # Define the expected circuit sequence
        expected_circuit_seq = (
            self.init_block_syndrome_measurement_circuit()
            + self.y_wall_circuit()
            + self.final_block_first_swap_then_qec_circuit()
            + self.final_block_second_swap_then_qec_circuit()
            + self.final_block_syndrome_measurement_circuit()
        )

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

        # Assert that there are no trivial detectors in the block, i.e. dependent on the
        # same syndrome
        for det in interpreted_eka.detectors:
            self.assertEqual(len(set(det.syndromes)), len(det.syndromes))

    def init_block_syndrome_measurement_circuit(
        self,
    ) -> tuple[tuple[Circuit, ...], ...]:
        """Obtain the circuit for one round of syndrome measurement of the
        initial block."""
        # Initialization circuit
        init_block_syndrome_measurement_reset_circuit = Circuit(
            name="Initialization of syndrome measurement ancilla",
            circuit=[
                [
                    Circuit(
                        f"reset_{'0' if stab.pauli_type == 'Z' else '+'}",
                        channels=[self.qubit_channels[stab.ancilla_qubits[0]]],
                    )
                    for stab in self.twisted_rsc_block_v3z.stabilizers
                ]
            ],
        )

        # CNOT circuit
        cnots = [
            # TIME SLICE 0
            [
                ((1, 0, 1), (1, 0, 0)),
                ((0, 2, 0), (0, 2, 1)),
                ((1, 1, 0), (1, 1, 1)),
                ((1, 3, 0), (1, 3, 1)),
                ((2, 2, 0), (2, 2, 1)),
                ((1, 2, 1), (1, 2, 0)),
                ((2, 1, 1), (2, 1, 0)),
            ],
            # TIME SLICE 1
            [
                ((1, 0, 1), (0, 0, 0)),
                ((0, 1, 0), (0, 2, 1)),
                ((0, 5, 1), (0, 5, 0)),
                ((1, 0, 0), (1, 1, 1)),
                ((1, 2, 0), (1, 3, 1)),
                ((1, 5, 0), (1, 5, 1)),
                ((2, 1, 0), (2, 2, 1)),
                ((2, 4, 0), (2, 4, 1)),
                ((1, 2, 1), (0, 2, 0)),
                ((1, 4, 1), (1, 4, 0)),
                ((2, 1, 1), (1, 1, 0)),
                ((2, 3, 1), (2, 3, 0)),
                ((2, 5, 1), (2, 5, 0)),
            ],
            # TIME SLICE 2
            [
                ((2, 1, 0), (3, 1, 1)),
                ((0, 5, 1), (0, 4, 0)),
                ((0, 1, 0), (1, 1, 1)),
                ((1, 4, 0), (1, 5, 1)),
                ((2, 3, 0), (2, 4, 1)),
                ((1, 2, 1), (1, 1, 0)),
                ((1, 4, 1), (1, 3, 0)),
                ((2, 1, 1), (2, 0, 0)),
                ((2, 3, 1), (2, 2, 0)),
                ((2, 5, 1), (1, 5, 0)),
            ],
            # TIME SLICE 3
            [
                ((1, 6, 1), (1, 5, 0)),
                ((2, 0, 0), (3, 1, 1)),
                ((2, 3, 0), (3, 3, 1)),
                ((2, 5, 0), (3, 5, 1)),
                ((0, 0, 0), (1, 1, 1)),
                ((0, 3, 0), (1, 3, 1)),
                ((0, 5, 0), (1, 5, 1)),
                ((1, 2, 0), (2, 2, 1)),
                ((1, 4, 0), (2, 4, 1)),
                ((1, 2, 1), (0, 1, 0)),
                ((1, 4, 1), (0, 4, 0)),
                ((2, 1, 1), (1, 0, 0)),
                ((2, 3, 1), (1, 3, 0)),
                ((2, 5, 1), (2, 4, 0)),
            ],
            # TIME SLICE 4
            [
                ((1, 6, 1), (0, 5, 0)),
                ((2, 2, 0), (3, 3, 1)),
                ((2, 4, 0), (3, 5, 1)),
                ((0, 2, 0), (1, 3, 1)),
                ((0, 4, 0), (1, 5, 1)),
                ((1, 1, 0), (2, 2, 1)),
                ((1, 3, 0), (2, 4, 1)),
                ((1, 4, 1), (0, 3, 0)),
                ((2, 3, 1), (1, 2, 0)),
                ((2, 5, 1), (1, 4, 0)),
            ],
        ]
        init_block_syndrome_measurement_cnot_circuit = Circuit(
            "Initial block syndrome measurement CNOT circuit",
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

        # Measure circuit
        init_block_syndrome_measurement_measure_circuit = Circuit(
            name="Measure of syndrome measurement ancilla",
            circuit=[
                [
                    Circuit(
                        f"measure_{'z' if stab.pauli_type == 'Z' else 'x'}",
                        channels=[
                            self.qubit_channels[stab.ancilla_qubits[0]],
                            Channel("classical"),
                        ],
                    )
                    for stab in self.twisted_rsc_block_v3z.stabilizers
                ]
            ],
        )

        # Assemble circuit
        init_block_syndrome_measurement_circuit = (
            (init_block_syndrome_measurement_reset_circuit,),
            (init_block_syndrome_measurement_cnot_circuit,),
            (init_block_syndrome_measurement_measure_circuit,),
        )
        return init_block_syndrome_measurement_circuit

    def y_wall_circuit(self) -> tuple[tuple[Circuit, ...], ...]:
        """
        Obtain the circuit that measures the qubits of the wall in the Y basis
        and applies Hadamard to qubits beyond the wall.
        """
        qubits_to_measure = [(0, 3, 0), (1, 3, 0), (2, 3, 0)]

        y_wall_measurement_circuit = Circuit(
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

        qubits_to_had = [
            (0, 4, 0),
            (0, 5, 0),
            (1, 4, 0),
            (1, 5, 0),
            (2, 4, 0),
            (2, 5, 0),
        ]

        hadamard_circuit = Circuit(
            "hadamard beyond the wall",
            [[Circuit("h", channels=[self.qubit_channels[q]]) for q in qubits_to_had]],
        )

        return ((y_wall_measurement_circuit, hadamard_circuit),)

    def final_block_first_swap_then_qec_circuit(
        self,
    ) -> tuple[tuple[Circuit, ...], ...]:
        """Obtain the circuit for the first SWAP-then-QEC round of the final block."""
        # Initialization circuit
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
        first_swap_then_qec_reset_circuit = Circuit(
            name="Initialization of qubits for first swap-then-qec",
            circuit=[
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

        # CNOT circuit
        cnots = [
            # TIME SLICE 0
            [
                ((0, 4, 0), (1, 4, 1)),
                ((1, 5, 1), (0, 5, 0)),
                ((2, 4, 1), (1, 4, 0)),
                ((1, 5, 0), (2, 5, 1)),
                ((2, 4, 0), (3, 4, 1)),
                ((3, 5, 1), (2, 5, 0)),
                ((0, 2, 0), (1, 3, 1)),
                ((1, 2, 1), (0, 1, 0)),
                ((0, 0, 0), (1, 1, 1)),
                ((2, 3, 1), (1, 2, 0)),
                ((1, 1, 0), (2, 2, 1)),
                ((2, 1, 1), (1, 0, 0)),
                ((2, 2, 0), (3, 3, 1)),
                ((3, 2, 1), (2, 1, 0)),
                ((2, 0, 0), (3, 1, 1)),
            ],
            # TIME SLICE 1
            [
                ((1, 0, 0), (1, 1, 1)),
                ((1, 2, 1), (0, 2, 0)),
                ((2, 1, 1), (1, 1, 0)),
                ((3, 2, 1), (2, 2, 0)),
                ((1, 2, 0), (1, 3, 1)),
                ((2, 1, 0), (2, 2, 1)),
                ((1, 5, 1), (0, 4, 0)),
                ((1, 4, 0), (2, 5, 1)),
                ((2, 4, 1), (1, 3, 0)),
                ((2, 3, 0), (3, 4, 1)),
            ],
            # TIME SLICE 2
            [
                ((3, 2, 1), (3, 1, 0)),
                ((1, 2, 1), (1, 1, 0)),
                ((1, 2, 0), (2, 2, 1)),
                ((2, 1, 0), (3, 1, 1)),
                ((1, 5, 1), (1, 5, 0)),
                ((3, 4, 0), (3, 4, 1)),
                ((1, 4, 0), (1, 4, 1)),
                ((2, 4, 1), (2, 4, 0)),
                ((2, 3, 1), (1, 3, 0)),
                ((2, 3, 0), (3, 3, 1)),
            ],
            # TIME SLICE 3
            [
                ((3, 1, 1), (3, 1, 0)),
                ((1, 1, 1), (1, 1, 0)),
                ((2, 3, 1), (2, 2, 0)),
                ((1, 2, 0), (1, 2, 1)),
                ((2, 1, 0), (2, 1, 1)),
                ((1, 4, 0), (1, 5, 1)),
                ((3, 5, 1), (2, 4, 0)),
                ((1, 4, 1), (1, 3, 0)),
                ((2, 3, 0), (2, 4, 1)),
            ],
            # TIME SLICE 4
            [
                ((2, 2, 1), (2, 2, 0)),
                ((3, 4, 0), (3, 5, 1)),
                ((2, 5, 1), (2, 4, 0)),
                ((1, 3, 1), (1, 3, 0)),
                ((2, 3, 0), (2, 3, 1)),
            ],
        ]
        first_swap_then_qec_cnots_circuit = Circuit(
            name="First SWAP-then-QEC final block syndrome measurement CNOT circuit",
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

        # Teleportation
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
        first_swap_then_qec_teleportation_finalization_circuit = Circuit(
            "teleportation",
            teleportation_circ_seq,
        )

        # Measurement
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
            ("measure_z", (2, 4, 0)),
            ("measure_z", (1, 3, 0)),
            ("measure_x", (2, 3, 0)),
        ]

        classical_channels = [Channel("classical", f"c_{dq}_0") for _, dq in mops_list]

        first_swap_then_qec_measurement_circuit = Circuit(
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

        # Assemble circuit
        final_block_first_swap_then_qec_circuit = (
            (first_swap_then_qec_reset_circuit,),
            (first_swap_then_qec_cnots_circuit,),
            (
                first_swap_then_qec_measurement_circuit,
                first_swap_then_qec_teleportation_finalization_circuit,
            ),
        )
        return final_block_first_swap_then_qec_circuit

    def final_block_second_swap_then_qec_circuit(
        self,
    ) -> tuple[tuple[Circuit, ...], ...]:
        """Obtain the circuit for the second SWAP-then-QEC round of the final block."""
        # Initialization circuit
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
            (1, 3, 0),
            (0, 2, 0),
            # The final two are teleportation qubits
            (2, 2, 0),
            (2, 4, 0),
        ]
        second_swap_then_qec_reset_circuit = Circuit(
            name="Initialization of qubits for second swap-then-qec",
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

        # CNOT circuit
        cnots = [
            # TIME SLICE 0
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
            # TIME SLICE 1
            [
                ((2, 1, 0), (3, 1, 1)),
                ((0, 1, 0), (1, 1, 1)),
                ((1, 2, 0), (2, 2, 1)),
                ((1, 2, 1), (1, 1, 0)),
                ((2, 1, 1), (2, 0, 0)),
                ((0, 3, 0), (1, 3, 1)),
            ],
            # TIME SLICE 2
            [
                ((1, 0, 1), (0, 0, 0)),
                ((0, 1, 0), (0, 2, 1)),
                ((1, 0, 0), (1, 1, 1)),
                ((2, 1, 0), (2, 2, 1)),
                ((1, 2, 1), (0, 2, 0)),
                ((2, 1, 1), (1, 1, 0)),
                ((1, 4, 0), (1, 5, 1)),
                ((3, 4, 1), (2, 4, 0)),
                ((0, 3, 0), (0, 4, 1)),
                ((1, 4, 1), (0, 4, 0)),
                ((2, 3, 0), (2, 4, 1)),
                ((1, 2, 0), (1, 3, 1)),
                ((2, 3, 1), (1, 3, 0)),
            ],
            # TIME SLICE 3
            [
                ((1, 0, 1), (1, 0, 0)),
                ((0, 2, 0), (0, 2, 1)),
                ((1, 1, 0), (1, 1, 1)),
                ((1, 2, 1), (1, 2, 0)),
                ((2, 1, 1), (2, 1, 0)),
                ((1, 4, 1), (1, 3, 0)),
                ((1, 4, 0), (2, 4, 1)),
                ((2, 3, 1), (2, 2, 0)),
            ],
            # TIME SLICE 4
            [
                ((2, 2, 0), (2, 2, 1)),
                ((0, 4, 0), (0, 4, 1)),
                ((1, 4, 1), (1, 4, 0)),
                ((2, 4, 0), (2, 4, 1)),
                ((1, 3, 0), (1, 3, 1)),
                ((2, 3, 1), (2, 3, 0)),
            ],
        ]
        second_swap_then_qec_cnots_circuit = Circuit(
            name=("Second SWAP-then-QEC final block syndrome measurement CNOT circuit"),
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

        # Teleportation
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
        second_swap_then_qec_teleportation_finalization_circuit = Circuit(
            "teleportation_final", teleportation_circ_seq
        )

        # Measurement circuit
        mops_list = [
            ("measure_x", (1, 0, 1)),
            ("measure_z", (0, 2, 1)),
            ("measure_z", (3, 1, 1)),
            ("measure_z", (1, 1, 1)),
            ("measure_z", (2, 2, 1)),
            ("measure_x", (1, 2, 1)),
            ("measure_x", (2, 1, 1)),
            ("measure_z", (1, 5, 1)),
            ("measure_x", (3, 4, 1)),
            ("measure_z", (0, 4, 1)),
            ("measure_x", (1, 4, 1)),
            ("measure_z", (2, 4, 1)),
            ("measure_z", (1, 3, 1)),
            ("measure_x", (2, 3, 1)),
        ]

        classical_channels = [Channel("classical", f"c_{dq}_1") for _, dq in mops_list]
        second_swap_then_qec_measurement_circuit = Circuit(
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

        # Assemble circuit
        final_block_second_swap_then_qec_circuit = (
            (second_swap_then_qec_reset_circuit,),
            (second_swap_then_qec_cnots_circuit,),
            (
                second_swap_then_qec_measurement_circuit,
                second_swap_then_qec_teleportation_finalization_circuit,
            ),
        )
        return final_block_second_swap_then_qec_circuit

    def final_block_syndrome_measurement_circuit(
        self,
    ) -> tuple[tuple[Circuit, ...], ...]:
        """Obtain the circuit for d-2 rounds of syndrome measurement of
        the final block."""
        # Initialization circuit
        qubits_to_init_in_x = [
            (1, 0, 1),
            (1, 2, 1),
            (2, 1, 1),
            (3, 4, 1),
            (1, 4, 1),
            (2, 3, 1),
        ]

        qubits_to_init_in_z = [
            (0, 2, 1),
            (3, 1, 1),
            (1, 1, 1),
            (2, 2, 1),
            (1, 5, 1),
            (0, 4, 1),
            (2, 4, 1),
            (1, 3, 1),
        ]
        final_block_syndrome_measurement_reset_circuit = Circuit(
            name="Initialization of syndrome measurement ancilla",
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

        # CNOT circuit
        cnots = [
            # TIME SLICE 0
            [
                ((2, 0, 0), (3, 1, 1)),
                ((0, 0, 0), (1, 1, 1)),
                ((1, 1, 0), (2, 2, 1)),
                ((1, 2, 1), (0, 1, 0)),
                ((2, 1, 1), (1, 0, 0)),
                ((0, 4, 0), (1, 5, 1)),
                ((3, 4, 1), (2, 3, 0)),
                ((1, 4, 1), (0, 3, 0)),
                ((1, 3, 0), (2, 4, 1)),
                ((0, 2, 0), (1, 3, 1)),
                ((2, 3, 1), (1, 2, 0)),
            ],
            # TIME SLICE 1
            [
                ((2, 1, 0), (3, 1, 1)),
                ((0, 1, 0), (1, 1, 1)),
                ((1, 2, 0), (2, 2, 1)),
                ((1, 2, 1), (1, 1, 0)),
                ((2, 1, 1), (2, 0, 0)),
                ((0, 3, 0), (1, 3, 1)),
            ],
            # TIME SLICE 2
            [
                ((1, 0, 1), (0, 0, 0)),
                ((0, 1, 0), (0, 2, 1)),
                ((1, 0, 0), (1, 1, 1)),
                ((2, 1, 0), (2, 2, 1)),
                ((1, 2, 1), (0, 2, 0)),
                ((2, 1, 1), (1, 1, 0)),
                ((1, 4, 0), (1, 5, 1)),
                ((3, 4, 1), (2, 4, 0)),
                ((0, 3, 0), (0, 4, 1)),
                ((1, 4, 1), (0, 4, 0)),
                ((2, 3, 0), (2, 4, 1)),
                ((1, 2, 0), (1, 3, 1)),
                ((2, 3, 1), (1, 3, 0)),
            ],
            # TIME SLICE 3
            [
                ((1, 0, 1), (1, 0, 0)),
                ((0, 2, 0), (0, 2, 1)),
                ((1, 1, 0), (1, 1, 1)),
                ((1, 2, 1), (1, 2, 0)),
                ((2, 1, 1), (2, 1, 0)),
                ((1, 4, 1), (1, 3, 0)),
                ((1, 4, 0), (2, 4, 1)),
                ((2, 3, 1), (2, 2, 0)),
            ],
            # TIME SLICE 4
            [
                ((2, 2, 0), (2, 2, 1)),
                ((0, 4, 0), (0, 4, 1)),
                ((1, 4, 1), (1, 4, 0)),
                ((2, 4, 0), (2, 4, 1)),
                ((1, 3, 0), (1, 3, 1)),
                ((2, 3, 1), (2, 3, 0)),
            ],
        ]
        final_block_syndrome_measurement_cnot_circuit = Circuit(
            "Final block syndrome measurement CNOT circuit",
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

        # Measure circuit
        final_block_syndrome_measurement_measure_circuit = Circuit(
            name="Measure of syndrome measurement ancilla",
            circuit=[
                [
                    Circuit(
                        "measure_x",
                        channels=[self.qubit_channels[q], Channel("classical")],
                    )
                    for q in qubits_to_init_in_x
                ]
                + [
                    Circuit(
                        "measure_z",
                        channels=[self.qubit_channels[q], Channel("classical")],
                    )
                    for q in qubits_to_init_in_z
                ]
            ],
        )

        # Assemble circuit
        final_block_syndrome_measurement_circuit = (
            (final_block_syndrome_measurement_reset_circuit,),
            (final_block_syndrome_measurement_cnot_circuit,),
            (final_block_syndrome_measurement_measure_circuit,),
        )
        return final_block_syndrome_measurement_circuit
