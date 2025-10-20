"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

import unittest
from copy import deepcopy

from loom.eka import Circuit, Channel, ChannelType, Lattice, PauliOperator
from loom.eka.utilities import Orientation, Direction, ResourceState
from loom.eka.operations import StateInjection
from loom.interpreter import InterpretationStep, Syndrome, Detector

from loom_rotated_surface_code.code_factory import RotatedSurfaceCode
from loom_rotated_surface_code.applicator.state_injection import (
    get_physical_state_reset,
    find_qubits_quadrant,
    reset_into_four_quadrants,
    find_centered_logical_operators,
    create_deterministic_syndromes,
    state_injection,
)


class TestRotatedSurfaceCodeTStateInjection(unittest.TestCase):
    """
    Tests the state injection operation on a RotatedSurfaceCode.
    """

    def setUp(self):
        self.lattice = Lattice.square_2d((7, 7))
        self.block = RotatedSurfaceCode.create(
            dx=3,
            dz=3,
            lattice=self.lattice,
            unique_label="test_block",
        )
        self.vertical_x_block = RotatedSurfaceCode.create(
            dx=3,
            dz=3,
            lattice=self.lattice,
            unique_label="test_block_vertical",
            x_boundary=Orientation.VERTICAL,
            weight_2_stab_is_first_row=False,
        )
        self.big_block = RotatedSurfaceCode.create(
            dx=5,
            dz=5,
            lattice=self.lattice,
            unique_label="big_block",
            weight_2_stab_is_first_row=False,
            x_boundary=Orientation.VERTICAL,
        )
        self.base_step = InterpretationStep(
            block_history=((self.block,),),
        )

    def test_t_injection_physical_circuit(self):
        """
        Test the get_physical_t_reset function to ensure it creates the correct
        circuit.
        """
        # Check the circuit that initialise the given qubit into the T state
        for qubit in self.block.data_qubits:
            t_init_circuit = get_physical_state_reset(
                interpretation_step=deepcopy(self.base_step),
                input_block=self.block,
                qubit_to_reset=qubit,
                resource_state=ResourceState.T,
            )
            q_channel = Channel(label=f"{qubit}", type=ChannelType.QUANTUM)
            expected_circuit = Circuit(
                name="Reset_T",
                circuit=(
                    (Circuit(name="Reset_+", channels=[q_channel]),),
                    (Circuit(name="T", channels=[q_channel]),),
                ),
                channels=[q_channel],
            )
            self.assertEqual(t_init_circuit, expected_circuit)

        # Check that the function raises an error if the qubit is not a data qubit:
        # One is outside the block, the other is an ancilla qubit of the block.
        for wrong_qubit in ((9, 9, 0), (1, 1, 1)):
            with self.assertRaises(ValueError) as cm:
                get_physical_state_reset(
                    interpretation_step=deepcopy(self.base_step),
                    input_block=self.block,
                    qubit_to_reset=wrong_qubit,
                    resource_state=ResourceState.T,
                )
            err_msg = (
                f"Qubit {wrong_qubit} is not a data qubit in the given block"
                f" {self.block.unique_label}."
            )
            self.assertEqual(err_msg, str(cm.exception))

    def test_s_injection_physical_circuit(self):
        """
        Test the get_physical_state_reset function to ensure it creates the correct
        circuit.
        """
        # Check the circuit that initialise the given qubit into the S state
        for qubit in self.block.data_qubits:
            s_init_circuit = get_physical_state_reset(
                interpretation_step=deepcopy(self.base_step),
                input_block=self.block,
                qubit_to_reset=qubit,
                resource_state=ResourceState.S,
            )
            q_channel = Channel(label=f"{qubit}", type=ChannelType.QUANTUM)
            expected_circuit = Circuit(
                name="Reset_S",
                circuit=(
                    (Circuit(name="Reset_+", channels=[q_channel]),),
                    (Circuit(name="Phase", channels=[q_channel]),),
                ),
                channels=[q_channel],
            )
            self.assertEqual(s_init_circuit, expected_circuit)

    def test_find_qubits_quadrant(self):
        """
        Test the find_qubits_quadrant function to ensure it correctly identifies
        qubits in each quadrant of the block.
        """
        # Check that the quadrants are correctly identified for the 3x3 block
        # Only the qubits included in the 2-body stabilizers are selected.
        standard_quadrants = {
            Direction.TOP: ((1, 0, 0), (2, 0, 0)),
            Direction.BOTTOM: ((0, 2, 0), (1, 2, 0)),
            Direction.LEFT: ((0, 0, 0), (0, 1, 0)),
            Direction.RIGHT: ((2, 1, 0), (2, 2, 0)),
        }
        non_standard_quadrants = {
            Direction.TOP: ((1, 0, 0), (0, 0, 0)),
            Direction.BOTTOM: ((2, 2, 0), (1, 2, 0)),
            Direction.LEFT: ((0, 2, 0), (0, 1, 0)),
            Direction.RIGHT: ((2, 1, 0), (2, 0, 0)),
        }
        quadrants_big = {
            Direction.TOP: (
                (1, 1, 0),
                (3, 0, 0),
                (2, 1, 0),
                (0, 0, 0),
                (1, 0, 0),
                (2, 0, 0),
            ),
            Direction.BOTTOM: (
                (3, 3, 0),
                (2, 4, 0),
                (3, 4, 0),
                (4, 4, 0),
                (1, 4, 0),
                (2, 3, 0),
            ),
            Direction.LEFT: (
                (0, 1, 0),
                (1, 2, 0),
                (0, 2, 0),
                (1, 3, 0),
                (0, 3, 0),
                (0, 4, 0),
            ),
            Direction.RIGHT: (
                (3, 1, 0),
                (4, 0, 0),
                (4, 1, 0),
                (4, 2, 0),
                (4, 3, 0),
                (3, 2, 0),
            ),
        }

        # Check that the quadrants are correctly identified for the 3 configurations
        for block, quadrants in zip(
            (self.block, self.vertical_x_block, self.big_block),
            (standard_quadrants, non_standard_quadrants, quadrants_big),
            strict=True,
        ):
            for direction in Direction:
                qubits = find_qubits_quadrant(block, direction)
                expected_qubits = quadrants[direction]
                self.assertEqual(qubits, expected_qubits)

    def test_reset_into_four_quadrants(self):
        """
        Test the reset_into_four_quadrants function to ensure it correctly resets
        the data qubits into four quadrants.
        """
        # Check that the 3x3 block is correctly reset into four quadrants
        reset_circuit, deterministic_stabs = reset_into_four_quadrants(
            interpretation_step=deepcopy(self.base_step),
            block=self.block,
        )
        central_qubit = (1, 1, 0)  # The central qubit of the 3x3 block
        outer_qubits = [q for q in self.block.data_qubits if q != central_qubit]
        channels = [Channel(label=f"{q}") for q in outer_qubits]
        expected_circuit = Circuit(
            name="reset four quadrants",
            circuit=(
                tuple(
                    Circuit(name="Reset_+", channels=[chan])
                    for chan in channels
                    if chan.label
                    in ("(0, 0, 0)", "(0, 1, 0)", "(2, 1, 0)", "(2, 2, 0)")
                )
                + tuple(
                    Circuit(name="Reset_0", channels=[chan])
                    for chan in channels
                    if chan.label
                    in ("(1, 0, 0)", "(2, 0, 0)", "(0, 2, 0)", "(1, 2, 0)")
                ),
            ),
            channels=channels,
        )
        self.assertEqual(reset_circuit, expected_circuit)

        expected_deterministic_stabs = tuple(
            stab for d in Direction for stab in self.block.boundary_stabilizers(d)
        )
        self.assertEqual(
            sorted(deterministic_stabs, key=lambda x: x.pauli),
            sorted(expected_deterministic_stabs, key=lambda x: x.pauli),
        )

        # Check that the 5x5 block is correctly reset into four quadrants
        reset_big_circuit, deterministic_big_stabs = reset_into_four_quadrants(
            interpretation_step=deepcopy(self.base_step),
            block=self.big_block,
        )
        central_qubit = (2, 2, 0)  # The central qubit of the 5x5 block quadrant
        channels_big = [
            Channel(label=f"{q}", type=ChannelType.QUANTUM)
            for q in self.big_block.data_qubits
            if q != central_qubit
        ]
        qubits_to_reset_zero = [
            (0, 1, 0),
            (0, 2, 0),
            (0, 3, 0),
            (0, 4, 0),
            (1, 2, 0),
            (1, 3, 0),
            (4, 0, 0),
            (4, 1, 0),
            (4, 2, 0),
            (4, 3, 0),
            (3, 1, 0),
            (3, 2, 0),
        ]
        qubits_to_reset_plus = [
            q
            for q in self.big_block.data_qubits
            if q not in qubits_to_reset_zero and q != central_qubit
        ]
        expected_big_circuit = Circuit(
            name="Reset_5x5",
            circuit=(
                tuple(
                    Circuit(name="Reset_+", channels=[chan])
                    for chan in channels_big
                    if chan.label in map(str, qubits_to_reset_plus)
                )
                + tuple(
                    Circuit(name="Reset_0", channels=[chan])
                    for chan in channels_big
                    if chan.label in map(str, qubits_to_reset_zero)
                ),
            ),
            channels=channels_big,
        )
        self.assertEqual(reset_big_circuit, expected_big_circuit)

        expected_deterministic_big_stabs = tuple(
            stab
            for stab in self.big_block.stabilizers
            if (
                set(stab.pauli) == {"Z"}
                and all((q in qubits_to_reset_zero) for q in stab.data_qubits)
            )
            or (
                set(stab.pauli) == {"X"}
                and all((q in qubits_to_reset_plus) for q in stab.data_qubits)
            )
        )
        self.assertEqual(
            sorted(deterministic_big_stabs, key=lambda x: x.pauli),
            sorted(expected_deterministic_big_stabs, key=lambda x: x.pauli),
        )

    def test_find_centered_logical_operators(self):
        """
        Test the find_centered_logical_operators function to ensure it correctly finds
        the centered logical operators.
        """
        # Check for the 3x3 block
        centered_log_x, centered_log_z = find_centered_logical_operators(
            input_block=self.block, center_qubit=(1, 1, 0)
        )
        self.assertEqual(
            centered_log_x, PauliOperator("XXX", ((0, 1, 0), (1, 1, 0), (2, 1, 0)))
        )
        self.assertEqual(
            centered_log_z, PauliOperator("ZZZ", ((1, 0, 0), (1, 1, 0), (1, 2, 0)))
        )

        # Check for the 5x5 block
        centered_log_x, centered_log_z = find_centered_logical_operators(
            input_block=self.big_block, center_qubit=(2, 2, 0)
        )
        self.assertEqual(
            centered_log_x, PauliOperator("XXXXX", tuple((2, i, 0) for i in range(5)))
        )
        self.assertEqual(
            centered_log_z, PauliOperator("ZZZZZ", tuple((i, 2, 0) for i in range(5)))
        )

    def test_create_deterministic_syndromes(self):
        """
        Test the create_deterministic_syndromes function to ensure it creates the
        right syndromes for the given stabilizers.
        """
        # Check for the 3x3 block
        deterministic_stabs = tuple(
            stab for d in Direction for stab in self.block.boundary_stabilizers(d)
        )
        generated_syndromes = create_deterministic_syndromes(
            interpretation_step=deepcopy(self.base_step),
            block=self.block,
            deterministic_stabs=deterministic_stabs,
        )
        expected_syndromes = tuple(
            Syndrome(
                stabilizer=stab.uuid,
                measurements=(),
                block=self.block.uuid,
                round=0,
            )
            for stab in deterministic_stabs
        )
        self.assertEqual(generated_syndromes, expected_syndromes)

        # Check for the 5x5 block
        deterministic_ancilla_qubits = (
            (1, 0, 1),
            (3, 0, 1),
            (2, 1, 1),
            (5, 1, 1),
            (5, 3, 1),
            (4, 2, 1),
            (0, 2, 1),
            (0, 4, 1),
            (1, 3, 1),
            (2, 5, 1),
            (4, 5, 1),
            (3, 4, 1),
        )
        deterministic_big_stabs = tuple(
            stab
            for stab in self.big_block.stabilizers
            if stab.ancilla_qubits[0] in deterministic_ancilla_qubits
        )
        big_generated_syndromes = create_deterministic_syndromes(
            interpretation_step=InterpretationStep(
                block_history=((self.big_block,),),
            ),
            block=self.big_block,
            deterministic_stabs=deterministic_big_stabs,
        )
        big_expected_syndromes = tuple(
            Syndrome(
                stabilizer=stab.uuid,
                measurements=(),
                block=self.big_block.uuid,
                round=0,
            )
            for stab in deterministic_big_stabs
        )
        self.assertEqual(big_generated_syndromes, big_expected_syndromes)

    def test_state_injection(self):  # pylint: disable=too-many-locals
        """
        Test the state_injection wrapper to ensure that it results in the right
        Circuit and Syndromes
        """
        for state in ResourceState:
            state_injection_op = StateInjection(self.block.unique_label, state)
            interpretation_step = state_injection(
                interpretation_step=deepcopy(self.base_step),
                operation=state_injection_op,
                same_timeslice=False,
                debug_mode=True,
            )
            new_block: RotatedSurfaceCode = interpretation_step.get_block(
                self.block.unique_label
            )

            # Check that the block is correctly updated
            expected_block = RotatedSurfaceCode.create(
                dx=self.block.size[0],
                dz=self.block.size[1],
                lattice=self.lattice,
                unique_label=self.block.unique_label,
                logical_x_operator=PauliOperator(
                    "XXX", ((0, 0, 0), (1, 0, 0), (2, 0, 0))
                ),
                logical_z_operator=PauliOperator(
                    "ZZZ", ((0, 0, 0), (0, 1, 0), (0, 2, 0))
                ),
            )
            self.assertEqual(new_block, expected_block)

            # Check that the circuit is correctly created
            center_qubit = (
                (new_block.upper_left_qubit[0] + new_block.size[0]) // 2,
                (new_block.upper_left_qubit[1] + new_block.size[1]) // 2,
                0,
            )
            qubits_in_zero = tuple(
                set(
                    q
                    for d in (Direction.TOP, Direction.BOTTOM)
                    for stab in self.block.boundary_stabilizers(d)
                    for q in stab.data_qubits
                )
            )
            qubits_in_plus = tuple(
                set(
                    q
                    for d in (Direction.LEFT, Direction.RIGHT)
                    for stab in new_block.boundary_stabilizers(d)
                    for q in stab.data_qubits
                )
            )
            reset_circuit = Circuit(
                name="reset four quadrants",
                circuit=(
                    tuple(
                        Circuit("reset_+", channels=[Channel(label=f"{q}")])
                        for q in qubits_in_plus
                    )
                    + tuple(
                        Circuit("reset_0", channels=[Channel(label=f"{q}")])
                        for q in qubits_in_zero
                    ),
                ),
            )
            resource_state_circuit = Circuit(
                "resource state reset",
                circuit=(
                    (
                        Circuit(
                            "reset_+",
                            channels=[
                                central_channel := Channel(label=f"{center_qubit}")
                            ],
                        ),
                    ),
                    (
                        Circuit(
                            "phase" if state == ResourceState.S else state.value,
                            channels=[central_channel],
                        ),
                    ),
                ),
            )
            expected_circuit = Circuit(
                name=(f"inject {state.value} into block {new_block.unique_label}"),
                circuit=((resource_state_circuit, reset_circuit),),
            )

            self.assertEqual(
                interpretation_step.intermediate_circuit_sequence[0][0].circuit[0][0],
                expected_circuit,
            )
            self.assertEqual(
                interpretation_step.intermediate_circuit_sequence[0][0]
                .circuit[2][0]
                .name,
                "measure test_block syndromes 1 time(s)",
            )
            self.assertEqual(
                interpretation_step.intermediate_circuit_sequence[0][0].name,
                (
                    f"inject {state.value} into block {new_block.unique_label} and "
                    f"measure syndromes"
                ),
            )

            # Check that the syndromes are correctly created
            centered_block = interpretation_step.block_history[-2][0]
            deterministic_stabs = tuple(
                stab
                for stab in centered_block.stabilizers
                if all(
                    (
                        (q in qubits_in_plus and p == "X")
                        or (q in qubits_in_zero and p == "Z")
                    )
                    for q, p in zip(stab.data_qubits, stab.pauli, strict=True)
                )
            )
            expected_quadrant_syndromes = tuple(
                Syndrome(
                    stabilizer=stab.uuid,
                    measurements=(),
                    block=centered_block.uuid,
                    round=0,
                )
                for stab in deterministic_stabs
            )
            expected_measureblock_syndromes = tuple(
                Syndrome(
                    stabilizer=stab.uuid,
                    measurements=((f"c_{stab.ancilla_qubits[0]}", 0),),
                    block=centered_block.uuid,
                    round=1,
                )
                for stab in centered_block.stabilizers
            )
            expected_syndromes = (
                expected_quadrant_syndromes + expected_measureblock_syndromes
            )
            self.assertEqual(interpretation_step.syndromes, expected_syndromes)

            # Check that the detectors are correctly created
            expected_detectors = tuple(
                Detector(
                    (quadrant_synd, synd),
                )
                for quadrant_synd in expected_quadrant_syndromes
                for synd in expected_measureblock_syndromes
                if quadrant_synd.stabilizer == synd.stabilizer
            )
            self.assertEqual(interpretation_step.detectors, expected_detectors)

            expected_x_updates = (("c_(2, 1, 1)", 0), ("c_(0, 1, 1)", 0))
            expected_z_updates = (("c_(1, 1, 1)", 0), ("c_(1, 3, 1)", 0))
            # Check that the logical operator updates are correctly recorded
            self.assertEqual(
                interpretation_step.logical_x_operator_updates[
                    new_block.logical_x_operators[0].uuid
                ],
                expected_x_updates,
            )
            self.assertEqual(
                interpretation_step.logical_z_operator_updates[
                    new_block.logical_z_operators[0].uuid
                ],
                expected_z_updates,
            )

            # Check that the block evolution is correctly recorded
            self.assertEqual(
                {
                    centered_block.uuid: (self.block.uuid,),
                    new_block.uuid: (centered_block.uuid,),
                },
                interpretation_step.block_evolution,
            )

    def test_invalid_state_injection(self):
        """
        Test that an error is raised when trying to apply state_injection to a block
        with even size.
        """
        invalid_blocks = tuple(
            RotatedSurfaceCode.create(
                dx=size[0],
                dz=size[1],
                lattice=self.lattice,
                unique_label=f"invalid_block_{size[0]}x{size[1]}",
            )
            for size in ((3, 4), (4, 3), (4, 4))
        )
        for block in invalid_blocks:
            base_step = InterpretationStep(
                block_history=((block,),),
            )
            with self.assertRaises(ValueError) as cm:
                state_injection(
                    interpretation_step=deepcopy(base_step),
                    operation=StateInjection(block.unique_label, ResourceState.T),
                    same_timeslice=False,
                    debug_mode=True,
                )
            self.assertEqual(
                str(cm.exception),
                f"Expected input_block.size to be all odd, but got {block.size}.",
            )


if __name__ == "__main__":
    unittest.main()
