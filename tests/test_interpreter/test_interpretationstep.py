"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

import unittest
from uuid import uuid4
from copy import deepcopy

from loom.eka import (
    Circuit,
    Channel,
    ChannelType,
    Stabilizer,
    Block,
    PauliOperator,
)
from loom.interpreter import InterpretationStep, Syndrome
from loom.eka.utilities import SyndromeMissingError


class TestInterpretationStep(unittest.TestCase):
    """
    Test for the InterpretationStep class.
    """

    def setUp(self):
        super().setUp()
        # pylint: disable=duplicate-code
        self.rot_surf_code_1 = Block(
            stabilizers=(
                Stabilizer(
                    pauli="ZZZZ",
                    data_qubits=((1, 0, 0), (1, 1, 0), (0, 0, 0), (0, 1, 0)),
                    ancilla_qubits=((1, 1, 1),),
                ),
                Stabilizer(
                    pauli="ZZZZ",
                    data_qubits=((2, 1, 0), (2, 2, 0), (1, 1, 0), (1, 2, 0)),
                    ancilla_qubits=((2, 2, 1),),
                ),
                Stabilizer(
                    pauli="XXXX",
                    data_qubits=((1, 1, 0), (0, 1, 0), (1, 2, 0), (0, 2, 0)),
                    ancilla_qubits=((1, 2, 1),),
                ),
                Stabilizer(
                    pauli="XXXX",
                    data_qubits=((2, 0, 0), (1, 0, 0), (2, 1, 0), (1, 1, 0)),
                    ancilla_qubits=((2, 1, 1),),
                ),
                Stabilizer(
                    pauli="XX",
                    data_qubits=((0, 0, 0), (0, 1, 0)),
                    ancilla_qubits=((0, 1, 1),),
                ),
                Stabilizer(
                    pauli="XX",
                    data_qubits=((2, 1, 0), (2, 2, 0)),
                    ancilla_qubits=((3, 2, 1),),
                ),
                Stabilizer(
                    pauli="ZZ",
                    data_qubits=((2, 0, 0), (1, 0, 0)),
                    ancilla_qubits=((2, 0, 1),),
                ),
                Stabilizer(
                    pauli="ZZ",
                    data_qubits=((1, 2, 0), (0, 2, 0)),
                    ancilla_qubits=((1, 3, 1),),
                ),
            ),
            logical_x_operators=[
                PauliOperator(
                    pauli="XXX", data_qubits=((0, 0, 0), (1, 0, 0), (2, 0, 0))
                )
            ],
            logical_z_operators=[
                PauliOperator(
                    pauli="ZZZ", data_qubits=((0, 0, 0), (0, 1, 0), (0, 2, 0))
                )
            ],
            unique_label="q1",
        )
        self.rot_surf_code_2 = self.rot_surf_code_1.shift(
            position=(4, 0), new_label="q2"
        )
        self.rot_surf_code_2new = Block(
            stabilizers=(
                Stabilizer(
                    pauli="ZZZZ",
                    data_qubits=((5, 0, 0), (5, 1, 0), (4, 0, 0), (4, 1, 0)),
                    ancilla_qubits=((5, 1, 1),),
                ),
                Stabilizer(
                    pauli="ZZZZ",
                    data_qubits=((5, 2, 0), (5, 3, 0), (4, 2, 0), (4, 3, 0)),
                    ancilla_qubits=((5, 3, 1),),
                ),
                Stabilizer(
                    pauli="ZZZZ",
                    data_qubits=((6, 1, 0), (6, 2, 0), (5, 1, 0), (5, 2, 0)),
                    ancilla_qubits=((6, 2, 1),),
                ),
                Stabilizer(
                    pauli="ZZZZ",
                    data_qubits=((6, 3, 0), (6, 4, 0), (5, 3, 0), (5, 4, 0)),
                    ancilla_qubits=((6, 4, 1),),
                ),
                Stabilizer(
                    pauli="XXXX",
                    data_qubits=((5, 1, 0), (4, 1, 0), (5, 2, 0), (4, 2, 0)),
                    ancilla_qubits=((5, 2, 1),),
                ),
                Stabilizer(
                    pauli="XXXX",
                    data_qubits=((5, 3, 0), (4, 3, 0), (5, 4, 0), (4, 4, 0)),
                    ancilla_qubits=((5, 4, 1),),
                ),
                Stabilizer(
                    pauli="XXXX",
                    data_qubits=((6, 0, 0), (5, 0, 0), (6, 1, 0), (5, 1, 0)),
                    ancilla_qubits=((6, 1, 1),),
                ),
                Stabilizer(
                    pauli="XXXX",
                    data_qubits=((6, 2, 0), (5, 2, 0), (6, 3, 0), (5, 3, 0)),
                    ancilla_qubits=((6, 3, 1),),
                ),
                Stabilizer(
                    pauli="XX",
                    data_qubits=((4, 0, 0), (4, 1, 0)),
                    ancilla_qubits=((4, 1, 1),),
                ),
                Stabilizer(
                    pauli="XX",
                    data_qubits=((4, 2, 0), (4, 3, 0)),
                    ancilla_qubits=((4, 3, 1),),
                ),
                Stabilizer(
                    pauli="XX",
                    data_qubits=((6, 1, 0), (6, 2, 0)),
                    ancilla_qubits=((7, 2, 1),),
                ),
                Stabilizer(
                    pauli="XX",
                    data_qubits=((6, 3, 0), (6, 4, 0)),
                    ancilla_qubits=((7, 4, 1),),
                ),
                Stabilizer(
                    pauli="ZZ",
                    data_qubits=((6, 0, 0), (5, 0, 0)),
                    ancilla_qubits=((6, 0, 1),),
                ),
                Stabilizer(
                    pauli="ZZ",
                    data_qubits=((5, 4, 0), (4, 4, 0)),
                    ancilla_qubits=((5, 5, 1),),
                ),
            ),
            logical_x_operators=[
                PauliOperator(
                    pauli="XXX", data_qubits=((4, 0, 0), (5, 0, 0), (6, 0, 0))
                )
            ],
            logical_z_operators=[
                PauliOperator(
                    pauli="ZZZZZ",
                    data_qubits=((4, 0, 0), (4, 1, 0), (4, 2, 0), (4, 3, 0), (4, 4, 0)),
                )
            ],
            unique_label="q2_new",
        )

    def test_create_empty_interpretationstep(self):
        """
        Tests that a new InterpretationStep can be created without providing any
        arguments.
        """
        _ = InterpretationStep()

    def test_get_block(self):
        """
        Tests that the `get_block` function returns the right blocks of the current
        configuration.
        """
        # First create an InterpretationStep with an initial configuration of 2 blocks
        step = InterpretationStep(
            block_history=[
                [self.rot_surf_code_1, self.rot_surf_code_2],
            ]
        )
        # Check that get_block returns the right blocks
        self.assertEqual(step.get_block("q1"), self.rot_surf_code_1)
        self.assertEqual(step.get_block("q2"), self.rot_surf_code_2)

        # Now create an updated InterpretationStep whose block_history field has an
        # additional element with the old block q1 and an updated block q2_new
        step_updated = InterpretationStep(
            block_history=[
                [self.rot_surf_code_1, self.rot_surf_code_2],
                [self.rot_surf_code_1, self.rot_surf_code_2new],
            ]
        )
        self.assertEqual(step_updated.get_block("q1"), self.rot_surf_code_1)
        self.assertEqual(step_updated.get_block("q2_new"), self.rot_surf_code_2new)
        # Check that the old block q2 is not found anymore
        with self.assertRaises(RuntimeError) as cm:
            _ = step_updated.get_block("q2")

        err_msg = "No block with label 'q2' found in the current configuration."
        self.assertIn(err_msg, str(cm.exception))

    def test_update_block_history_and_evolution_MUT(  # pylint: disable=invalid-name
        self,
    ):
        """
        Test that the `update_block_history_and_evolution_MUT` function updates the block history
        correctly.
        """
        step = InterpretationStep(
            block_history=[
                [self.rot_surf_code_1, self.rot_surf_code_2],
            ]
        )
        # Check that if we provide both the new and old blocks, the block history is
        # updated correctly
        step.update_block_history_and_evolution_MUT(
            new_blocks=(self.rot_surf_code_2new,),
            old_blocks=(self.rot_surf_code_2,),
        )
        self.assertEqual(
            step.block_history,
            (
                (self.rot_surf_code_1, self.rot_surf_code_2),
                (self.rot_surf_code_1, self.rot_surf_code_2new),
            ),
        )
        self.assertEqual(
            step.block_evolution,
            {
                self.rot_surf_code_2new.uuid: (self.rot_surf_code_2.uuid,),
            },
        )

        step = InterpretationStep(
            block_history=[
                [self.rot_surf_code_1],
            ]
        )
        # Check that if we provide only the new blocks, the block history is updated
        # the new blocks are added and no block is removed
        step.update_block_history_and_evolution_MUT(
            new_blocks=(self.rot_surf_code_2,),
        )
        self.assertEqual(
            step.block_history,
            (
                (self.rot_surf_code_1,),
                (self.rot_surf_code_1, self.rot_surf_code_2),
            ),
        )
        # No evolution because there is only new blocks
        self.assertEqual(step.block_evolution, {})

        step = InterpretationStep(
            block_history=[
                [self.rot_surf_code_1, self.rot_surf_code_2],
            ]
        )
        # Check that if we provide only the old blocks, the block history is updated
        # no block is added and the old blocks are removed
        step.update_block_history_and_evolution_MUT(
            old_blocks=(self.rot_surf_code_2,),
        )
        self.assertEqual(
            step.block_history,
            (
                (self.rot_surf_code_1, self.rot_surf_code_2),
                (self.rot_surf_code_1,),
            ),
        )
        # No evolution because there is only old blocks
        self.assertEqual(step.block_evolution, {})

        # Check that if the old blocks are not found in the current configuration, an
        # error is raised
        with self.assertRaises(ValueError) as cm:
            step.update_block_history_and_evolution_MUT(
                old_blocks=(self.rot_surf_code_2new,),
            )
        err_msg = "Block 'q2_new' is not in the current block configuration."
        self.assertIn(err_msg, str(cm.exception))
        # Check that if the new blocks are already in the current configuration, an
        # error is raised
        with self.assertRaises(ValueError) as cm:
            step.update_block_history_and_evolution_MUT(
                new_blocks=(self.rot_surf_code_1,),
            )
        err_msg = "Block 'q1' is already in the current block configuration."
        self.assertIn(err_msg, str(cm.exception))

        # Check that two blocks with the same label but different uuid are allowed.
        step = InterpretationStep(
            block_history=[
                [self.rot_surf_code_1, self.rot_surf_code_2],
            ]
        )
        duplicate_q2_label_block = self.rot_surf_code_1.shift(
            position=(0, 0), new_label="q2"
        )
        step.update_block_history_and_evolution_MUT(
            new_blocks=(duplicate_q2_label_block,),
            old_blocks=(self.rot_surf_code_2,),
        )
        self.assertEqual(step.get_block("q2"), duplicate_q2_label_block)
        self.assertNotEqual(step.get_block("q2"), self.rot_surf_code_2)

    def test_update_logical_operator_updates_MUT(self):  # pylint: disable=invalid-name
        """
        Tests the `update_logical_operator_updates_MUT` function.
        """
        # Create a list of logical operator IDs
        logical_x_ids = [str(uuid4()) for _ in range(3)]
        logical_z_ids = [str(uuid4()) for _ in range(3)]
        initial_step = InterpretationStep(
            block_history=[],
            logical_x_operator_updates={
                logical_x_ids[0]: (
                    ("c_(1, 0, 0)", 0),
                    ("c_(1, 0, 1)", 0),
                ),
                logical_x_ids[1]: (
                    ("c_(5, 0, 0)", 0),
                    ("c_(5, 0, 1)", 0),
                ),
                logical_x_ids[2]: (("c_(5, 5, 0)", 0),),
            },
            logical_z_operator_updates={
                logical_z_ids[0]: (
                    ("c_(0, 1, 0)", 0),
                    ("c_(0, 1, 1)", 0),
                ),
                logical_z_ids[1]: (
                    ("c_(13, 1, 0)", 5),
                    ("c_(13, 1, 1)", 5),
                ),
                # logical_z_ids[2] has no updates associated
            },
            logical_x_evolution={
                logical_x_ids[0]: (
                    logical_x_ids[1],
                    logical_x_ids[2],
                ),
            },
            logical_z_evolution={
                logical_z_ids[0]: (
                    logical_z_ids[1],
                    logical_z_ids[2],
                ),
            },
        )

        input_args_and_expected_results = (
            (  # Test that the X update dictionary is populated with the correct values,
                # inherit_updates=False
                {
                    "operator_type": "X",
                    "logical_operator_id": logical_x_ids[0],
                    "new_updates": (("c_(0, 0, 0)", 1),),
                    "inherit_updates": False,
                },
                (
                    ("c_(1, 0, 0)", 0),
                    ("c_(1, 0, 1)", 0),
                    ("c_(0, 0, 0)", 1),
                ),
            ),
            (  # Test that the X update dictionary is populated with the correct values,
                # inherit_updates=True
                {
                    "operator_type": "X",
                    "logical_operator_id": logical_x_ids[0],
                    "new_updates": (("c_(0, 0, 0)", 1),),
                    "inherit_updates": True,
                },
                (
                    ("c_(1, 0, 0)", 0),
                    ("c_(1, 0, 1)", 0),
                    ("c_(0, 0, 0)", 1),
                    ("c_(5, 0, 0)", 0),
                    ("c_(5, 0, 1)", 0),
                    ("c_(5, 5, 0)", 0),
                ),
            ),
            (  # Test that the X update dictionary is populated with the correct values,
                # inherit_updates=True, but the logical operator ID is not in the
                # logical_x_operator_updates dictionary
                {
                    "operator_type": "X",
                    "logical_operator_id": logical_x_ids[1],
                    "new_updates": (("c_(1, 1, 1)", 1),),
                    "inherit_updates": True,
                },
                (
                    ("c_(5, 0, 0)", 0),
                    ("c_(5, 0, 1)", 0),
                    ("c_(1, 1, 1)", 1),
                ),
            ),
            (  # Test that the Z update dictionary is populated with the correct values,
                # inherit_updates=False
                {
                    "operator_type": "Z",
                    "logical_operator_id": logical_z_ids[0],
                    "new_updates": (("c_(2, 2, 0)", 1),),
                    "inherit_updates": False,
                },
                (
                    ("c_(0, 1, 0)", 0),
                    ("c_(0, 1, 1)", 0),
                    ("c_(2, 2, 0)", 1),
                ),
            ),
            (  # Test that the Z update dictionary is populated with the correct values,
                # inherit_updates=True
                {
                    "operator_type": "Z",
                    "logical_operator_id": logical_z_ids[0],
                    "new_updates": (("c_(2, 2, 0)", 1),),
                    "inherit_updates": True,
                },
                (
                    ("c_(0, 1, 0)", 0),
                    ("c_(0, 1, 1)", 0),
                    ("c_(2, 2, 0)", 1),
                    ("c_(13, 1, 0)", 5),
                    ("c_(13, 1, 1)", 5),
                ),
            ),
            (  # Test that the Z update dictionary is populated with the correct values,
                # inherit_updates=True, but the logical operator ID is not in the
                # logical_z_operator_updates dictionary
                {
                    "operator_type": "Z",
                    "logical_operator_id": logical_z_ids[2],
                    "new_updates": (("c_(2, 2, 1)", 4),),
                    "inherit_updates": True,
                },
                (("c_(2, 2, 1)", 4),),
            ),
            (  # Test that an empty updates doesn't change the dictionary
                {
                    "operator_type": "X",
                    "logical_operator_id": logical_x_ids[0],
                    "new_updates": (),
                    "inherit_updates": False,
                },
                (
                    ("c_(1, 0, 0)", 0),
                    ("c_(1, 0, 1)", 0),
                ),
            ),
        )
        # Loop over the input arguments and expected results
        for input_args, expected_result in input_args_and_expected_results:
            # Copy to re-use the initial step
            modified_step = deepcopy(initial_step)
            # Call the function with the input arguments
            modified_step.update_logical_operator_updates_MUT(**input_args)
            check_update_dictionary = (
                modified_step.logical_x_operator_updates
                if input_args["operator_type"] == "X"
                else modified_step.logical_z_operator_updates
            )
            self.assertEqual(
                check_update_dictionary[input_args["logical_operator_id"]],
                expected_result,
            )

    def test_get_channel_MUT(self):  # pylint: disable=invalid-name
        """
        Tests the `get_channel_MUT` function.
        """
        step = InterpretationStep(
            block_history=[
                [self.rot_surf_code_1, self.rot_surf_code_2],
            ]
        )
        ch_labels = [
            "(0, 0, 0)",
            "(0, 0, 0)",
            "(0, 0, 0)",
            "(0, 1, 0)",
            "(0, 0, 1)",
            "(0, 1, 0)",
            "(0, 1, 0)",
            "(0, 0, 0)",
            "(0, 0, 1)",
        ]
        channels = {step.get_channel_MUT(ch_label) for ch_label in ch_labels}
        channel_labels = set(ch.label for ch in channels)
        # Check that only 3 channels are created and that their labels match the unique
        # labels of the `ch_labels` list
        self.assertEqual(len(channels), 3)
        self.assertEqual(channel_labels, set(ch_labels))
        # Check that the `channel_dict` property of the InterpretationStep stores the
        # three channels and that the keys are the unique labels of the channels
        self.assertEqual(set(step.channel_dict.values()), channels)
        self.assertEqual(set(step.channel_dict.keys()), set(ch_labels))

    def test_append_circuit_MUT(self):  # pylint: disable=invalid-name
        """
        Tests the `append_circuit_MUT` function.
        """
        step = InterpretationStep(
            block_history=[
                [self.rot_surf_code_1, self.rot_surf_code_2],
            ]
        )
        self.assertEqual(step.intermediate_circuit_sequence, tuple())
        channels = [step.get_channel_MUT(ch_label) for ch_label in ["d0", "d1"]]
        c1 = Circuit(name="h", channels=channels[0])
        # Check that the append_circuit_MUT function does not return anything
        self.assertEqual(step.append_circuit_MUT(c1), None)
        # Check that after append_circuit_MUT was called, step.circuit has now a single
        # circuit in its `circuit` field and that this circuit is equal to c1
        self.assertEqual(len(step.intermediate_circuit_sequence), 1)
        self.assertEqual(step.intermediate_circuit_sequence[0][0], c1)

        # Add 2 more circuits
        c2 = Circuit(name="cnot", channels=channels)
        c3 = Circuit(
            name="measurement",
            channels=[channels[1], Channel(type=ChannelType.CLASSICAL, label="c0")],
        )
        self.assertEqual(step.append_circuit_MUT(c2), None)
        self.assertEqual(step.append_circuit_MUT(c3), None)
        # step.circuit.circuit contains the previous circuit + the number of newly added
        # circuits
        self.assertEqual(len(step.intermediate_circuit_sequence), 3)
        self.assertEqual(step.intermediate_circuit_sequence[0][0], c1)
        self.assertEqual(step.intermediate_circuit_sequence[1][0], c2)
        self.assertEqual(step.intermediate_circuit_sequence[2][0], c3)

        # Check that the circuit can be added in the same timestep as the last operation
        c4 = Circuit(name="h", channels=channels[0])
        self.assertEqual(step.append_circuit_MUT(c4, same_timeslice=True), None)
        expected_intermediate_circuit_sequence = (
            (c1,),
            (c2,),
            (c3, c4),
        )
        self.assertEqual(
            step.intermediate_circuit_sequence, expected_intermediate_circuit_sequence
        )

        # Check that an error is thrown if the circuit is added in the same timestep as
        # a circuit acting on the same channel
        # Check that the append_circuit_MUT function does not return anything
        with self.assertRaises(ValueError) as cm:
            step.append_circuit_MUT(c4, same_timeslice=True)
        err_msg = (
            "The channels of the new circuit are already in use in the current "
            "timeslice. Please use a new timeslice."
        )
        self.assertIn(err_msg, str(cm.exception))

    def test_get_new_cbit_MUT(self):  # pylint: disable=invalid-name
        """
        Tests the `get_new_cbit_MUT` function.
        """
        step = InterpretationStep(
            block_history=[
                [self.rot_surf_code_1, self.rot_surf_code_2],
            ]
        )
        self.assertEqual(step.get_new_cbit_MUT("c"), ("c", 0))
        self.assertEqual(step.get_new_cbit_MUT("c"), ("c", 1))
        self.assertEqual(step.get_new_cbit_MUT("c"), ("c", 2))
        self.assertEqual(step.get_new_cbit_MUT("d"), ("d", 0))
        self.assertEqual(step.get_new_cbit_MUT("d"), ("d", 1))
        self.assertEqual(step.get_new_cbit_MUT("d"), ("d", 2))
        self.assertEqual(step.get_new_cbit_MUT("reg1"), ("reg1", 0))
        self.assertEqual(step.cbit_counter, {"c": 3, "d": 3, "reg1": 1})

    def test_get_prev_syndrome(self):
        """
        Test the `get_prev_syndrome` function.
        """

        ## 1 - Direct mapping of stabilizers
        int_step = InterpretationStep(
            block_history=[
                [self.rot_surf_code_1],
            ],
            syndromes=[
                Syndrome(
                    stabilizer=stab.uuid,
                    measurements=((f"c_{stab.ancilla_qubits[0]}", i),),
                    block=self.rot_surf_code_1.uuid,
                    round=5 + i,
                )
                for stab in self.rot_surf_code_1.stabilizers
                for i in range(2)
            ],
        )
        # Test that the right syndromes are returned
        for stab in self.rot_surf_code_1.stabilizers:
            self.assertEqual(
                int_step.get_prev_syndrome(stab.uuid, self.rot_surf_code_1.uuid),
                [
                    Syndrome(
                        stabilizer=stab.uuid,
                        measurements=((f"c_{stab.ancilla_qubits[0]}", 1),),
                        block=self.rot_surf_code_1.uuid,
                        round=6,
                    )
                ],
            )

        ## 2 - The stabilizers don't change, but the block does
        # The syndromes from the first block are measured but we get them through the
        # block evolution
        # Create a copy of the block with a different label
        rsc_1_copy = Block(
            stabilizers=self.rot_surf_code_1.stabilizers,
            logical_x_operators=self.rot_surf_code_1.logical_x_operators,
            logical_z_operators=self.rot_surf_code_1.logical_z_operators,
            syndrome_circuits=self.rot_surf_code_1.syndrome_circuits,
            stabilizer_to_circuit=self.rot_surf_code_1.stabilizer_to_circuit,
            unique_label="q1_copy",
        )
        int_step = InterpretationStep(
            block_history=[
                [self.rot_surf_code_1],
                [rsc_1_copy],
            ],
            block_evolution={
                rsc_1_copy.uuid: (self.rot_surf_code_1.uuid,),
            },
            stabilizer_evolution={},
            syndromes=[
                Syndrome(
                    stabilizer=stab.uuid,
                    measurements=((f"c_{stab.ancilla_qubits[0]}", i + 1),),
                    block=self.rot_surf_code_1.uuid,
                    round=i,
                )
                # Syndromes are created for the initial block
                for stab in self.rot_surf_code_1.stabilizers
                for i in range(2)
            ],
        )
        # Test that the right syndromes are returned
        for stab in rsc_1_copy.stabilizers:
            calculated_syndrome = int_step.get_prev_syndrome(stab.uuid, rsc_1_copy.uuid)
            expected_syndrome = [
                Syndrome(
                    stabilizer=stab.uuid,
                    measurements=((f"c_{stab.ancilla_qubits[0]}", 2),),
                    block=self.rot_surf_code_1.uuid,
                    round=1,
                )
            ]
            self.assertEqual(calculated_syndrome, expected_syndrome)

        ## 3 - Both the stabilizers and the block change, e.g. two blocks of same size
        # and in a different position with 1-to-1 stabilizer mapping
        int_step = InterpretationStep(
            block_history=[
                [self.rot_surf_code_1],
                [self.rot_surf_code_2],
            ],
            block_evolution={
                self.rot_surf_code_2.uuid: (self.rot_surf_code_1.uuid,),
            },
            stabilizer_evolution={
                stab2.uuid: (stab1.uuid,)
                for stab1, stab2 in zip(
                    self.rot_surf_code_1.stabilizers,
                    self.rot_surf_code_2.stabilizers,
                    strict=True,
                )
            },
            syndromes=[
                Syndrome(
                    stabilizer=stab.uuid,
                    measurements=((f"c_{stab.ancilla_qubits[0]}", i + 1),),
                    block=self.rot_surf_code_1.uuid,
                    round=i,
                )
                # Syndromes are created for the initial block
                for stab in self.rot_surf_code_1.stabilizers
                for i in range(2)
            ],
        )

        for stab1, stab2 in zip(
            self.rot_surf_code_1.stabilizers,
            self.rot_surf_code_2.stabilizers,
            strict=True,
        ):
            calculated_syndrome = int_step.get_prev_syndrome(
                stab2.uuid, self.rot_surf_code_2.uuid
            )
            expected_syndrome = [
                Syndrome(
                    stabilizer=stab1.uuid,
                    measurements=((f"c_{stab1.ancilla_qubits[0]}", 2),),
                    block=self.rot_surf_code_1.uuid,
                    round=1,
                )
            ]
            self.assertEqual(calculated_syndrome, expected_syndrome)

        ## 4 - The stabilizer and the Block changed, the new stabilizer was measured
        # and then only the Block changed twice before the next measurement.
        rsc_2_copy_a = self.rot_surf_code_2.rename(self.rot_surf_code_2.unique_label)
        rsc_2_copy_b = self.rot_surf_code_2.rename(self.rot_surf_code_2.unique_label)
        int_step = InterpretationStep(
            block_evolution={
                self.rot_surf_code_2.uuid: (self.rot_surf_code_1.uuid,),
                rsc_2_copy_a.uuid: (self.rot_surf_code_2.uuid,),
                rsc_2_copy_b.uuid: (rsc_2_copy_a.uuid,),
            },
            stabilizer_evolution={
                stab2.uuid: (stab1.uuid,)
                for stab1, stab2 in zip(
                    self.rot_surf_code_1.stabilizers,
                    self.rot_surf_code_2.stabilizers,
                    strict=True,
                )
            },
            syndromes=[
                # Syndromes are created for the initial block
                Syndrome(
                    stabilizer=stab.uuid,
                    measurements=((f"c_{stab.ancilla_qubits[0]}", round + 1),),
                    block=self.rot_surf_code_1.uuid,
                    round=round,
                )
                for stab in self.rot_surf_code_1.stabilizers
                for round in range(2)
            ]
            + [
                # Syndromes are created for the second block
                Syndrome(
                    stabilizer=stab.uuid,
                    measurements=((f"c_{stab.ancilla_qubits[0]}", round + 1),),
                    block=self.rot_surf_code_2.uuid,
                    round=round,
                )
                for stab in self.rot_surf_code_2.stabilizers
                for round in range(2)
            ],
        )

        final_block = rsc_2_copy_b
        for stab in final_block.stabilizers:
            calculated_syndrome = int_step.get_prev_syndrome(
                stab.uuid, final_block.uuid
            )
            expected_syndrome = [
                Syndrome(
                    stabilizer=stab.uuid,
                    measurements=((f"c_{stab.ancilla_qubits[0]}", 2),),
                    # Last measured in the second block
                    block=self.rot_surf_code_2.uuid,
                    # 2 rounds created : 0 and 1
                    round=1,
                )
            ]
            self.assertEqual(calculated_syndrome, expected_syndrome)

        ## 5 - The stabilizers have never been measured
        # We should get an empty list

        base_step = InterpretationStep(
            block_history=[[self.rot_surf_code_1]],
            syndromes=tuple(),
        )
        # Test that an empty list is returned
        for stab in self.rot_surf_code_1.stabilizers:
            self.assertEqual(
                base_step.get_prev_syndrome(stab.uuid, self.rot_surf_code_1.uuid),
                [],
            )

    def test_append_syndromes_MUT(self):  # pylint: disable=invalid-name
        """
        Tests the `append_syndromes_MUT` function.
        """
        step = InterpretationStep(
            block_history=[
                [self.rot_surf_code_1, self.rot_surf_code_2],
            ]
        )
        valid_syndromes = (
            Syndrome(
                stabilizer=self.rot_surf_code_1.stabilizers[0].uuid,
                measurements=(
                    ("c", 0),
                    ("c", 1),
                ),
                block=self.rot_surf_code_1.uuid,
                round=0,
            ),
        )
        step.append_syndromes_MUT(valid_syndromes)
        self.assertEqual(step.syndromes, valid_syndromes)

        valid_syndrome = Syndrome(
            stabilizer=self.rot_surf_code_2.stabilizers[0].uuid,
            measurements=(
                ("c", 0),
                ("c", 1),
            ),
            block=self.rot_surf_code_2.uuid,
            round=0,
        )
        step.append_syndromes_MUT(valid_syndrome)
        self.assertEqual(step.syndromes, valid_syndromes + (valid_syndrome,))

        # Test that an error is raised if the syndromes are not of the right type
        invalid_syndrome = ("c", 0)
        with self.assertRaises(TypeError) as cm:
            step.append_syndromes_MUT(invalid_syndrome)
        self.assertIn(
            "All elements in the tuple must be Syndrome objects.", str(cm.exception)
        )

        invalid_syndrome = "hello world"
        with self.assertRaises(TypeError) as cm:
            step.append_syndromes_MUT(invalid_syndrome)
        self.assertIn(
            "Syndrome must be a Syndrome object or a tuple of Syndromes",
            str(cm.exception),
        )

    def test_retrieve_cbits_from_stabilizers(self):
        """
        Tests the `retrieve_cbits_from_stabilizers` function.
        """
        # Empty InterpretationStep
        empty_step = InterpretationStep(
            block_history=[
                [self.rot_surf_code_1],
            ]
        )
        # Test that an error is raised if we try to retrieve cbits
        with self.assertRaises(SyndromeMissingError) as cm:
            empty_step.retrieve_cbits_from_stabilizers(
                self.rot_surf_code_1.stabilizers, self.rot_surf_code_1
            )
        self.assertIn(
            "Could not find a syndrome for some stabilizers.", str(cm.exception)
        )

        # InterpretationStep with syndromes for the stabilizers requested
        step = InterpretationStep(
            block_history=[
                [self.rot_surf_code_1],
            ],
            syndromes=[
                Syndrome(
                    stabilizer=stab.uuid,
                    measurements=((f"c_{stab.ancilla_qubits[0]}", 0),),
                    block=self.rot_surf_code_1.uuid,
                    round=0,
                )
                for stab in self.rot_surf_code_1.stabilizers
            ],
        )
        # Test that all cbits are retrieved correctly
        cbits = step.retrieve_cbits_from_stabilizers(
            self.rot_surf_code_1.stabilizers, self.rot_surf_code_1
        )
        expected_cbits = tuple(
            (f"c_{stab.ancilla_qubits[0]}", 0)
            for stab in self.rot_surf_code_1.stabilizers
        )
        self.assertEqual(cbits, expected_cbits)

        # Test that only the right cbits are returned in case we ask for a subset:
        # Add an extra syndrome to the step for the first stabilizer
        step.syndromes += (
            Syndrome(
                stabilizer=self.rot_surf_code_1.stabilizers[0].uuid,
                measurements=(
                    (f"c_{self.rot_surf_code_1.stabilizers[0].ancilla_qubits[0]}", 1),
                ),
                block=self.rot_surf_code_1.uuid,
                round=1,
            ),
        )
        # Retrieve only the cbits for the first two stabilizer
        cbits = step.retrieve_cbits_from_stabilizers(
            self.rot_surf_code_1.stabilizers[:2], self.rot_surf_code_1
        )
        # The Cbit associated to the first stabilizer is associated to a new measurement, round 1
        # The Cbit associated to the second stabilizer is associated the measurement of round 0
        expected_cbits = (
            (f"c_{self.rot_surf_code_1.stabilizers[0].ancilla_qubits[0]}", 1),
            (f"c_{self.rot_surf_code_1.stabilizers[1].ancilla_qubits[0]}", 0),
        )
        self.assertEqual(cbits, expected_cbits)


if __name__ == "__main__":
    unittest.main()
