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

from uuid import uuid4
from copy import deepcopy
import pytest
from loom.eka import (
    Circuit,
    Channel,
    ChannelType,
    Block,
    Stabilizer,
)
from loom.interpreter import InterpretationStep, Syndrome
from loom.eka.utilities import SyndromeMissingError

# pylint: disable=redefined-outer-name, too-many-lines


@pytest.fixture(scope="module")
def two_rsc_blocks(n_rsc_block_factory) -> list[Block]:
    """Fixture for two rotated surface code blocks."""
    rsc_blocks = n_rsc_block_factory(2)
    rsc_blocks[0] = rsc_blocks[0].rename("q1")
    rsc_blocks[1] = rsc_blocks[1].rename("q2")
    return rsc_blocks


class TestInterpretationStep:
    """
    Test for the InterpretationStep class.
    """

    def test_create_empty_interpretationstep(self):
        """
        Tests that a new InterpretationStep can be created without providing any
        arguments.
        """
        i = InterpretationStep.create(())
        assert isinstance(i, InterpretationStep)

    def test_get_block(self, two_rsc_blocks):
        """
        Tests that the `get_block` function returns the right blocks of the current
        configuration.
        """

        rsc_blocks = two_rsc_blocks
        # First create an InterpretationStep with an initial configuration of 2 blocks
        step = InterpretationStep.create([rsc_blocks[0], rsc_blocks[1]])
        # Check that get_block returns the right blocks (the "is" operator checks that
        # the returned block is the same instance as the original)
        assert step.get_block("q1") is rsc_blocks[0]
        assert step.get_block("q2") is rsc_blocks[1]

        new_block2 = rsc_blocks[1].rename("q2_new")
        # Now create an updated InterpretationStep whose block_history field has an
        # additional element with the old block q1 and an updated block q2_new
        step_updated = InterpretationStep.create([rsc_blocks[0], rsc_blocks[1]])
        step_updated.update_block_history_and_evolution_MUT(
            new_blocks=(new_block2,),
            old_blocks=(rsc_blocks[1],),
        )
        assert step_updated.get_block("q1") is rsc_blocks[0]
        assert step_updated.get_block("q2_new") is new_block2
        # Check that the old block q2 is not found anymore
        with pytest.raises(RuntimeError) as cm:
            _ = step_updated.get_block("q2")

        err_msg = "No block with label 'q2' found in the current configuration."
        assert err_msg in str(cm.value)

    def test_update_block_history_and_evolution_MUT(  # pylint: disable=invalid-name
        self, two_rsc_blocks, subtests
    ):
        """
        Test that the `update_block_history_and_evolution_MUT` function updates the
        block history correctly.
        """

        new_2nd_block = two_rsc_blocks[1].rename("q2_new")

        # Check that if we provide both the new and old blocks, the block history is
        # updated correctly.
        with subtests.test(msg="simple update of the history"):
            step = InterpretationStep.create(two_rsc_blocks)

            step.update_block_history_and_evolution_MUT(
                new_blocks=(new_2nd_block,),
                old_blocks=(two_rsc_blocks[1],),
            )

            # Check that the blocks are updated correctly
            assert step.get_block("q1") is two_rsc_blocks[0]
            assert step.get_block("q2_new") is new_2nd_block
            # Check that old block q2 is no longer present
            with pytest.raises(RuntimeError):
                step.get_block("q2")

            assert step.block_evolution == {
                new_2nd_block.uuid: (two_rsc_blocks[1].uuid,),
            }

        # Check that if we provide only the new blocks, the block history is updated
        # the new blocks are added and no block is removed
        with subtests.test(msg="add new block to the history"):

            step = InterpretationStep.create([two_rsc_blocks[0]])

            step.update_block_history_and_evolution_MUT(
                new_blocks=(two_rsc_blocks[1],),
            )

            # Check that both blocks are now present
            assert step.get_block("q1") is two_rsc_blocks[0]
            assert step.get_block("q2") is two_rsc_blocks[1]

            # No evolution because there is only new blocks
            assert step.block_evolution == {}

        # Check that if we provide only the old blocks, the block history is updated
        # no block is added and the old blocks are removed
        with subtests.test(msg="remove block from the history"):

            step = InterpretationStep.create([two_rsc_blocks[0], two_rsc_blocks[1]])

            step.update_block_history_and_evolution_MUT(
                old_blocks=(two_rsc_blocks[1],),
            )

            # Check that only block 0 is present now
            assert step.get_block("q1") is two_rsc_blocks[0]
            with pytest.raises(RuntimeError):
                step.get_block("q2")

            # No evolution because there is only old blocks
            assert step.block_evolution == {}

        with subtests.test(msg="check error raised when updating the history"):
            # Create a fresh step for error checking
            step = InterpretationStep.create(two_rsc_blocks)

            # Check that if the old blocks are not found in the current configuration,
            # an error is raised
            with pytest.raises(RuntimeError) as cm:
                step.update_block_history_and_evolution_MUT(
                    old_blocks=(new_2nd_block,),
                )

            err_msg = "Block 'q2_new' is not in the current block configuration."
            assert err_msg in str(cm.value) or "not present" in str(cm.value).lower()

            # Check that if the new blocks are already in the current configuration, an
            # error is raised
            with pytest.raises(RuntimeError) as cm:
                step.update_block_history_and_evolution_MUT(
                    new_blocks=(two_rsc_blocks[0],),
                )
            err_msg = "Block 'q1' is already in the current block configuration."
            assert err_msg in str(cm.value) or "already" in str(cm.value).lower()

        # Check that two blocks with the same label but different uuid are allowed.
        with subtests.test(msg="duplicate block labels allowed"):
            step = InterpretationStep.create([two_rsc_blocks[0], two_rsc_blocks[1]])
            duplicate_q2_label_block = two_rsc_blocks[0].shift(
                position=(0, 0), new_label="q2"
            )
            step.update_block_history_and_evolution_MUT(
                new_blocks=(duplicate_q2_label_block,),
                old_blocks=(two_rsc_blocks[1],),
            )

            assert step.get_block("q2") == duplicate_q2_label_block
            assert step.get_block("q2") != two_rsc_blocks[1]

    def test_update_logical_operator_updates_MUT(
        self, subtests
    ):  # pylint: disable=invalid-name
        """
        Tests the `update_logical_operator_updates_MUT` function.
        """
        # Create a list of logical operator IDs
        logical_x_ids = [str(uuid4()) for _ in range(3)]
        logical_z_ids = [str(uuid4()) for _ in range(3)]
        initial_step = InterpretationStep.create(
            (),
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
                "X operator, inherit_updates=False",
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
                "X operator, inherit_updates=True",
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
                "X operator, inherit_updates=True, operator ID not in dictionary",
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
                "Z operator, inherit_updates=False",
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
                "Z operator, inherit_updates=True",
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
                "Z operator, inherit_updates=True, operator ID not in dictionary",
                {
                    "operator_type": "Z",
                    "logical_operator_id": logical_z_ids[2],
                    "new_updates": (("c_(2, 2, 1)", 4),),
                    "inherit_updates": True,
                },
                (("c_(2, 2, 1)", 4),),
            ),
            (  # Test that an empty updates doesn't change the dictionary
                "X operator, empty updates",
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
        # Loop over the input arguments and expected results using subtests
        for test_name, input_args, expected_result in input_args_and_expected_results:
            with subtests.test(msg=test_name):
                # Copy to re-use the initial step
                modified_step = deepcopy(initial_step)
                # Call the function with the input arguments
                modified_step.update_logical_operator_updates_MUT(**input_args)
                check_update_dictionary = (
                    modified_step.logical_x_operator_updates
                    if input_args["operator_type"] == "X"
                    else modified_step.logical_z_operator_updates
                )
                assert (
                    check_update_dictionary[input_args["logical_operator_id"]]
                    == expected_result
                )

    def test_get_channel_MUT(self, two_rsc_blocks):  # pylint: disable=invalid-name
        """
        Tests the `get_channel_MUT` function.
        """
        step = InterpretationStep.create(two_rsc_blocks)
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
        assert len(channels) == 3
        assert channel_labels == set(ch_labels)
        # Check that the `channel_dict` property of the InterpretationStep stores the
        # three channels and that the keys are the unique labels of the channels
        assert set(step.channel_dict.values()) == channels
        assert set(step.channel_dict.keys()) == set(ch_labels)

    def test_append_circuit_MUT(self, two_rsc_blocks):  # pylint: disable=invalid-name
        """
        Tests the `append_circuit_MUT` function.
        """
        step = InterpretationStep.create(two_rsc_blocks)
        assert step.intermediate_circuit_sequence == tuple()
        channels = [step.get_channel_MUT(ch_label) for ch_label in ["d0", "d1"]]
        c1 = Circuit(name="h", channels=channels[0])
        # Check that the append_circuit_MUT function does not return anything
        assert step.append_circuit_MUT(c1) is None
        # Check that after append_circuit_MUT was called, step.circuit has now a single
        # circuit in its `circuit` field and that this circuit is equal to c1
        assert len(step.intermediate_circuit_sequence) == 1
        assert step.intermediate_circuit_sequence[0][0] == c1

        # Add 2 more circuits
        c2 = Circuit(name="cnot", channels=channels)
        c3 = Circuit(
            name="measurement",
            channels=[channels[1], Channel(type=ChannelType.CLASSICAL, label="c0")],
        )
        assert step.append_circuit_MUT(c2) is None
        assert step.append_circuit_MUT(c3) is None
        # step.circuit.circuit contains the previous circuit + the number of newly added
        # circuits
        assert len(step.intermediate_circuit_sequence) == 3
        assert step.intermediate_circuit_sequence[0][0] == c1
        assert step.intermediate_circuit_sequence[1][0] == c2
        assert step.intermediate_circuit_sequence[2][0] == c3

        # Check that the circuit can be added in the same timestep as the last operation
        c4 = Circuit(name="h", channels=channels[0])
        assert step.append_circuit_MUT(c4, same_timeslice=True) is None

        expected_intermediate_circuit_sequence = (
            (c1,),
            (c2,),
            (c3, c4),
        )

        assert (
            step.intermediate_circuit_sequence == expected_intermediate_circuit_sequence
        )

        # Check that an error is thrown if the circuit is added in the same timestep as
        # a circuit acting on the same channel
        # Check that the append_circuit_MUT function does not return anything
        with pytest.raises(ValueError) as cm:
            step.append_circuit_MUT(c4, same_timeslice=True)

        err_msg = (
            "The channels of the new circuit are already in use in the current "
            "timeslice. Please use a new timeslice."
        )
        assert err_msg in str(cm.value)

    def test_get_new_cbit_MUT(self, two_rsc_blocks):  # pylint: disable=invalid-name
        """
        Tests the `get_new_cbit_MUT` function.
        """
        step = InterpretationStep.create(two_rsc_blocks)
        assert step.get_new_cbit_MUT("c") == ("c", 0)
        assert step.get_new_cbit_MUT("c") == ("c", 1)
        assert step.get_new_cbit_MUT("c") == ("c", 2)
        assert step.get_new_cbit_MUT("d") == ("d", 0)
        assert step.get_new_cbit_MUT("d") == ("d", 1)
        assert step.get_new_cbit_MUT("d") == ("d", 2)
        assert step.get_new_cbit_MUT("reg1") == ("reg1", 0)
        assert step.cbit_counter == {"c": 3, "d": 3, "reg1": 1}

    def test_get_prev_syndrome(self, two_rsc_blocks):
        """
        Test the `get_prev_syndrome` function.
        """
        rsc_block = two_rsc_blocks[0]
        ## 1 - Direct mapping of stabilizers
        int_step = InterpretationStep.create(
            [rsc_block],
            syndromes=[
                Syndrome(
                    stabilizer=stab.uuid,
                    measurements=((f"c_{stab.ancilla_qubits[0]}", i),),
                    block=rsc_block.uuid,
                    round=5 + i,
                )
                for stab in rsc_block.stabilizers
                for i in range(2)
            ],
        )

        # Test that the right syndromes are returned
        for stab in rsc_block.stabilizers:
            assert int_step.get_prev_syndrome(stab.uuid, rsc_block.uuid) == [
                Syndrome(
                    stabilizer=stab.uuid,
                    measurements=((f"c_{stab.ancilla_qubits[0]}", 1),),
                    block=rsc_block.uuid,
                    round=6,
                )
            ]

        ## 2 - The stabilizers don't change, but the block does
        # The syndromes from the first block are measured but we get them through the
        # block evolution
        # Create a copy of the block with a different label
        rsc_1_copy = rsc_block.rename("q1_copy")

        int_step = InterpretationStep.create(
            [rsc_block],
            syndromes=[
                Syndrome(
                    stabilizer=stab.uuid,
                    measurements=((f"c_{stab.ancilla_qubits[0]}", i + 1),),
                    block=rsc_1_copy.uuid,
                    round=i,
                )
                # Syndromes are created for the initial block
                for stab in rsc_block.stabilizers
                for i in range(2)
            ],
        )
        int_step.update_block_history_and_evolution_MUT(
            new_blocks=(rsc_1_copy,),
            old_blocks=(rsc_block,),
            update_evolution=True,
        )

        # Test that the right syndromes are returned
        for stab in rsc_1_copy.stabilizers:
            calculated_syndrome = int_step.get_prev_syndrome(stab.uuid, rsc_1_copy.uuid)
            expected_syndrome = [
                Syndrome(
                    stabilizer=stab.uuid,
                    measurements=((f"c_{stab.ancilla_qubits[0]}", 2),),
                    block=rsc_1_copy.uuid,
                    round=1,
                )
            ]
            assert calculated_syndrome == expected_syndrome

        ## 3 - Both the stabilizers and the block change, e.g. two blocks of same size
        # and in a different position with 1-to-1 stabilizer mapping
        int_step = InterpretationStep.create(
            [two_rsc_blocks[0]],
            syndromes=[
                Syndrome(
                    stabilizer=stab.uuid,
                    measurements=((f"c_{stab.ancilla_qubits[0]}", i + 1),),
                    block=two_rsc_blocks[0].uuid,
                    round=i,
                )
                # Syndromes are created for the initial block
                for stab in two_rsc_blocks[0].stabilizers
                for i in range(2)
            ],
        )
        int_step.update_block_history_and_evolution_MUT(
            new_blocks=(two_rsc_blocks[1],),
            old_blocks=(two_rsc_blocks[0],),
            update_evolution=True,
        )
        # Set up stabilizer evolution manually
        int_step.stabilizer_evolution = {
            stab2.uuid: (stab1.uuid,)
            for stab1, stab2 in zip(
                two_rsc_blocks[0].stabilizers,
                two_rsc_blocks[1].stabilizers,
                strict=True,
            )
        }

        for stab1, stab2 in zip(
            two_rsc_blocks[0].stabilizers,
            two_rsc_blocks[1].stabilizers,
            strict=True,
        ):
            calculated_syndrome = int_step.get_prev_syndrome(
                stab2.uuid, two_rsc_blocks[1].uuid
            )
            expected_syndrome = [
                Syndrome(
                    stabilizer=stab1.uuid,
                    measurements=((f"c_{stab1.ancilla_qubits[0]}", 2),),
                    block=two_rsc_blocks[0].uuid,
                    round=1,
                )
            ]
            assert calculated_syndrome == expected_syndrome

        ## 4 - The stabilizer and the Block changed, the new stabilizer was measured
        # and then only the Block changed twice before the next measurement.
        rsc_2_copy_a = two_rsc_blocks[1].rename(two_rsc_blocks[1].unique_label)
        rsc_2_copy_b = two_rsc_blocks[1].rename(two_rsc_blocks[1].unique_label)
        int_step = InterpretationStep.create(
            [two_rsc_blocks[0]],
            syndromes=[
                # Syndromes are created for the initial block
                Syndrome(
                    stabilizer=stab.uuid,
                    measurements=((f"c_{stab.ancilla_qubits[0]}", round + 1),),
                    block=two_rsc_blocks[0].uuid,
                    round=round,
                )
                for stab in two_rsc_blocks[0].stabilizers
                for round in range(2)
            ]
            + [
                # Syndromes are created for the second block
                Syndrome(
                    stabilizer=stab.uuid,
                    measurements=((f"c_{stab.ancilla_qubits[0]}", round + 1),),
                    block=two_rsc_blocks[1].uuid,
                    round=round,
                )
                for stab in two_rsc_blocks[1].stabilizers
                for round in range(2)
            ],
        )
        # Manually set up the complex block evolution
        int_step.block_registry.update(
            {
                two_rsc_blocks[1].uuid: two_rsc_blocks[1],
                rsc_2_copy_a.uuid: rsc_2_copy_a,
                rsc_2_copy_b.uuid: rsc_2_copy_b,
            }
        )
        int_step.block_evolution = {
            two_rsc_blocks[1].uuid: (two_rsc_blocks[0].uuid,),
            rsc_2_copy_a.uuid: (two_rsc_blocks[1].uuid,),
            rsc_2_copy_b.uuid: (rsc_2_copy_a.uuid,),
        }
        int_step.stabilizer_evolution = {
            stab2.uuid: (stab1.uuid,)
            for stab1, stab2 in zip(
                two_rsc_blocks[0].stabilizers,
                two_rsc_blocks[1].stabilizers,
                strict=True,
            )
        }

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
                    block=two_rsc_blocks[1].uuid,
                    # 2 rounds created : 0 and 1
                    round=1,
                )
            ]
            assert calculated_syndrome == expected_syndrome

        ## 5 - The stabilizers have never been measured
        # We should get an empty list

        base_step = InterpretationStep.create([two_rsc_blocks[0]], syndromes=tuple())
        # Test that an empty list is returned
        for stab in two_rsc_blocks[0].stabilizers:
            assert base_step.get_prev_syndrome(stab.uuid, two_rsc_blocks[0].uuid) == []

    def test_append_syndromes_MUT(self, two_rsc_blocks):  # pylint: disable=invalid-name
        """
        Tests the `append_syndromes_MUT` function.
        """
        step = InterpretationStep.create([two_rsc_blocks[0], two_rsc_blocks[1]])
        valid_syndromes = (
            Syndrome(
                stabilizer=two_rsc_blocks[0].stabilizers[0].uuid,
                measurements=(
                    ("c", 0),
                    ("c", 1),
                ),
                block=two_rsc_blocks[0].uuid,
                round=0,
            ),
        )
        step.append_syndromes_MUT(valid_syndromes)

        assert step.syndromes == valid_syndromes

        valid_syndrome = Syndrome(
            stabilizer=two_rsc_blocks[1].stabilizers[0].uuid,
            measurements=(
                ("c", 0),
                ("c", 1),
            ),
            block=two_rsc_blocks[1].uuid,
            round=0,
        )
        step.append_syndromes_MUT(valid_syndrome)

        assert step.syndromes == valid_syndromes + (valid_syndrome,)

        # Test that an error is raised if the syndromes are not of the right type
        invalid_syndrome = ("c", 0)
        with pytest.raises(TypeError) as cm:
            step.append_syndromes_MUT(invalid_syndrome)

        err_msg = "All elements in the tuple must be Syndrome objects."
        assert err_msg in str(cm.value)

        invalid_syndrome = "hello world"
        with pytest.raises(TypeError) as cm:
            step.append_syndromes_MUT(invalid_syndrome)
        assert "Syndrome must be a Syndrome object or a tuple of Syndromes" in str(
            cm.value
        )

    def test_retrieve_cbits_from_stabilizers(self, two_rsc_blocks):
        """
        Tests the `retrieve_cbits_from_stabilizers` function.
        """
        # Empty InterpretationStep
        empty_step = InterpretationStep.create([two_rsc_blocks[0]])
        # Test that an error is raised if we try to retrieve cbits
        with pytest.raises(SyndromeMissingError) as cm:
            empty_step.retrieve_cbits_from_stabilizers(
                two_rsc_blocks[0].stabilizers, two_rsc_blocks[0]
            )
        assert "Could not find a syndrome for some stabilizers." in str(cm.value)

        # InterpretationStep with syndromes for the stabilizers requested
        step = InterpretationStep.create(
            [two_rsc_blocks[0]],
            syndromes=[
                Syndrome(
                    stabilizer=stab.uuid,
                    measurements=((f"c_{stab.ancilla_qubits[0]}", 0),),
                    block=two_rsc_blocks[0].uuid,
                    round=0,
                )
                for stab in two_rsc_blocks[0].stabilizers
            ],
        )
        # Test that all cbits are retrieved correctly
        cbits = step.retrieve_cbits_from_stabilizers(
            two_rsc_blocks[0].stabilizers, two_rsc_blocks[0]
        )
        expected_cbits = tuple(
            (f"c_{stab.ancilla_qubits[0]}", 0) for stab in two_rsc_blocks[0].stabilizers
        )
        assert cbits == expected_cbits

        # Test that only the right cbits are returned in case we ask for a subset:
        # Add an extra syndrome to the step for the first stabilizer
        step.syndromes += (
            Syndrome(
                stabilizer=two_rsc_blocks[0].stabilizers[0].uuid,
                measurements=(
                    (f"c_{two_rsc_blocks[0].stabilizers[0].ancilla_qubits[0]}", 1),
                ),
                block=two_rsc_blocks[0].uuid,
                round=1,
            ),
        )
        # Retrieve only the cbits for the first two stabilizer
        cbits = step.retrieve_cbits_from_stabilizers(
            two_rsc_blocks[0].stabilizers[:2], two_rsc_blocks[0]
        )
        # The Cbit associated to the first stabilizer is associated to a new
        # measurement, round 1
        # The Cbit associated to the second stabilizer is associated the measurement of
        # round 0
        expected_cbits = (
            (f"c_{two_rsc_blocks[0].stabilizers[0].ancilla_qubits[0]}", 1),
            (f"c_{two_rsc_blocks[0].stabilizers[1].ancilla_qubits[0]}", 0),
        )
        assert cbits == expected_cbits

    def test_composite_operation_sessions(self, two_rsc_blocks):
        """
        Tests the begin_composite_operation_session_MUT and
        end_composite_operation_session_MUT functions.
        """
        step = InterpretationStep.create([two_rsc_blocks[0], two_rsc_blocks[1]])
        had0 = Circuit(
            name="h",
            channels=[step.get_channel_MUT(two_rsc_blocks[0].data_qubits[0])],
        )
        # Initially, no composite operation sessions
        assert step.composite_operation_session_stack == []

        with pytest.raises(ValueError) as cm:
            step.end_composite_operation_session_MUT()
        assert (
            "No composite operation session to end. Please begin a session first."
            in str(cm.value)
        )

        # Create a composite operation session
        step.begin_composite_operation_session_MUT(
            same_timeslice=False, circuit_name="ses0_circuit_name"
        )
        ses0 = step.composite_operation_session_stack[-1]
        assert ses0.start_timeslice_index == 0

        # Append a circuit of duration 1 to the step
        ses_0_circuit_to_append = Circuit(name="1 hadamard", circuit=[[had0.clone()]])
        step.append_circuit_MUT(ses_0_circuit_to_append)

        # Begin a second composite operation session
        step.begin_composite_operation_session_MUT(
            same_timeslice=False, circuit_name="ses1_circuit_name"
        )
        ses1 = step.composite_operation_session_stack[-1]
        assert ses1.start_timeslice_index == 1
        assert step.composite_operation_session_stack == [ses0, ses1]

        # Append a circuit of duration 2 to the step
        ses_1_circuit_to_append = Circuit(
            name="2 hadamards", circuit=[[had0.clone()], [had0.clone()]]
        )
        step.append_circuit_MUT(ses_1_circuit_to_append)

        # End the last session first and append the circuit back to the step
        ses_1_circuit = step.end_composite_operation_session_MUT()
        step.append_circuit_MUT(ses_1_circuit, False)
        # End the first session and append the circuit back to the step
        ses_0_circuit = step.end_composite_operation_session_MUT()
        step.append_circuit_MUT(ses_0_circuit, False)

        # Check that the popped ses0_circuit is correct
        assert ses_0_circuit == Circuit(
            name="ses0_circuit_name",
            circuit=[
                [had0.clone()],
                [had0.clone()],
                [had0.clone()],
            ],
        )

        # Check that the circuits have the right names and durations
        assert ses_1_circuit.name == "ses1_circuit_name"
        assert ses_1_circuit.duration == ses_1_circuit_to_append.duration
        assert ses_0_circuit.name == "ses0_circuit_name"
        assert (
            ses_0_circuit.duration
            == ses_0_circuit_to_append.duration + ses_1_circuit.duration
        )
        assert step.composite_operation_session_stack == []

    def test_timestamp_method_simple_ops(self, two_rsc_blocks):
        """
        Tests the get_timestamp method function of the InterpretationStep class.

        "circuit_a": |--3--|                                time = 3
        "circuit_b":       |--2--|                          time = 5
        "circuit_c":       |-----5-----|                    time = 8
        "circuit_d":       |---3---|                        time = 6
        "circuit_e":                   |---3---|            time = 11
        """
        step = InterpretationStep.create([two_rsc_blocks[0], two_rsc_blocks[1]])
        circuit_list = [
            Circuit("hadamard", channels=[step.get_channel_MUT(q)])
            for q in two_rsc_blocks[0].qubits + two_rsc_blocks[1].qubits
        ]

        # No circuits added yet, time should be 0
        assert step.get_timestamp() == 0

        # - circuit_a
        step.append_circuit_MUT(
            Circuit(
                name="circuit_a",
                circuit=[circuit_list[0], circuit_list[1], circuit_list[2]],
            ),
            same_timeslice=False,
        )
        assert step.get_timestamp() == 3

        # - circuit_b
        step.append_circuit_MUT(
            Circuit(
                name="circuit_b",
                circuit=[circuit_list[3], circuit_list[4]],
            ),
            same_timeslice=False,
        )
        assert step.get_timestamp() == 5

        # - circuit_c
        step.append_circuit_MUT(
            Circuit(
                name="circuit_c",
                circuit=[circuit_list[i] for i in range(5, 10)],
            ),
            same_timeslice=True,
        )
        assert step.get_timestamp() == 8

        # - circuit_d
        step.append_circuit_MUT(
            Circuit(
                name="circuit_d",
                circuit=[circuit_list[10], circuit_list[11], circuit_list[12]],
            ),
            same_timeslice=True,
        )
        assert step.get_timestamp() == 6

        # - circuit_e
        step.append_circuit_MUT(
            Circuit(
                name="circuit_e",
                circuit=[circuit_list[13], circuit_list[14], circuit_list[15]],
            ),
            same_timeslice=False,
        )
        assert step.get_timestamp() == 11

    def test_timestamp_method_composite_ops(self, two_rsc_blocks):
        """
        Tests the get_timestamp method function of the InterpretationStep class in
        presence of composite operation sessions.

        Base circuit:
            "some circuit":                   |--2--|                       time = 2

        Session 0:
                "ses0 circuit":                     |---------9---------|   time = 11

        Session 1:
                "ses1 first circuit":               |--2--|                 time = 4

                Nested session 0:
                        "parallel circuit_0":             |----4----|       time = 8
                Nested session 1:
                        "parallel circuit_1":             |--2--|           time = 6
                Nested session 2:
                        "parallel circuit_2":             |-----5-----|     time = 9
                Nested session 3:
                        "parallel circuit_3":             |-1-|             time = 5

        NOTE: Session ses1 has same_timeslice=True, so its circuit runs in parallel
        with the circuit of ses0. Also, it doesn't end until after all parallel
        circuits have ended.

        After all parallel circuits:
        - After closing ses1: time = 9
        """
        # SetUp the InterpretationStep and get some Circuits to work with
        step = InterpretationStep.create([two_rsc_blocks[0], two_rsc_blocks[1]])
        circuit_list = [
            Circuit("hadamard", channels=[step.get_channel_MUT(q)])
            for q in two_rsc_blocks[0].qubits + two_rsc_blocks[1].qubits
        ]

        # No circuits added yet, time should be 0
        current_time = 0
        assert step.get_timestamp() == 0

        # Append a Circuit of duration 2 and check that time is updated correctly
        step.append_circuit_MUT(
            Circuit(
                name="some circuit",
                circuit=[circuit_list[0], circuit_list[1]],
            )
        )
        current_time = 2
        assert step.get_timestamp() == current_time

        # --- Begin Session 0 ---
        step.begin_composite_operation_session_MUT(
            same_timeslice=False, circuit_name="ses0_circuit_name"
        )
        # Append a Circuit of duration 9
        step.append_circuit_MUT(
            Circuit(
                name="ses0 circuit",
                circuit=[circuit_list[i] for i in range(2, 2 + 9)],
            )
        )
        current_time = 11
        assert step.get_timestamp() == current_time
        ses0_circuit = step.end_composite_operation_session_MUT()
        step.append_circuit_MUT(ses0_circuit, False)
        # --- End Session 0 ---

        # --- Begin Session 1 ---

        # Start another session in parallel with previous one with same_timeslice=True
        step.begin_composite_operation_session_MUT(
            same_timeslice=True, circuit_name="ses1 circuit"
        )
        # Append a Circuit of duration 2 that will run in the same timeslice as the
        # previous circuit
        step.append_circuit_MUT(
            Circuit(
                name="ses1 first circuit",
                circuit=[circuit_list[11], circuit_list[12]],
            )
        )
        current_time = 2 + 2
        assert step.get_timestamp() == current_time

        # Now we shall start and end some sessions that will be in parallel
        nested_sessions_args_list = [
            # First session needs to have same_timeslice = False because it is the first
            # session running in parallel
            {"duration": 4, "same_timeslice": False, "time_after": 8},
            {"duration": 2, "same_timeslice": True, "time_after": 6},
            {"duration": 5, "same_timeslice": True, "time_after": 9},
            {"duration": 1, "same_timeslice": True, "time_after": 5},
        ]

        circuit_list_idx = 13
        for idx, session_args in enumerate(nested_sessions_args_list):
            # --- Begin Nested Session {idx} ---
            step.begin_composite_operation_session_MUT(
                same_timeslice=session_args["same_timeslice"],
                circuit_name=f"parallel_circuit_name_{idx}",
            )
            # Append a Circuit of the specified duration
            step.append_circuit_MUT(
                Circuit(
                    name=f"parallel circuit {idx}",
                    circuit=[
                        [circuit_list[circuit_list_idx + i]]
                        for i in range(session_args["duration"])
                    ],
                )
            )
            circuit_list_idx += session_args["duration"]

            # Since these run in parallel to the previous circuit, time should be -
            # previous duration + new duration
            assert step.get_timestamp() == session_args["time_after"]
            # End the last session and append the circuit back to the step
            ses_circuit = step.end_composite_operation_session_MUT()
            step.append_circuit_MUT(ses_circuit, session_args["same_timeslice"])
            # --- End Nested Session {idx} ---

        # Finally, end Session 1 and check that the time is updated correctly
        ses1_circuit = step.end_composite_operation_session_MUT()
        step.append_circuit_MUT(ses1_circuit, True)
        # Since Session 1 was the last thing that was appended to the step, the
        # timestamp should correspond to the end of Session 1 circuit, i.e. 9
        assert step.get_timestamp() == 9

        # --- End Session 1 ---

    def test_append_circuit_mut_raises_error_sessions(self):
        """
        Test that append_circuit_MUT raises a ValueError when trying to add the first
        circuit of a composite operation to the same timeslice.
        """
        # 1. Initialize an InterpretationStep
        step = InterpretationStep.create(())

        # 2. Append an initial circuit
        initial_circuit = Circuit(name="initial")
        step.append_circuit_MUT(initial_circuit, same_timeslice=False)

        # 3. Start a new composite operation session
        composite_circuit_name = "composite_op"
        step.begin_composite_operation_session_MUT(
            same_timeslice=False, circuit_name=composite_circuit_name
        )

        # 4. Attempt to append another circuit with same_timeslice=True
        # This should fail because it's the first circuit of a new composite session.
        next_circuit = Circuit(name="next")
        expected_error_msg = (
            "The first circuit of a composite operation session cannot be "
            "added to the same timeslice as the previous circuit. Please set "
            "same_timeslice to False for the first circuit of composite operation "
            f"with circuit name '{composite_circuit_name}'."
        )

        # 5. Assert that the expected ValueError is raised
        with pytest.raises(ValueError) as cm:
            step.append_circuit_MUT(next_circuit, same_timeslice=True)
        assert expected_error_msg in str(cm.value)

    @pytest.mark.parametrize(
        "method_name, attr_name",
        [
            (
                "update_reset_single_qubit_stabilizers_MUT",
                "reset_single_qubit_stabilizers",
            ),
            (
                "update_measured_single_qubit_stabilizers_MUT",
                "measured_single_qubit_stabilizers",
            ),
        ],
    )
    def test_update_single_qubit_stabilizers(
        self, method_name, attr_name, two_rsc_blocks
    ):
        # pylint: disable=line-too-long
        """
        Test that the update functions for single qubit stabilizers update the
        corresponding attributes correctly.
        Namely:
        update_reset_single_qubit_stabilizers_MUT -> reset_single_qubit_stabilizers
        update_measured_single_qubit_stabilizers_MUT -> measured_single_qubit_stabilizers /
        """

        block = two_rsc_blocks[0]
        # 1. Initialize InterpretationStep
        step = InterpretationStep.create(
            [block],
        )
        block_id = block.uuid

        # 2. Add single-qubit stabilizers
        single_qubit_stabs = {
            Stabilizer(
                pauli="Z",
                data_qubits=((-1, -1),),
            ),
            Stabilizer(
                pauli="X",
                data_qubits=((-1, -2),),
            ),
        }

        update_method = getattr(step, method_name)

        update_method(block_id, single_qubit_stabs)
        assert (
            getattr(step, attr_name)[block_id]  # pylint: disable=unsubscriptable-object
            == single_qubit_stabs
        )

        # 3. Add new single-qubit stabilizers
        new_single_qubit_stabs = {
            Stabilizer(
                pauli="Z",
                data_qubits=((-1, -1),),
            ),
            Stabilizer(
                pauli="Z",
                data_qubits=((-2, -1),),
            ),
        }

        update_method(block_id, new_single_qubit_stabs)
        assert (
            getattr(step, attr_name)[block_id]  # pylint: disable=unsubscriptable-object
            == single_qubit_stabs | new_single_qubit_stabs
        )

        # 4. Try adding invalid single-qubit stabilizers
        invalid_single_qubit_stabs = {
            Stabilizer(
                pauli="XX",
                data_qubits=((-2, -2), (-3, -3)),
            ),
            Stabilizer(
                pauli="Z",
                data_qubits=((-1, -1),),
            ),
        }
        not_stab_set = {
            "not_a_stabilizer_object",
        }

        # 4.1. Not a stabilizer
        err_msg = (
            f"Invalid single-qubit stabilizer: '{next(iter(not_stab_set))}'. Must be of type "
            "`Stabilizer`"
        )
        with pytest.raises(TypeError) as cm:
            update_method(block_id, not_stab_set)
        assert err_msg in str(cm.value)

        # 4.2. Not a single-qubit stabilizer
        err_msg = "Each single-qubit stabilizer must contain exactly one data qubit."
        with pytest.raises(ValueError) as cm:
            update_method(block_id, invalid_single_qubit_stabs)
        assert err_msg in str(cm.value)

        # 5. Try adding to a block not present in block history
        # at the current timestep
        missing_block_id = uuid4()

        err_msg = f"Block {missing_block_id} not present at current timestep in block history."
        with pytest.raises(ValueError) as cm:
            update_method(missing_block_id, new_single_qubit_stabs)
        assert err_msg in str(cm.value)
