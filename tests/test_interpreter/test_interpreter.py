"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

import unittest
from copy import deepcopy

from loom.eka import (
    Lattice,
    Eka,
    Block,
    Circuit,
    ChannelType,
    Channel,
    SyndromeCircuit,
    Stabilizer,
    PauliOperator,
)
from loom.eka.operations import MeasureBlockSyndromes
from loom.interpreter import InterpretationStep, interpret_eka, cleanup_final_step


# pylint: disable=too-many-instance-attributes
class TestInterpreter(unittest.TestCase):
    """
    Tests the interpreter API, interpret_eka, and abstracted functions like
    cleanup_final_step.
    """

    def setUp(self):
        self.lattice = Lattice.square_2d((10, 20))
        # pylint: disable=duplicate-code
        # These Blocks only contain 4-body stabilizers and logical operators
        self.rep_code_1 = Block(
            stabilizers=(
                Stabilizer(
                    pauli="ZZ",
                    data_qubits=((0, 0, 0), (0, 1, 0)),
                    ancilla_qubits=((0, 1, 1),),
                ),
                Stabilizer(
                    pauli="ZZ",
                    data_qubits=((0, 1, 0), (0, 2, 0)),
                    ancilla_qubits=((0, 2, 1),),
                ),
            ),
            logical_x_operators=[
                PauliOperator(
                    pauli="XXX", data_qubits=((0, 0, 0), (0, 1, 0), (0, 2, 0))
                )
            ],
            logical_z_operators=[PauliOperator(pauli="Z", data_qubits=((0, 0, 0),))],
            unique_label="q1",
        )
        self.rep_code_2 = self.rep_code_1.shift(position=(4, 0), new_label="q2")
        self.rep_code_custom = Block(
            stabilizers=self.rep_code_2.stabilizers,
            logical_x_operators=self.rep_code_2.logical_x_operators,
            logical_z_operators=self.rep_code_2.logical_z_operators,
            unique_label="q3",
        )
        self.eka = Eka(
            self.lattice,
            blocks=[self.rep_code_1, self.rep_code_2],
        )

        #  Rotated Surface Code
        self.square_2d_lattice = Lattice.square_2d((10, 10))
        channels = {
            "a": [
                Channel(type=ChannelType.QUANTUM, label="a0"),
            ],
            "q": [Channel(type=ChannelType.QUANTUM, label=f"d{i}") for i in range(4)],
            "c": [Channel(type=ChannelType.CLASSICAL, label="c0")],
        }
        xxxx_circuit = SyndromeCircuit(
            name="xxxx",
            pauli="XXXX",
            circuit=Circuit(
                name="xxxx",
                circuit=(
                    (Circuit("Reset_0", channels=channels["a"]),),
                    (Circuit("H", channels=channels["a"]),),
                    (Circuit("CX", channels=[channels["a"][0], channels["q"][0]]),),
                    (Circuit("CX", channels=[channels["a"][0], channels["q"][1]]),),
                    (Circuit("CX", channels=[channels["a"][0], channels["q"][2]]),),
                    (Circuit("CX", channels=[channels["a"][0], channels["q"][3]]),),
                    (Circuit("H", channels=channels["a"]),),
                    (
                        Circuit(
                            "Measurement", channels=[channels["a"][0], channels["c"][0]]
                        ),
                    ),
                ),
                channels=channels["q"] + channels["a"] + channels["c"],
            ),
        )
        zzzz_circuit = SyndromeCircuit(
            name="zzzz",
            pauli="ZZZZ",
            circuit=Circuit(
                name="zzzz",
                circuit=(
                    (
                        Circuit(
                            "Reset_0",
                            channels=channels["a"],
                        ),
                    ),
                    (Circuit("H", channels=channels["a"]),),
                    (Circuit("CZ", channels=[channels["a"][0], channels["q"][0]]),),
                    (Circuit("CZ", channels=[channels["a"][0], channels["q"][1]]),),
                    (Circuit("CZ", channels=[channels["a"][0], channels["q"][2]]),),
                    (Circuit("CZ", channels=[channels["a"][0], channels["q"][3]]),),
                    (Circuit("H", channels=channels["a"]),),
                    (
                        Circuit(
                            "Measurement",
                            channels=[channels["a"][0], channels["c"][0]],
                        ),
                    ),
                ),
                channels=channels["q"] + channels["a"] + channels["c"],
            ),
        )
        left_xx_circuit = SyndromeCircuit(
            pauli="XX",
            name="left_xx",
            circuit=Circuit(
                name="left_xx",
                circuit=(
                    (
                        Circuit(
                            "Reset_0",
                            channels=channels["a"],
                        ),
                    ),
                    (Circuit("H", channels=channels["a"]),),
                    (Circuit("CX", channels=[channels["a"][0], channels["q"][0]]),),
                    (Circuit("CX", channels=[channels["a"][0], channels["q"][1]]),),
                    (),
                    (),
                    (Circuit("H", channels=channels["a"]),),
                    (
                        Circuit(
                            "Measurement",
                            channels=[channels["a"][0], channels["c"][0]],
                        ),
                    ),
                ),
                channels=channels["q"][:2] + channels["a"] + channels["c"],
            ),
        )
        right_xx_circuit = SyndromeCircuit(
            pauli="XX",
            name="right_xx",
            circuit=Circuit(
                name="right_xx",
                circuit=(
                    (
                        Circuit(
                            "Reset_0",
                            channels=channels["a"],
                        ),
                    ),
                    (Circuit("H", channels=channels["a"]),),
                    (),
                    (),
                    (Circuit("CX", channels=[channels["a"][0], channels["q"][0]]),),
                    (Circuit("CX", channels=[channels["a"][0], channels["q"][1]]),),
                    (Circuit("H", channels=channels["a"]),),
                    (
                        Circuit(
                            "Measurement",
                            channels=[channels["a"][0], channels["c"][0]],
                        ),
                    ),
                ),
                channels=channels["q"][:2] + channels["a"] + channels["c"],
            ),
        )
        top_zz_circuit = SyndromeCircuit(
            pauli="ZZ",
            name="top_zz",
            circuit=Circuit(
                name="top_zz",
                circuit=(
                    (
                        Circuit(
                            "Reset_0",
                            channels=channels["a"],
                        ),
                    ),
                    (Circuit("H", channels=channels["a"]),),
                    (),
                    (),
                    (Circuit("CZ", channels=[channels["a"][0], channels["q"][0]]),),
                    (Circuit("CZ", channels=[channels["a"][0], channels["q"][1]]),),
                    (Circuit("H", channels=channels["a"]),),
                    (
                        Circuit(
                            "Measurement",
                            channels=[channels["a"][0], channels["c"][0]],
                        ),
                    ),
                ),
                channels=channels["q"][:2] + channels["a"] + channels["c"],
            ),
        )
        bottom_zz_circuit = SyndromeCircuit(
            name="bottom_zz",
            pauli="ZZ",
            circuit=Circuit(
                name="bottom_zz",
                circuit=(
                    (
                        Circuit(
                            "Reset_0",
                            channels=channels["a"],
                        ),
                    ),
                    (Circuit("H", channels=channels["a"]),),
                    (Circuit("CZ", channels=[channels["a"][0], channels["q"][0]]),),
                    (Circuit("CZ", channels=[channels["a"][0], channels["q"][1]]),),
                    (),
                    (),
                    (Circuit("H", channels=channels["a"]),),
                    (
                        Circuit(
                            "Measurement",
                            channels=[channels["a"][0], channels["c"][0]],
                        ),
                    ),
                ),
                channels=channels["q"][:2] + channels["a"] + channels["c"],
            ),
        )
        # pylint: disable=duplicate-code
        rsc_stabilizers = (
            Stabilizer(
                "ZZZZ",
                ((1, 0, 0), (0, 0, 0), (1, 1, 0), (0, 1, 0)),
                ancilla_qubits=((1, 1, 1),),
            ),
            Stabilizer(
                "ZZZZ",
                ((2, 1, 0), (1, 1, 0), (2, 2, 0), (1, 2, 0)),
                ancilla_qubits=((2, 2, 1),),
            ),
            Stabilizer(
                "XXXX",
                ((1, 1, 0), (1, 2, 0), (0, 1, 0), (0, 2, 0)),
                ancilla_qubits=((1, 2, 1),),
            ),
            Stabilizer(
                "XXXX",
                ((2, 0, 0), (2, 1, 0), (1, 0, 0), (1, 1, 0)),
                ancilla_qubits=((2, 1, 1),),
            ),
            Stabilizer(
                "XX",
                ((0, 0, 0), (0, 1, 0)),
                ancilla_qubits=((0, 1, 1),),
            ),
            Stabilizer(
                "XX",
                ((2, 1, 0), (2, 2, 0)),
                ancilla_qubits=((3, 2, 1),),
            ),
            Stabilizer(
                "ZZ",
                ((2, 0, 0), (1, 0, 0)),
                ancilla_qubits=((2, 0, 1),),
            ),
            Stabilizer(
                "ZZ",
                ((1, 2, 0), (0, 2, 0)),
                ancilla_qubits=((1, 3, 1),),
            ),
        )
        self.rot_surf_code_1 = Block(
            unique_label="q1",
            stabilizers=rsc_stabilizers,
            logical_x_operators=(
                PauliOperator("XXX", ((0, 0, 0), (1, 0, 0), (2, 0, 0))),
            ),
            logical_z_operators=(
                PauliOperator("ZZZ", ((0, 0, 0), (0, 1, 0), (0, 2, 0))),
            ),
            syndrome_circuits=[
                left_xx_circuit,
                right_xx_circuit,
                top_zz_circuit,
                bottom_zz_circuit,
                zzzz_circuit,
                xxxx_circuit,
            ],
            stabilizer_to_circuit={
                rsc_stabilizers[0].uuid: zzzz_circuit.uuid,
                rsc_stabilizers[1].uuid: zzzz_circuit.uuid,
                rsc_stabilizers[2].uuid: xxxx_circuit.uuid,
                rsc_stabilizers[3].uuid: xxxx_circuit.uuid,
                rsc_stabilizers[4].uuid: left_xx_circuit.uuid,
                rsc_stabilizers[5].uuid: right_xx_circuit.uuid,
                rsc_stabilizers[6].uuid: top_zz_circuit.uuid,
                rsc_stabilizers[7].uuid: bottom_zz_circuit.uuid,
            },
        )
        # self.rot_surf_code_1 = RotatedSurfaceCode.create(
        #     dx=3, dz=3, lattice=self.square_2d_lattice, unique_label="q1"
        # )
        self.meas_block_op = MeasureBlockSyndromes(
            self.rot_surf_code_1.unique_label, n_cycles=3
        )
        self.eka_rsc = Eka(
            self.square_2d_lattice,
            blocks=[self.rot_surf_code_1],
            operations=[self.meas_block_op],
        )

    def test_run_interpreter_without_operations(self):
        """
        Tests where the interpreter runs without an error where the Eka does not
        include any operations yet.
        """
        final_step = interpret_eka(self.eka)
        self.assertTrue(isinstance(final_step, InterpretationStep))

    def test_run_interpreter_parallel_operations(self):
        """
        Tests the interpretation for multiple operations happening in parallel.
        """
        operations = [
            [MeasureBlockSyndromes("q1", 2), MeasureBlockSyndromes("q2", 2)],
        ]
        eka_w_ops = Eka(
            self.lattice,
            blocks=[self.rep_code_1, self.rep_code_2],
            operations=operations,
        )
        new_step = interpret_eka(eka_w_ops)
        circ_meas_block_1 = new_step.final_circuit.circuit[0][0]
        circ_meas_block_2 = new_step.final_circuit.circuit[0][1]
        # Check that the right circuits are generated
        self.assertEqual(circ_meas_block_1.name, "measure q1 syndromes 2 time(s)")
        self.assertEqual(circ_meas_block_2.name, "measure q2 syndromes 2 time(s)")
        # Check that both operations are indeed done in parallel
        self.assertEqual(len(new_step.final_circuit.circuit), 12)

    def test_run_interpreter_parallel_operations_different_lengths(self):
        """
        Tests the interpretation for multiple operations happening in parallel but
        they have different lengths.
        """
        operations = [
            [
                MeasureBlockSyndromes("q2", 2),
                MeasureBlockSyndromes("q1", 1),
            ],
        ]
        eka_w_ops = Eka(
            self.lattice,
            blocks=[self.rep_code_1, self.rep_code_2],
            operations=operations,
        )
        new_step = interpret_eka(eka_w_ops)
        circ_meas_block_2 = new_step.final_circuit.circuit[0][0]
        circ_meas_block_1 = new_step.final_circuit.circuit[0][1]
        # Check that the right circuits are generated
        self.assertEqual(circ_meas_block_2.name, "measure q2 syndromes 2 time(s)")
        self.assertEqual(circ_meas_block_1.name, "measure q1 syndromes 1 time(s)")
        # Check that both operations are indeed done in parallel
        self.assertEqual(len(new_step.final_circuit.circuit), 12)

    def test_change_final_step(self):
        """
        Tests that an exception is raised if methods of InterpretationStep are called
        after the interpreter has finished and which would change the final step.
        """
        operations = []
        eka = Eka(
            self.lattice,
            blocks=[self.rep_code_1, self.rep_code_custom],
            operations=operations,
        )
        final_step = interpret_eka(eka)

        err_msg = (
            "Cannot change properties of the final InterpretationStep after the "
            "interpretation is finished."
        )

        # Call those methods which mutate the InterpretationStep and check that an
        # exception is raised
        with self.assertRaises(ValueError) as cm:
            final_step.get_channel_MUT("DUMMY_CHANNEL_NAME")
        self.assertIn(err_msg, str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            circ = Circuit(name="test", channels=[])
            final_step.append_circuit_MUT(circ)
        self.assertIn(err_msg, str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            final_step.get_new_cbit_MUT("c")
        self.assertIn(err_msg, str(cm.exception))

    def test_cleanup_final_step(self):
        """
        Tests the clean up function.
        """
        # Check that channels that are not part of the final circuit are removed
        # from the channel_dict.
        operations = []
        eka = Eka(
            self.lattice,
            blocks=[self.rep_code_1, self.rep_code_custom],
            operations=operations,
        )
        final_step = interpret_eka(eka)
        original_channel_dict = deepcopy(final_step.channel_dict)
        final_step.channel_dict["DUMMY_CHANNEL_NAME"] = Channel(
            label="DUMMY_CHANNEL_NAME"
        )
        # After the previous line, the channel_dict should have changed
        self.assertNotEqual(original_channel_dict, final_step.channel_dict)
        # Check that the cleaned-up version corresponds to the original dict
        final_step = cleanup_final_step(final_step)
        self.assertEqual(original_channel_dict, final_step.channel_dict)

        # Check that all the channels in channel_dict also appear in the circuit
        if final_step.final_circuit is not None:
            for ch in final_step.channel_dict.values():
                self.assertIn(ch, final_step.final_circuit.channels)

    def test_cleanup_final_step_with_circuit(self):
        """
        Test that the cleanup function generates the right circuit from a given
        intermediate_circuit_sequence.
        """
        operations = []
        eka = Eka(
            self.lattice,
            blocks=[self.rep_code_1],
            operations=operations,
        )
        final_step = interpret_eka(eka)
        channels = [Channel(label=f"{i}") for i in range(4)]
        bell_pair = Circuit(
            name="bell",
            circuit=[
                [Circuit("H", channels=[channels[0]])],
                [Circuit("CX", channels=channels[:2], duration=2)],
            ],
        )
        single_x = Circuit(
            name="single_x",
            circuit=[
                [Circuit("X", channels=[channel]) for channel in channels[2:]],
            ],
        )
        another_bell_pair = bell_pair.clone(channels[2:])
        final_step.intermediate_circuit_sequence = (
            (bell_pair, single_x),
            (another_bell_pair,),
        )
        final_step = cleanup_final_step(final_step)
        expected_circuit = Circuit(
            name="Final circuit",
            circuit=(
                (bell_pair, single_x),
                tuple(),
                tuple(),
                (another_bell_pair,),
                tuple(),
                tuple(),
            ),
        )
        self.assertEqual(final_step.final_circuit.circuit, expected_circuit.circuit)

    def test_cleanup_final_step_sorted_channels(self):
        """Test that the channels after cleanup function are sorted."""
        # Channels get scrambled after MeasureBlockSyndromes operation

        eka = Eka(
            self.lattice,
            blocks=[self.rot_surf_code_1],
            operations=[MeasureBlockSyndromes(self.rot_surf_code_1.unique_label)],
        )
        final_step = interpret_eka(eka)

        # Sort channels in final circuit by channel labels
        sorted_channels = tuple(
            sorted(final_step.final_circuit.channels, key=lambda item: item.label)
        )

        # Ensure that sorted_channel_dict is equal to the channel_dict after cleanup
        self.assertEqual(final_step.final_circuit.channels, sorted_channels)


if __name__ == "__main__":
    unittest.main()
