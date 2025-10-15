"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

# pylint: disable=too-many-lines

import logging
import unittest
import re

from pydantic import ValidationError

from loom.eka import Circuit, Channel, ChannelType
from loom.eka.utilities import dumps, loads, findall


# pylint: disable=too-many-public-methods
class TestCircuit(unittest.TestCase):
    """
    Test for the Circuit class.
    """

    def setUp(self):
        self.channels = tuple(Channel() for _ in range(4))
        self.hadamard = Circuit("hadamard", channels=self.channels[0:1])
        self.cnot = Circuit("cnot", channels=self.channels[0:2], duration=5)
        self.entangle = Circuit("entangle", [self.hadamard, self.cnot])
        self.cnot2 = self.cnot.clone([self.channels[0], self.channels[2]])
        self.entangle2 = Circuit(
            "entangle2",
            [self.hadamard, self.cnot, self.cnot2],
        )

        # Set to ERROR to avoid cluttering the test output
        logging.getLogger().setLevel(logging.ERROR)
        self.formatter = lambda lvl, msg: f"{lvl}:loom.eka.circuit:{msg}"

    def test_logging__eq__(self):
        """
        Test that the logging in __eq__ method of Circuit works as expected.

        Note: There is no way to test logging formatting, so we only check that the
        logging is called
        """
        with self.assertLogs("loom.eka.circuit", level="DEBUG") as cm:
            self.assertNotEqual(self.hadamard, self.cnot)
        err_msg1 = "The two circuits have a different number of time slices."
        err_msg2 = f"{self.hadamard.duration} != {self.cnot.duration}\n"
        self.assertEqual(
            [self.formatter("INFO", err_msg1), self.formatter("DEBUG", err_msg2)],
            cm.output,
        )

        with self.assertLogs("loom.eka.circuit", level="DEBUG") as cm:
            self.assertNotEqual(
                Circuit("1", [[], [self.cnot]]),
                Circuit("2", [[self.hadamard], [self.cnot]]),
            )
        err_msg1 = "The two circuits have a different number of gates in a time slice."
        err_msg2 = "0 != 1 for time slices () and (hadamard (base gate),)\n"
        self.assertEqual(
            [self.formatter("INFO", err_msg1), self.formatter("DEBUG", err_msg2)],
            cm.output,
        )

        with self.assertLogs("loom.eka.circuit", level="DEBUG") as cm:
            self.assertNotEqual(
                Circuit("1", [self.hadamard]),
                Circuit("2", [Circuit("X", channels=Channel())]),
            )
        err_msg1 = "The two circuits have different gates in a time slice."
        err_msg2 = (
            "For time steps 0: (hadamard (base gate),) and " + "0: (x (base gate),), \n"
            "    hadamard != x for gates hadamard (base gate) and x (base gate)\n"
        )
        self.assertEqual(
            [self.formatter("INFO", err_msg1), self.formatter("DEBUG", err_msg2)],
            cm.output,
        )

        with self.assertLogs("loom.eka.circuit", level="DEBUG") as cm:
            self.assertNotEqual(
                Circuit("1", [Circuit("X", channels=[Channel(), Channel()])]),
                Circuit("2", [Circuit("X", channels=Channel())]),
            )
        err_msg1 = "The two circuits have different channels in a gate."
        err_msg2 = (
            "\n    [(<ChannelType.QUANTUM: 'quantum'>, 'data_qubit')]\n"
            "        !=\n"
            "    [(<ChannelType.QUANTUM: 'quantum'>, 'data_qubit'), "
            "(<ChannelType.QUANTUM: 'quantum'>, 'data_qubit')]\n"
        )
        self.assertEqual(
            [self.formatter("INFO", err_msg1), self.formatter("DEBUG", err_msg2)],
            cm.output,
        )

    def test_setup_basic(self):
        """
        Tests that the creation of a basic gate is done correctly:
        - test validation of the Circuit data class: non-empty channels and name
        - test correct inference in Circuit.create(): default and custom channels
        """
        with self.assertRaises(Exception) as cm:
            Circuit("", [Channel()])
        self.assertIn(
            "Names of Circuit objects need to have at least one letter.",
            str(cm.exception),
        )
        self.assertIn("Input should be a valid tuple", str(cm.exception))

        test_h = Circuit("hadamard")
        self.assertEqual(len(test_h.channels), 1)
        self.assertEqual(type(test_h.channels[0]), Channel)
        self.assertEqual(test_h.circuit, tuple())

        channel = Channel()
        testh_2 = Circuit("hadamard", channels=channel)
        self.assertEqual(testh_2.channels, (channel,))
        self.assertEqual(testh_2.circuit, tuple())

    def test_setup_composite_no_channels(self):
        """
        Tests that the channels are correctly inferred from the sub-circuit channels
        """
        test_circuit = Circuit("entangle", [self.hadamard, self.cnot])
        self.assertEqual(set(test_circuit.channels), set(self.channels[0:2]))

    def test_count_qubits(self):
        """
        Tests that the number of qubits in a circuit is correctly counted.
        """
        self.assertEqual(self.hadamard.nr_of_qubits_in_circuit(), 1)
        self.assertEqual(self.cnot.nr_of_qubits_in_circuit(), 2)
        self.assertEqual(self.entangle2.nr_of_qubits_in_circuit(), 3)

    def test_base_gate_cloning1(self):
        """
        Tests that the cloning of a base gate is done correctly:
        - the name is transferred,
        - the id is newly generated,
        - the inputs and outputs are given by the parameters in clone_circuit().
        """
        new_in = Channel()
        new_hadamard = self.hadamard.clone(new_in)
        self.assertNotEqual(new_hadamard.id, self.hadamard.id)
        self.assertNotEqual(new_hadamard.channels[0].id, self.hadamard.channels[0].id)
        self.assertEqual(new_hadamard.name, self.hadamard.name)
        self.assertEqual(new_hadamard.channels[0].id, new_in.id)
        self.assertEqual(new_hadamard.duration, self.hadamard.duration)

    def test_base_gate_cloning2(self):
        """
        Tests that the cloning of a base gate is done correctly:
        - the name is transferred,
        - the id is newly generated,
        - the input and output channel ids are newly generated.
        """
        new_hadamard = self.hadamard.clone()
        self.assertNotEqual(new_hadamard.id, self.hadamard.id)
        self.assertNotEqual(new_hadamard.channels[0].id, self.hadamard.channels[0].id)
        self.assertEqual(new_hadamard.name, self.hadamard.name)
        self.assertEqual(new_hadamard.duration, self.hadamard.duration)

    def test_nested_circuit_cloning(self):
        """
        Tests that the cloning of a nested gate is done correctly:
        - the name is transferred,
        - all circuit channel ids are newly generated
        - check that channel ids that appeared multiple times in the original circuit
            are also the same in the new circuit
        """
        new_entangle = self.entangle.clone()
        old_ent_ids = [channel.id for channel in self.entangle.channels]
        new_ent_ids = [channel.id for channel in new_entangle.channels]
        old_had_ids = [channel.id for channel in self.entangle.circuit[0][0].channels]
        new_had_ids = [channel.id for channel in new_entangle.circuit[0][0].channels]
        old_cnot_ids = [channel.id for channel in self.entangle.circuit[1][0].channels]
        new_cnot_ids = [channel.id for channel in new_entangle.circuit[1][0].channels]
        old_ids = old_ent_ids + old_had_ids + old_cnot_ids
        new_ids = new_ent_ids + new_had_ids + new_cnot_ids

        self.assertEqual(new_entangle.name, self.entangle.name)
        self.assertEqual(
            new_entangle.circuit[0][0].name, self.entangle.circuit[0][0].name
        )
        self.assertEqual(
            new_entangle.circuit[1][0].name, self.entangle.circuit[1][0].name
        )
        self.assertEqual(
            new_entangle.circuit[0][0].duration, self.entangle.circuit[0][0].duration
        )
        self.assertEqual(
            new_entangle.circuit[1][0].duration, self.entangle.circuit[1][0].duration
        )
        self.assertNotIn(
            False, [old != new for old, new in zip(old_ids, new_ids, strict=True)]
        )
        self.assertEqual(
            [findall(old_had_ids, id) for id in old_ent_ids],
            [findall(new_had_ids, id) for id in new_ent_ids],
        )
        self.assertEqual(
            [findall(old_ent_ids, id) for id in old_cnot_ids],
            [findall(new_ent_ids, id) for id in new_cnot_ids],
        )
        self.assertEqual(
            [findall(old_cnot_ids, id) for id in old_had_ids],
            [findall(new_cnot_ids, id) for id in new_had_ids],
        )
        self.assertEqual(new_entangle.duration, self.entangle.duration)

    def test_load_dump(self):
        """
        Test that the load and dump functions work correctly.
        """
        test_circuit_j = dumps(self.entangle2)
        loaded_test_circuit = loads(Circuit, test_circuit_j)

        self.assertEqual(loaded_test_circuit, self.entangle2)

    def test_setup_composite_correct_channels(self):
        """
        Tests that consistency is confirmed for the correct channels and given order
        of channels is maintained.
        """
        channels = tuple(Channel() for _ in range(2))
        hadamard = Circuit("hadamard", channels=(channels[1],))
        cnot = Circuit("cnot", channels=channels[::-1])

        test_circuit = Circuit("entangle", [hadamard, cnot], channels)
        # Check that the channels are consistent with the given channels
        self.assertEqual(set(test_circuit.channels), set(channels))
        # Check that the sub-circuits still act on the right channels
        self.assertEqual(test_circuit.circuit[0][0].channels, (channels[1],))
        self.assertEqual(test_circuit.circuit[1][0].channels, channels[::-1])

    def test_setup_composite_order_channels(self):
        """
        Tests that the correct channels are inferred from the sub-circuit channels in
        the right order and that the idle channels are added.
        """
        channels = tuple(Channel() for _ in range(2))
        hadamard = Circuit("hadamard", channels=channels[1])
        cnot = Circuit("cnot", channels=channels[::-1])

        entangle = Circuit("entangle", [hadamard, cnot])
        # Check that the channels are consistent with the given channels
        self.assertEqual(set(entangle.channels), set(channels))

        meas = Circuit(
            "Measurement",
            channels=[
                Channel(),
                Channel(ChannelType.QUANTUM),
                Channel(ChannelType.CLASSICAL),
            ],
        )
        ancilla = Channel(ChannelType.QUANTUM)
        cchannels = tuple(Channel(ChannelType.CLASSICAL) for _ in range(2))
        circuit = Circuit(
            "complex_test",
            [
                entangle,
                meas.clone([channels[0], ancilla, cchannels[0]]),
                meas.clone([channels[1], ancilla, cchannels[1]]),
            ],
        )
        # Check that the channels are consistent with the given channels
        self.assertEqual(set(circuit.channels), set(channels + (ancilla,) + cchannels))
        # Check that the sub-circuits still act on the right channels
        self.assertEqual(circuit.circuit[0][0].channels, entangle.channels)
        self.assertEqual(circuit.circuit[1], ())  # Empty timestep
        self.assertEqual(
            circuit.circuit[2][0].channels, (channels[0], ancilla, cchannels[0])
        )
        self.assertEqual(
            circuit.circuit[3][0].channels, (channels[1], ancilla, cchannels[1])
        )

        self.assertTrue(isinstance(circuit.channels, tuple))

    def test_setup_composite_empty_timesteps(self):
        """
        Tests that empty time steps are resolved correctly and that duration is
        correctly inferred.
        """
        channels = tuple(Channel() for _ in range(3))
        # The circuit is created with extra empty time steps
        circuit_w_wait1 = Circuit(
            "circuit_w_wait",
            [
                [Circuit("cnot", channels=[channels[0], channels[1]], duration=5)],
                [Circuit("hadamard", channels=[channels[2]])],
                [],
                [],
                [Circuit("hadamard", channels=[channels[2]])],
                [],
            ],
        )
        self.assertEqual(circuit_w_wait1.duration, 6)
        # The circuit is created with empty time steps but also with a duration longer
        # than the existing time_steps (they are not all included)
        circuit_w_wait2 = Circuit(
            "circuit_w_wait",
            [
                [Circuit("hadamard", channels=[channels[2]])],
                [Circuit("cnot", channels=[channels[0], channels[1]], duration=5)],
                [Circuit("hadamard", channels=[channels[2]])],
                [],
            ],
        )
        self.assertEqual(circuit_w_wait2.duration, 6)

    def test_duration(self):
        """
        Tests that the duration of a circuit is correctly calculated.
        """
        self.assertEqual(self.hadamard.duration, 1)
        self.assertEqual(self.cnot.duration, 5)
        self.assertEqual(self.entangle2.duration, 11)
        test_circuit = Circuit(
            "test",
            [
                [self.hadamard.clone(self.channels[1]), self.cnot2],
                [],
                [],
                [self.cnot.clone([self.channels[1], self.channels[3]])],
            ],
        )
        self.assertEqual(test_circuit.duration, 8)
        with self.assertRaises(ValidationError) as cm:
            _ = Circuit(
                "fail", [[self.cnot], [self.cnot.clone(self.channels[2:4])]], duration=8
            )
        self.assertIn(
            "Error while setting up composite circuit: Provided duration (8)"
            + " does not match the duration of the sub-circuits (6).",
            str(cm.exception),
        )

    def test_op_timing(self):
        """
        Tests validation of timing of operations. A qubit can only be acted on by one
        gate each tick.
        """
        with self.assertRaises(ValidationError) as cm:
            _ = Circuit(
                "test",
                [
                    [self.hadamard.clone(self.channels[1]), self.cnot2],
                    [],
                    [],
                    [self.cnot.clone([self.channels[1], self.channels[3]])],
                    [self.hadamard.clone(self.channels[1])],
                ],
            )
        # Replace the random 6-digit uuid hex string to be able to use the assert statement
        cleaned_message = re.sub(
            r"data_qubit\([a-z0-9]{6}\.\.\)", "data_qubit(uuid)", str(cm.exception)
        )
        self.assertIn(
            "Error while setting up composite circuit: Channel data_qubit(uuid) is subject"
            + " to more than one operation at tick 4.",
            cleaned_message,
        )

    def test_as_gate(self):
        """
        Tests gate construction with the as_gate() convenience function
        """
        test_gate = Circuit.as_gate(
            "test_gate", nr_qchannels=3, nr_cchannels=2, duration=5
        )

        self.assertEqual(test_gate.name, "test_gate")
        self.assertEqual(len(test_gate.channels), 5)
        self.assertEqual(test_gate.duration, 5)
        self.assertEqual(test_gate.channels[0].type, ChannelType.QUANTUM)
        self.assertEqual(test_gate.channels[0].label, "data_qubit")
        self.assertEqual(test_gate.channels[1].type, ChannelType.QUANTUM)
        self.assertEqual(test_gate.channels[2].type, ChannelType.QUANTUM)
        self.assertEqual(test_gate.channels[3].type, ChannelType.CLASSICAL)
        self.assertEqual(test_gate.channels[4].type, ChannelType.CLASSICAL)

        test_gate = Circuit.as_gate("default_gate", 1)

        self.assertEqual(test_gate.name, "default_gate")
        self.assertEqual(len(test_gate.channels), 1)
        self.assertEqual(test_gate.duration, 1)
        self.assertEqual(test_gate.channels[0].type, ChannelType.QUANTUM)

    def test_from_circuit(self):
        """
        Tests circuit construction via the from_circuit() convenience function
        """
        test_circuit = Circuit.from_circuits(
            "test_circuit", [(self.hadamard, [0]), (self.cnot, [0, 1])]
        )

        self.assertEqual(test_circuit.name, "test_circuit")
        self.assertEqual(len(test_circuit.channels), 2)
        self.assertEqual(test_circuit.duration, 6)
        self.assertEqual(test_circuit.channels[0].type, ChannelType.QUANTUM)
        self.assertEqual(test_circuit.channels[1].type, ChannelType.QUANTUM)
        self.assertEqual(
            test_circuit.circuit[0][0].channels[0],
            test_circuit.circuit[1][0].channels[0],
        )
        self.assertNotEqual(
            test_circuit.circuit[0][0].channels[0],
            test_circuit.circuit[1][0].channels[1],
        )
        with self.assertRaises(ValueError) as cm:
            test_circuit = Circuit.from_circuits("test_circuit")
        self.assertIn(
            "Error while creating circuit via from_circuit(): "
            + "The circuit must be a list of circuits. If the intention is to copy "
            + "a circuit to deal with Channel objects directly, use the clone() "
            + "method instead.",
            str(cm.exception),
        )

        # check Channel type consistency
        zmeasure = Circuit.as_gate(name="measure_z", nr_qchannels=1, nr_cchannels=1)
        _ = Circuit.from_circuits(
            "measure_xx",
            [
                (self.hadamard, [0]),
                (self.cnot, [0, 2]),
                (self.cnot, [0, 3]),
                (self.hadamard, [0]),
                (zmeasure, [0, 1]),
            ],
        )
        with self.assertRaises(ValueError) as cm:
            _ = Circuit.from_circuits(
                "measure_xx",
                [
                    (self.hadamard, [0]),
                    (self.cnot, [0, 2]),
                    (self.cnot, [0, 3]),
                    (self.hadamard, [0]),
                    (zmeasure, [0, 2]),
                ],
            )
        self.assertIn(
            "Provided channel indices are not consistent with respect to their types. "
            + f"Offending channel 2 has type {ChannelType.CLASSICAL} but has previously been "
            f"used with a channel of type {ChannelType.QUANTUM}.",
            str(cm.exception),
        )

    def test_flatten_circuit_is_flat(self):
        """
        Tests whether the flattening function returns a flat circuit,
        i.e. whether all subcircuits have no further subcircuits.
        """

        def is_flat(circuit: Circuit):
            for subcirc in circuit.circuit:
                if len(subcirc) > 0 and len(subcirc[0].circuit) > 0:
                    return False  # Subcircuit contains more subcircuits
            return True

        full_circ = Circuit("full_circ", [self.entangle, self.hadamard])
        self.assertEqual(False, is_flat(full_circ))
        self.assertEqual(True, is_flat(full_circ.flatten()))

    def test_flatten_several_levels_nesting(self):
        """
        Tests the flattening function for a circuit with more levels of nesting.
        """
        data_qbs = [Channel(label=f"D{i}") for i in range(3)]
        data_cregs = [
            Channel(type=ChannelType.CLASSICAL, label=f"creg_D{i+1}") for i in range(3)
        ]
        data_inits = [Circuit("Reset", channels=[q]) for q in data_qbs]
        data_init_circ = Circuit(
            "Initialization", channels=data_qbs, circuit=data_inits
        )

        hadamards = Circuit(
            "Block1",
            circuit=(
                (Circuit("H", channels=[data_qbs[0]])),
                (Circuit("H", channels=[data_qbs[1]])),
            ),
        )
        cnots = Circuit(
            "Block2",
            circuit=(
                (Circuit("CNOT", channels=[data_qbs[0], data_qbs[1]])),
                (Circuit("CNOT", channels=[data_qbs[1], data_qbs[2]])),
            ),
        )
        blocks_circ = Circuit("Combined blocks", circuit=(hadamards, cnots))
        data_measurements = [
            Circuit("Measurement", channels=[q, creg])
            for q, creg in zip(data_qbs, data_cregs, strict=True)
        ]
        data_meas_circ = Circuit(
            "Final data qubit readout",
            channels=data_qbs + data_cregs,
            circuit=data_measurements,
        )
        full_circ = Circuit(
            "Full circuit", circuit=(data_init_circ, blocks_circ, data_meas_circ)
        )

        # This is the flattened circuit we expect
        flat_circuit_expected = Circuit(
            "Full circuit",
            circuit=(
                (Circuit("Reset", channels=[data_qbs[0]])),
                (Circuit("Reset", channels=[data_qbs[1]])),
                (Circuit("Reset", channels=[data_qbs[2]])),
                (Circuit("H", channels=[data_qbs[0]])),
                (Circuit("H", channels=[data_qbs[1]])),
                (Circuit("CNOT", channels=[data_qbs[0], data_qbs[1]])),
                (Circuit("CNOT", channels=[data_qbs[1], data_qbs[2]])),
                (Circuit("Measurement", channels=[data_qbs[0], data_cregs[0]])),
                (Circuit("Measurement", channels=[data_qbs[1], data_cregs[1]])),
                (Circuit("Measurement", channels=[data_qbs[2], data_cregs[2]])),
            ),
        )

        for tick_flattened, tick_exp in zip(
            full_circ.flatten().circuit, flat_circuit_expected.circuit, strict=True
        ):
            self.assertEqual(tick_flattened[0].name, tick_exp[0].name)
            self.assertEqual(tick_flattened[0].channels, tick_exp[0].channels)

    def test_flatten_multiple_circuits_per_tick(self):
        """
        Tests the flattening function for a circuit with multiple circuits per tick.
        """
        qreg = [Channel(label=f"q{i}") for i in range(3)]
        tick1 = [
            Circuit("a", channels=qreg[2]),
            Circuit("b", channels=qreg[1]),
            Circuit("c", channels=qreg[0]),
        ]
        tick2 = [
            Circuit("A", channels=qreg[0]),
            Circuit("B", channels=qreg[1]),
            Circuit("C", channels=qreg[2]),
        ]
        tick3 = [Circuit("d", channels=qreg[2], duration=3)]
        new_circ = Circuit(name="circ", circuit=[tick1, tick2, tick3])
        new_circ = new_circ.flatten()
        flat_circuit = Circuit(name="flat_circuit", circuit=tick1 + tick2 + tick3)
        self.assertEqual(new_circ.circuit, flat_circuit.circuit)

    def test_unroll_circuit(self):
        """
        Tests the unroll method for a circuit defined recursively.
        """
        qreg = [Channel(label=f"q{i}") for i in range(4)]
        circ_1a = Circuit(
            "circ_1a",
            circuit=[
                [
                    Circuit("a0", channels=qreg[0]),
                    Circuit("a1", channels=qreg[1]),
                ],
                [
                    Circuit("a0", channels=qreg[0]),
                    Circuit("a1", channels=qreg[1]),
                ],
            ],
        )
        circ_1b = Circuit(
            "circ_1b",
            circuit=[
                [
                    Circuit("b2", channels=qreg[2]),
                    Circuit("b3", channels=qreg[3]),
                ],
                [
                    Circuit("b2", channels=qreg[2]),
                    Circuit("b3", channels=qreg[3]),
                ],
            ],
        )
        circ_2a = Circuit(
            "circ_2a",
            circuit=[
                [
                    Circuit("aa0", channels=qreg[0]),
                    Circuit("aa1", channels=qreg[1]),
                    Circuit("aa2", channels=qreg[2]),
                    Circuit("aa3", channels=qreg[3]),
                ],
                [
                    Circuit("aa0", channels=qreg[0]),
                    Circuit("aa1", channels=qreg[1]),
                ],
            ],
        )
        circ_3a = Circuit("circ_3a", circuit=(), channels=qreg)
        composite_circuit = Circuit(
            "composite_circuit",
            circuit=[
                [circ_1a, circ_1b],
                [],
                [circ_2a],
                [],
                [circ_3a],
            ],
        )

        unrolled_circ = Circuit.unroll(composite_circuit)
        expected_circ = Circuit(
            "expected_circuit",
            circuit=[
                [
                    Circuit("a0", channels=qreg[0]),
                    Circuit("a1", channels=qreg[1]),
                    Circuit("b2", channels=qreg[2]),
                    Circuit("b3", channels=qreg[3]),
                ],
                [
                    Circuit("a0", channels=qreg[0]),
                    Circuit("a1", channels=qreg[1]),
                    Circuit("b2", channels=qreg[2]),
                    Circuit("b3", channels=qreg[3]),
                ],
                [
                    Circuit("aa0", channels=qreg[0]),
                    Circuit("aa1", channels=qreg[1]),
                    Circuit("aa2", channels=qreg[2]),
                    Circuit("aa3", channels=qreg[3]),
                ],
                [
                    Circuit("aa0", channels=qreg[0]),
                    Circuit("aa1", channels=qreg[1]),
                ],
                [
                    Circuit("circ_3a", channels=qreg),
                ],
            ],
        )
        self.assertEqual(unrolled_circ, expected_circ.circuit)

    def test_unroll_circuit_with_empty_time_steps(self):
        """
        Test the unroll method for a circuit with empty time steps.
        """
        qreg = [Channel(label=f"q{i}") for i in range(4)]
        circ1 = Circuit(
            "circ1",
            circuit=[
                [Circuit("x", channels=qreg[0]), Circuit("y", channels=qreg[1])],
                [Circuit("z", channels=qreg[2])],
            ],
        )
        circ2 = Circuit(
            "circ2",
            circuit=[
                [Circuit("phase", channels=qreg[3])],
                [],
                [],
                [Circuit("phase", channels=qreg[3])],
            ],
        )
        circ_w_wait = Circuit(
            "circ_w_wait",
            circuit=[
                [circ1, circ2],
                [],
                [],
                [Circuit("zzz", channels=qreg[2])],
                [],
                [],
            ],
        )
        unrolled_circ = Circuit(
            "unrolled_circuit",
            circuit=Circuit.unroll(circ_w_wait),
        )
        expected_circ = Circuit(
            "expected_circuit",
            circuit=[
                [
                    Circuit("x", channels=qreg[0]),
                    Circuit("y", channels=qreg[1]),
                    Circuit("phase", channels=qreg[3]),
                ],
                [Circuit("z", channels=qreg[2])],
                [],
                [Circuit("zzz", channels=qreg[2]), Circuit("phase", channels=qreg[3])],
                [],
                [],
            ],
        )
        self.assertEqual(expected_circ, unrolled_circ)

        # Check that unrolling a circuit twice gives the same result
        self.assertEqual(Circuit.unroll(unrolled_circ), unrolled_circ.circuit)

    def test_circuit_equivalence(self):
        """
        Check the equivalence of two Circuit objects.
        """
        qb_set1 = [Channel(label=f"D{i}") for i in range(3)]
        qb_set2 = [Channel(label=f"D{i}") for i in range(3)]

        c1 = Circuit("CNOT", channels=[qb_set1[0], qb_set1[1]])
        c2 = Circuit("CNOT", channels=[qb_set2[0], qb_set2[1]])
        c3 = Circuit("CZ", channels=[qb_set2[0], qb_set2[1]])
        # c1 and c2 contain the same gate (CNOT) on the same qubits, although they
        # are different objects (with different uuids)
        self.assertEqual(c1, c2)
        self.assertNotEqual(c2, c3)  # Different gate (CNOT vs CZ)

        two_cnots_1 = Circuit(
            name="2CNOTS",
            circuit=(
                (Circuit("CNOT", channels=[qb_set1[0], qb_set1[1]])),
                (Circuit("CNOT", channels=[qb_set1[1], qb_set1[2]])),
            ),
        )
        two_cnots_2 = Circuit(
            name="2CNOTS",
            circuit=(
                (Circuit("CNOT", channels=[qb_set2[0], qb_set2[1]])),
                (Circuit("CNOT", channels=[qb_set2[1], qb_set2[2]])),
            ),
        )
        two_cnots_3 = Circuit(
            name="2CNOTS",
            circuit=(
                (Circuit("CNOT", channels=[qb_set2[0], qb_set2[2]])),
                (Circuit("CNOT", channels=[qb_set2[1], qb_set2[2]])),
            ),
        )
        # Same gates on the same qubits:
        self.assertEqual(two_cnots_1, two_cnots_2)
        # Different qubits involved: CNOT(0,1), CNOT(1,2) vs CNOT(0,2), CNOT(1,2)
        self.assertNotEqual(two_cnots_1, two_cnots_3)
        # Test a few more edge cases:
        # The following two circuits are equivalent although there are a few differences
        # - the two circuits are nested in different ways
        # - the two circuits have different names
        # - subcircuits which are not physical gates have different names
        # - the two circuits use the same quantum channels but in a permuted order
        #   (qubit 0 -> qubit 2, qubit 1 -> qubit 0, qubit 2 -> qubit 1)
        nested1 = Circuit(
            name="nested_circ",
            circuit=(
                (
                    Circuit(
                        name="block1",
                        circuit=(
                            (Circuit("H", channels=[qb_set1[0]])),
                            (Circuit("CNOT", channels=[qb_set1[0], qb_set1[1]])),
                            (Circuit("CNOT", channels=[qb_set1[1], qb_set1[2]])),
                        ),
                    )
                ),
                (
                    Circuit(
                        name="block2",
                        circuit=(
                            (Circuit("Z", channels=[qb_set1[1]])),
                            (Circuit("CY", channels=[qb_set1[0], qb_set1[2]])),
                        ),
                    )
                ),
            ),
        )
        nested2 = Circuit(
            name="differently_nested_circ",
            circuit=(
                (
                    Circuit(
                        name="DifferentNameButThisIsIgnored",
                        circuit=((Circuit("H", channels=[qb_set1[2]])),),
                    )
                ),
                (
                    Circuit(
                        name="DifferentNameButThisIsIgnored",
                        circuit=(
                            (
                                Circuit(
                                    name="EvenMoreNesting",
                                    circuit=(
                                        (
                                            Circuit(
                                                "CNOT",
                                                channels=[qb_set1[2], qb_set1[0]],
                                            )
                                        ),
                                        (
                                            Circuit(
                                                "CNOT",
                                                channels=[qb_set1[0], qb_set1[1]],
                                            )
                                        ),
                                    ),
                                )
                            ),
                            (Circuit("Z", channels=[qb_set1[0]])),
                        ),
                    )
                ),
                (Circuit("CY", channels=[qb_set1[2], qb_set1[1]])),
            ),
        )
        self.assertEqual(nested1, nested2)
        # Test eq method for cloned circuits
        nested2_clone = nested2.clone()
        self.assertEqual(nested2_clone, nested2)

    def test_eq_method_for_multiple_ticks(self):
        """
        Tests the __eq__() function for a circuit with multiple circuits per tick.
        """
        qreg = [Channel(label=f"q{i}") for i in range(3)]
        tick1 = [
            Circuit("a", channels=qreg[2]),
            Circuit("b", channels=qreg[1]),
            Circuit("c", channels=qreg[0]),
        ]
        tick2 = [
            Circuit("a2", channels=qreg[0]),
            Circuit("b2", channels=qreg[1]),
            Circuit("c2", channels=qreg[2]),
        ]
        tick3 = [Circuit("d", channels=qreg[2], duration=3)]
        circ_w_ticks = Circuit(name="circ_w_ticks", circuit=[tick1, tick2, tick3])
        circ_wo_ticks = Circuit(name="circ_wo_ticks", circuit=tick1 + tick2 + tick3)
        # The two circuit are not equal because of the tick structure
        self.assertNotEqual(circ_w_ticks, circ_wo_ticks)
        # The two circuits are equal when flattened
        self.assertEqual(circ_w_ticks.flatten(), circ_wo_ticks)

    def test_equality_w_time_structure(self):
        """
        Tests the __eq__() function for circuits with different_time_structure.
        """
        # Case 1 - two circuits with different time structure
        qubits = [Channel(label=f"q{i}") for i in range(3)]
        circ_1a = Circuit(
            name="circ_1a",
            circuit=[
                [Circuit("H", channels=[qubits[0]])],
                [Circuit("H", channels=[qubits[1]])],
            ],
        )
        circ_1b = Circuit(
            name="circ_1b",
            circuit=[
                [
                    Circuit("H", channels=[qubits[0]]),
                    Circuit("H", channels=[qubits[1]]),
                ],
            ],
        )
        self.assertNotEqual(circ_1a, circ_1b)
        # Case 2 - example with multiple time-steps and change in ordering of the channels
        # Note we use the cnot gate to enforce ordering of the qubits
        circ_2a = Circuit(
            name="circ_2a",
            circuit=[
                [
                    Circuit("H", channels=[qubits[0]]),
                ],
                [
                    Circuit("X", channels=[qubits[1]]),
                ],
                [
                    Circuit("CNOT", channels=[qubits[0], qubits[1]]),
                ],
            ],
        )
        # This circuit is equivalent to circ_2a but with other Channel objects
        circ_2a_copy = Circuit(
            name="circ_2a",
            circuit=[
                [
                    Circuit("H", channels=[qubits[1]]),
                ],
                [
                    Circuit("X", channels=[qubits[0]]),
                ],
                [
                    Circuit("CNOT", channels=[qubits[1], qubits[0]]),
                ],
            ],
        )
        # This circuit is different from circ_2a, because the CNOT acts on 0 then 1
        circ_2b = Circuit(
            name="circ_2b",
            circuit=[
                [
                    Circuit("H", channels=[qubits[0]]),
                ],
                [
                    Circuit("X", channels=[qubits[1]]),
                ],
                [
                    Circuit("CNOT", channels=[qubits[1], qubits[0]]),
                ],
            ],
        )
        circ_2c = Circuit(
            name="circ_2c",
            circuit=[
                [
                    Circuit("H", channels=[qubits[0]]),
                    Circuit("X", channels=[qubits[1]]),
                ],
                [
                    Circuit("CNOT", channels=[qubits[0], qubits[1]]),
                ],
            ],
        )
        self.assertEqual(circ_2a, circ_2a_copy)
        self.assertNotEqual(circ_2a, circ_2b)
        self.assertNotEqual(circ_2b, circ_2c)
        self.assertNotEqual(circ_2a, circ_2c)

    def test_equality_empty_time_steps(self):
        """
        Tests the __eq__() function for circuits with or without empty time steps.
        """
        qubits = [Channel(label="q")]
        circ_1 = Circuit(
            name="circ_1",
            circuit=[
                [Circuit("H", channels=qubits)],
                [],
            ],
        )
        circ_2 = Circuit(
            name="circ_2",
            circuit=[
                [Circuit("H", channels=qubits)],
            ],
        )
        self.assertNotEqual(circ_1, circ_2)

    def test_equality_permuted_steps(self):
        """
        Tests the __eq__() function for circuits with permuted steps.
        """
        qubits = [Channel(label=f"q{i}") for i in range(2)]
        circ_1 = Circuit(
            name="circ_1",
            circuit=[
                [Circuit("H", channels=[qubits[0]])],
                [Circuit("X", channels=[qubits[1]])],
            ],
        )
        circ_2 = Circuit(
            name="circ_2",
            circuit=[
                [Circuit("X", channels=[qubits[1]])],
                [Circuit("H", channels=[qubits[0]])],
            ],
        )
        self.assertNotEqual(circ_1, circ_2)

    def test_construct_padded_circuit_time_sequence(self):
        """
        Tests that the function construct_padded_circuit_time_sequence constructs
        a padded circuit time sequence correctly.
        """
        channels = tuple(Channel(label=f"q{i}") for i in range(3))
        hadamard = Circuit("h", duration=1, channels=[channels[0]])
        long_hadamard_1 = Circuit("h", duration=3, channels=[channels[1]])
        long_hadamard_2 = Circuit("h", duration=3, channels=[channels[2]])
        cnot = Circuit("cnot", duration=2, channels=channels[0:2])
        toffoli = Circuit("toffoli", duration=4, channels=channels)

        initial_and_padded_sequences = [
            (  # Time sequence with a single sub-circuit and no duration
                ((hadamard,),),  # initial
                ((hadamard,),),  # padded sequence is the same
            ),
            (  # Time sequence with a single sub-circuit and a duration
                ((long_hadamard_1,),),  # initial
                ((long_hadamard_1,), (), ()),  # padded sequence
            ),
            (  # Time sequence where the gates can be executed in parallel
                # but exist in different timesteps
                ((long_hadamard_1,), (long_hadamard_2,)),  # initial
                ((long_hadamard_1,), (long_hadamard_2,), (), ()),  # padded sequence
            ),
            (  # Time sequence with multiple sub-circuits of different duration
                # on different channels
                (
                    (
                        long_hadamard_2,
                        hadamard,
                    ),
                ),  # initial
                (
                    (
                        long_hadamard_2,
                        hadamard,
                    ),
                    (),
                    (),
                ),  # padded sequence
            ),
            (  # Time sequence with multiple sub-circuits on a single channel
                ((hadamard,), (cnot,)),  # initial
                ((hadamard,), (cnot,), ()),  # padded sequence
            ),
            (  # Time sequence with multiple sub-circuits on a single channel (different order)
                ((cnot,), (hadamard,)),  # initial
                ((cnot,), (), (hadamard,)),  # padded sequence
            ),
            (  # Time sequence with multiple time steps and sub-circuits
                # of different duration on different channels
                (
                    (
                        long_hadamard_2,
                        hadamard,
                    ),
                    (cnot,),
                    (toffoli,),
                ),  # initial
                (
                    (
                        long_hadamard_2,
                        hadamard,
                    ),
                    (cnot,),
                    (),
                    (toffoli,),
                    (),
                    (),
                    (),
                ),  # padded sequence
            ),
            (  # Time sequence with multiple time steps and sub-circuits of
                # different duration on different channels
                (
                    (
                        long_hadamard_1,
                        hadamard,
                    ),
                    (cnot,),
                    (toffoli,),
                ),  # initial
                (
                    (
                        long_hadamard_1,
                        hadamard,
                    ),
                    (),
                    (),
                    (cnot,),
                    (),
                    (toffoli,),
                    (),
                    (),
                    (),
                ),  # padded sequence
            ),
            (  # Time sequence with extra empty time steps that are not modified
                ((hadamard,), (), (cnot,)),  # initial with extra padding
                ((hadamard,), (), (cnot,), ()),  # padded sequence
            ),
            (  # Time sequence with extra empty time steps that are part of the expected padding
                ((long_hadamard_1,), (), (cnot,)),  # initial with extra padding
                ((long_hadamard_1,), (), (), (cnot,), ()),  # padded sequence
            ),
        ]
        for initial_sequence, expected_padded_sequence in initial_and_padded_sequences:
            padded_circ_timeseq = Circuit.construct_padded_circuit_time_sequence(
                initial_sequence
            )
            self.assertEqual(padded_circ_timeseq, expected_padded_sequence)


if __name__ == "__main__":
    unittest.main()
