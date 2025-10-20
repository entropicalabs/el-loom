"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

import unittest

from loom.eka import Channel, Circuit, ChannelType, Stabilizer, PauliOperator, Block
from loom.validator import is_syndrome_extraction_circuit_valid


class TestSECValidatorWorkflows(unittest.TestCase):
    """
    Test cases for validating syndrome extraction circuits using the Validator module.
    """

    def setUp(self) -> None:
        self.rep_code = lambda d: Block(
            unique_label="q1",
            stabilizers=tuple(
                Stabilizer(
                    pauli="ZZ",
                    data_qubits=(
                        (i, 0),
                        (i + 1, 0),
                    ),
                    ancilla_qubits=((i, 1),),
                )
                for i in range(d - 1)
            ),
            logical_x_operators=(PauliOperator("Z", ((0, 0),)),),
            logical_z_operators=(
                PauliOperator("X" * d, tuple((i, 0) for i in range(d))),
            ),
        )

    # pylint: disable=too-many-locals
    def test_reichard_circuit(self):
        """Test that Reichardt circuit for 7-qubit code passes the tests.
        Reference: https://arxiv.org/abs/1804.06995
        """
        # For demonstration purposes we use repeating tuple rather than a simple int
        qubits = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)]

        # Define stabilizers
        stabs_dict: dict[str, list[Stabilizer]] = {
            "X": [],
            "Z": [],
        }
        # First X, Z stabilizers
        stab_qubs = [(0, 0), (2, 2), (4, 4), (6, 6)]
        # make the stabilizers
        stabs_dict["X"] += [Stabilizer("XXXX", stab_qubs)]
        stabs_dict["Z"] += [Stabilizer("ZZZZ", stab_qubs)]

        # Second X, Z stabilizers
        stab_qubs = [(1, 1), (2, 2), (5, 5), (6, 6)]
        # make the stabilizers
        stabs_dict["X"] += [Stabilizer("XXXX", stab_qubs)]
        stabs_dict["Z"] += [Stabilizer("ZZZZ", stab_qubs)]

        # Third X, Z stabilizers
        stab_qubs = [(3, 3), (4, 4), (5, 5), (6, 6)]
        # make the stabilizers
        stabs_dict["X"] += [Stabilizer("XXXX", stab_qubs)]
        stabs_dict["Z"] += [Stabilizer("ZZZZ", stab_qubs)]

        # Block
        block = Block(
            stabilizers=[stab for stabs in stabs_dict.values() for stab in stabs],
            logical_x_operators=[PauliOperator("X" * 7, qubits)],
            logical_z_operators=[PauliOperator("Z" * 7, qubits)],
        )

        # Define data qubits channels and necessary mappings
        dq_channels = {qub: Channel("quantum", str(qub)) for qub in qubits}

        # Define ancilla qubits
        n_ancillas = 3
        aqubits = [
            Channel(type=ChannelType.QUANTUM, label=f"auxqubit_{i}")
            for i in range(n_ancillas)
        ]

        # Define classical channels for ancillas
        ac_channels = [
            Channel(type=ChannelType.CLASSICAL, label=f"c_{aux_q.label}")
            for aux_q in aqubits
        ]

        # X0X2X4X6
        had00 = Circuit("H", channels=[aqubits[0]])
        cnot00 = Circuit(
            "CNOT",
            channels=[aqubits[0], dq_channels[0, 0]],
        )
        cnot02 = Circuit(
            "CNOT",
            channels=[aqubits[0], dq_channels[2, 2]],
        )
        cnot04 = Circuit(
            "CNOT",
            channels=[aqubits[0], dq_channels[4, 4]],
        )
        cnot06 = Circuit(
            "CNOT",
            channels=[aqubits[0], dq_channels[6, 6]],
        )
        had01 = Circuit("H", channels=[aqubits[0]])

        # Z3Z4Z5Z6
        cnot13 = Circuit(
            "CNOT",
            channels=[aqubits[1], dq_channels[3, 3]][::-1],
        )
        cnot14 = Circuit(
            "CNOT",
            channels=[aqubits[1], dq_channels[4, 4]][::-1],
        )
        cnot15 = Circuit(
            "CNOT",
            channels=[aqubits[1], dq_channels[5, 5]][::-1],
        )
        cnot16 = Circuit(
            "CNOT",
            channels=[aqubits[1], dq_channels[6, 6]][::-1],
        )

        # Z1Z2Z5Z6
        cnot21 = Circuit(
            "CNOT",
            channels=[aqubits[2], dq_channels[1, 1]][::-1],
        )
        cnot22 = Circuit(
            "CNOT",
            channels=[aqubits[2], dq_channels[2, 2]][::-1],
        )
        cnot25 = Circuit(
            "CNOT",
            channels=[aqubits[2], dq_channels[5, 5]][::-1],
        )
        cnot26 = Circuit(
            "CNOT",
            channels=[aqubits[2], dq_channels[6, 6]][::-1],
        )

        # the cnots between ancillas to achieve FT
        acnot02 = Circuit("CNOT", channels=[aqubits[0], aqubits[2]])
        acnot01 = Circuit("CNOT", channels=[aqubits[0], aqubits[1]])
        # the measurement operations
        meas1 = Circuit("Measurement", channels=[aqubits[0], ac_channels[0]])
        meas2 = Circuit("Measurement", channels=[aqubits[1], ac_channels[1]])
        meas3 = Circuit("Measurement", channels=[aqubits[2], ac_channels[2]])

        # construct the circuit
        syndrome_extraction_circ = Circuit(
            "first_round",
            circuit=[
                had00,
                cnot04,
                cnot16,
                cnot25,
                acnot02,
                cnot00,
                cnot14,
                cnot21,
                cnot02,
                cnot13,
                cnot26,
                acnot01,
                cnot06,
                cnot15,
                cnot22,
                had01,
                meas1,
                meas2,
                meas3,
            ],
        )

        # We need to specify the stabilizers that were measured in the order that they
        # were measured.
        # Dictionary has as key the stabilizer and as value the index of the measurement
        # operation (considering only measurement operations).
        measurement_to_stabilizer_map = {
            ac_channels[0].label: stabs_dict["X"][0],
            ac_channels[1].label: stabs_dict["Z"][2],
            ac_channels[2].label: stabs_dict["Z"][1],
        }

        debug_data = is_syndrome_extraction_circuit_valid(
            syndrome_extraction_circ,
            block,
            measurement_to_input_stabilizer_map=measurement_to_stabilizer_map,
        )

        self.assertTrue(debug_data.valid)

    def test_repetition_code_using_ghz(self):
        """Test syndrome extraction for repetition code using a ghz state."""
        # Get block
        block = self.rep_code(3)

        # Define circuit
        # Define data qubits and necessary mappings
        dqubits = [
            Channel(type=ChannelType.QUANTUM, label=str(q)) for q in block.data_qubits
        ]

        # Define ancilla qubits
        n_ancillas = 3
        aqubits = [
            Channel(type=ChannelType.QUANTUM, label=f"auxqubit_{i}")
            for i in range(n_ancillas)
        ]
        ac_channels = [
            Channel(type=ChannelType.CLASSICAL, label=f"c_{aux_q.label}")
            for aux_q in aqubits
        ]

        # Define ancillas that act as classical register
        n_creg_qubits = 2
        creg_qubits = [
            Channel(type=ChannelType.QUANTUM, label=f"creg_qubit_{i}")
            for i in range(n_creg_qubits)
        ]
        cregc_channels = [
            Channel(type=ChannelType.CLASSICAL, label=f"c_{creg_qubit.label}")
            for creg_qubit in creg_qubits
        ]

        # prepare ghz state
        h_a0 = Circuit("H", channels=[aqubits[0]])
        cnot_a0a1 = Circuit("CNOT", channels=[aqubits[0], aqubits[1]])
        cnot_a0a2 = Circuit("CNOT", channels=[aqubits[0], aqubits[2]])

        prepare_ghz = [h_a0, cnot_a0a1, cnot_a0a2]

        # entangle with data qubits
        entangle_registers = [
            Circuit("CNOT", channels=[dqubits[i], aqubits[i]]) for i in range(3)
        ]

        # measure ancillas
        meas_ancillas = [
            Circuit("Measurement", channels=[aqubits[i], ac_channels[i]])
            for i in range(3)
        ]

        # put xor values on the classical qubits
        xor_ops = [
            Circuit("CNOT", channels=[aqubits[i], creg_qubits[j]])
            for j in range(2)
            for i in range(j, j + 2)
        ]

        # measure the two classical qubits
        meas_creg = [
            Circuit("Measurement", channels=[creg_qubits[i], cregc_channels[i]])
            for i in range(2)
        ]

        # Construct circuit
        rep_code_syndrome_extraction = Circuit(
            "rep_code_via_ghz",
            prepare_ghz + entangle_registers + meas_ancillas + xor_ops + meas_creg,
        )

        # Define where the stabilizers will be found
        # The first three measurements are the ancillas
        # The fourth and fifth measurements are the stabilizer values
        measurement_to_stabilizer_map = {
            cregc_channels[0].label: block.stabilizers[0],
            cregc_channels[1].label: block.stabilizers[1],
        }

        # Run Validator and check result
        debug_data = is_syndrome_extraction_circuit_valid(
            rep_code_syndrome_extraction,
            block,
            measurement_to_input_stabilizer_map=measurement_to_stabilizer_map,
        )

        self.assertTrue(debug_data.valid)

    def test_overdefined_code_sec(self):
        """Test syndrome extraction for a repetition code that is overdefined, i.e. it
        has stabilizers that are redundant and are product of the other stabilizers.
        """
        # Define Repetition Code in an overdefined way
        nqubs = 4
        stabilizers = [
            Stabilizer("ZZ", [(0,), (1,)]),
            Stabilizer("ZZ", [(1,), (2,)]),
            Stabilizer("ZZ", [(2,), (3,)]),
            Stabilizer("ZZ", [(0,), (2,)]),  # redundant stabilizer
            Stabilizer("ZZ", [(0,), (3,)]),  # redundant stabilizer
            Stabilizer("ZZ", [(1,), (3,)]),  # redundant stabilizer
        ]
        x_log_op = PauliOperator("X" * nqubs, [(0,), (1,), (2,), (3,)])
        z_log_op = PauliOperator("Z", [(0,)])

        # Get the block and overwrite the stabilizers with the lscrd stabilizers
        block = Block(
            stabilizers=stabilizers,
            logical_x_operators=[x_log_op],
            logical_z_operators=[z_log_op],
        )

        # Define data qubits and necessary mappings
        dqubits = [
            Channel(type=ChannelType.QUANTUM, label=str(q)) for q in block.data_qubits
        ]
        # Define ancilla qubits
        n_ancillas = len(stabilizers)
        aqubits = [
            Channel(type=ChannelType.QUANTUM, label=f"auxqubit_{i}")
            for i in range(n_ancillas)
        ]
        # Define classical channels for ancillas
        ac_channels = [
            Channel(type=ChannelType.CLASSICAL, label=f"c_{aux_q.label}")
            for aux_q in aqubits
        ]

        # CASE 1: Measure all the stabilizers

        # define syndrome extraction circuit
        measure_z0z1 = Circuit(
            "MeasureZ0Z1",
            (
                Circuit("CNOT", channels=(dqubits[0], aqubits[0])),
                Circuit("CNOT", channels=(dqubits[1], aqubits[0])),
                Circuit("Measurement", channels=[aqubits[0], ac_channels[0]]),
            ),
            (dqubits[0], dqubits[1], aqubits[0], ac_channels[0]),
        )

        # Define the rest of the stabilizer measurements
        measure_z1z2 = measure_z0z1.clone(
            (dqubits[1], dqubits[2], aqubits[1], ac_channels[1])
        )
        measure_z2z3 = measure_z0z1.clone(
            (dqubits[2], dqubits[3], aqubits[2], ac_channels[2])
        )
        measure_z0z2 = measure_z0z1.clone(
            (dqubits[0], dqubits[2], aqubits[3], ac_channels[3])
        )
        measure_z0z3 = measure_z0z1.clone(
            (dqubits[0], dqubits[3], aqubits[4], ac_channels[4])
        )
        measure_z1z3 = measure_z0z1.clone(
            (dqubits[1], dqubits[3], aqubits[5], ac_channels[5])
        )

        full_syndrome_extraction = Circuit(
            "full_syndrome_extraction",
            (
                # Measure the stabilizers in the order they were defined
                measure_z0z1,
                measure_z1z2,
                measure_z2z3,
                measure_z0z2,
                measure_z0z3,
                measure_z1z3,
            ),
        )

        # Use validator to check the circuit
        debug_data = is_syndrome_extraction_circuit_valid(
            circuit=full_syndrome_extraction,
            input_block=block,
            measurement_to_input_stabilizer_map={
                ac_channels[i].label: stabilizers[i] for i in range(len(stabilizers))
            },
        )

        self.assertTrue(debug_data.valid)

        # CASE 2: Measure the final stabilizer only
        # Define the circuit
        full_sec_z1z3 = Circuit(
            "full_sec_z1z3",
            (measure_z1z3,),
        )

        # Use validator to check the circuit
        debug_data_z1z3 = is_syndrome_extraction_circuit_valid(
            circuit=full_sec_z1z3,
            input_block=block,
            measurement_to_input_stabilizer_map={ac_channels[5].label: stabilizers[5]},
        )

        # Check that the circuit is valid
        self.assertTrue(debug_data_z1z3.valid)

    def test_stab_mres_output(self):
        """Test the StabilizerMeasurementCheckOutput class. This is done by creating a
        circuit that doesn't fully correspond to the stabilizer measurements that are
        expected to be made. The test checks that the output of the check is as
        expected."""
        # Define repetition code of distance 4
        block = self.rep_code(4)

        n_aux_qubits = 4
        data_qubits = [Channel("quantum", str(q)) for q in block.data_qubits]
        aux_qubits = [
            Channel("quantum", str((aux_q, 1))) for aux_q in range(n_aux_qubits)
        ]
        c_channels = [
            Channel("classical", f"c_{aux_q_chan.label}") for aux_q_chan in aux_qubits
        ]
        # anc0: Define a probabilistic measurement
        anc0_circ = Circuit(
            "anc0_circ",
            circuit=[
                Circuit("H", channels=[aux_qubits[0]]),
                Circuit("Measurement", channels=[aux_qubits[0], c_channels[0]]),
            ],
        )

        # anc1: Define a deterministic measurement that doesn't measure any stabilizer
        anc1_circ = Circuit("Measurement", channels=[aux_qubits[1], c_channels[1]])

        # anc2: Measure stabilizer 0
        anc2_circ = Circuit(
            "anc2_circ",
            circuit=[
                Circuit("CNOT", channels=[data_qubits[0], aux_qubits[2]]),
                Circuit("CNOT", channels=[data_qubits[1], aux_qubits[2]]),
                Circuit("Measurement", channels=[aux_qubits[2], c_channels[2]]),
            ],
        )

        # anc3: Measure stabilizer 1 and 2
        anc3_circ = Circuit(
            "anc3_circ",
            circuit=[
                # Project stabilizer 1
                Circuit("CNOT", channels=[data_qubits[1], aux_qubits[3]]),
                Circuit("CNOT", channels=[data_qubits[2], aux_qubits[3]]),
                # Project stabilizer 2
                Circuit("CNOT", channels=[data_qubits[2], aux_qubits[3]]),
                Circuit("CNOT", channels=[data_qubits[3], aux_qubits[3]]),
                # Measure
                Circuit("Measurement", channels=[aux_qubits[3], c_channels[3]]),
            ],
        )

        # Get full circuit
        full_circuit = Circuit(
            "stab_mres",
            circuit=[
                anc0_circ,
                anc1_circ,
                anc2_circ,
                anc3_circ,
            ],
        )

        # Create a check that doesn't fully correspond to the above circuit
        measurement_to_input_stabilizer_map = {
            c_channels[0].label: block.stabilizers[2],
            c_channels[1].label: block.stabilizers[0],
            c_channels[2].label: block.stabilizers[0],  # Only this one is correct
            c_channels[3].label: block.stabilizers[1],
        }

        debug_data = is_syndrome_extraction_circuit_valid(
            full_circuit,
            block,
            measurement_to_input_stabilizer_map=measurement_to_input_stabilizer_map,
        )

        # Check that it failed without failing stabilizer and logical operator checks
        self.assertFalse(debug_data.valid)
        self.assertTrue(debug_data.checks.code_stabilizers.valid)
        self.assertTrue(debug_data.checks.logical_operators.valid)

        # Ensure the correct message is given
        self.assertEqual(
            debug_data.checks.stabilizers_measured.message,
            (
                "Some measurement(s) were not deterministic and some did not "
                "measure the assigned stabilizer."
            ),
        )

        # Check that the output is as expected
        stab_mres_output = debug_data.checks.stabilizers_measured.output

        # The first measurement is probabilistic
        self.assertEqual(
            stab_mres_output.probabilistic_measurements, (c_channels[0].label,)
        )

        # Check that expected vs measured stabilizers are as expected
        exp_v_meas_stabs = stab_mres_output.expected_vs_measured_stabs
        # First entry for anc1:
        # The deterministic measurement that doesn't measure any stabilizer should
        # be an empty tuple
        self.assertEqual(exp_v_meas_stabs[0][2], ())
        # Second entry for anc2:
        # The circuit incorrectly measures stabilizers 1 and 2 instead of 0
        self.assertEqual(
            set(exp_v_meas_stabs[1][2]),
            {block.stabilizers[1], block.stabilizers[2]},
        )

        # Check correct message for probabilistic measurements
        self.assertEqual(
            is_syndrome_extraction_circuit_valid(
                full_circuit,
                block,
                measurement_to_input_stabilizer_map={
                    c_channels[0].label: block.stabilizers[2]
                },
            ).checks.stabilizers_measured.message,
            "Some measurement(s) were not deterministic.",
        )

        # Check correct message for incorrect stabilizer measurements
        self.assertEqual(
            is_syndrome_extraction_circuit_valid(
                full_circuit,
                block,
                measurement_to_input_stabilizer_map={
                    c_channels[1].label: block.stabilizers[0]
                },
            ).checks.stabilizers_measured.message,
            "Some measurement(s) did not measure the assigned stabilizer.",
        )


if __name__ == "__main__":
    unittest.main()
