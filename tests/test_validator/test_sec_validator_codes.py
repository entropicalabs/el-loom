"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

import unittest

from loom import validator
from loom.eka import Block, SyndromeCircuit, Circuit, Channel, Stabilizer, PauliOperator


# pylint: disable=duplicate-code
class TestSECValidator(unittest.TestCase):
    """
    Test cases for validating syndrome extraction circuits using the Validator module.
    """

    def setUp(self):
        # RepetitionCode.create(3, "Z", Lattice.linear()),
        # RepetitionCode.create(3, "X", Lattice.linear()),
        # RotatedSurfaceCode.create(3, 3, Lattice.square_2d()),
        # SteaneCode.create(Lattice.square_2d()),
        # BivariateBicycleCode.create(
        #     (3, 3), [(0, 0), (1, 0), (1, 2)], [(0, 0), (0, 1), (2, 1)]
        # ),
        repc = Block(
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
                for i in range(2)
            ),
            logical_x_operators=(PauliOperator("Z", ((0, 0),)),),
            logical_z_operators=(
                PauliOperator("X" * 3, tuple((i, 0) for i in range(3))),
            ),
        )
        rsc = Block(
            stabilizers=(
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
                Stabilizer("XX", ((0, 0, 0), (0, 1, 0)), ancilla_qubits=((0, 1, 1),)),
                Stabilizer("XX", ((2, 1, 0), (2, 2, 0)), ancilla_qubits=((3, 2, 1),)),
                Stabilizer("ZZ", ((2, 0, 0), (1, 0, 0)), ancilla_qubits=((2, 0, 1),)),
                Stabilizer("ZZ", ((1, 2, 0), (0, 2, 0)), ancilla_qubits=((1, 3, 1),)),
            ),
            logical_x_operators=[
                PauliOperator(
                    pauli="XXX", data_qubits=tuple((i, 0, 0) for i in range(3))
                )
            ],
            logical_z_operators=[
                PauliOperator(
                    pauli="ZZZ", data_qubits=((0, 0, 0), (0, 1, 0), (0, 2, 0))
                )
            ],
        )
        self.blocks_to_test: list[Block] = [
            repc,
            rsc,
        ]

    @staticmethod
    def get_block_default_sec(block: Block) -> Circuit:
        """Given a Block object, return the default syndrome extraction circuit that
        measures all of the stabilizers in the block in the order in which they appear.
        """
        # Create data qubit channels appropriately labeled
        data_qubit_to_channel_map = {
            q: Channel("quantum", str(q)) for q in block.data_qubits
        }
        # Create ancilla channels
        ancilla_channels = [
            Channel("quantum", str(a))
            for stab in block.stabilizers
            for a in stab.ancilla_qubits
        ]
        # Create classical channels
        classical_channels = [
            Channel("classical", f"c_{a.label}_0") for a in ancilla_channels
        ]

        subcircuits = [
            SyndromeCircuit(stab.pauli).circuit.clone(
                [data_qubit_to_channel_map[qub] for qub in stab.data_qubits]
                + [ancilla_channels[idx]]
                + [classical_channels[idx]]
            )
            for idx, stab in enumerate(block.stabilizers)
        ]

        return Circuit(
            "full_sec",
            subcircuits,
            list(data_qubit_to_channel_map.values())
            + ancilla_channels
            + classical_channels,
        )

    def test_default(self):
        """Test that default circuits pass the tests."""
        for block in self.blocks_to_test:
            def_circ = self.get_block_default_sec(block)
            # Get the measurement channel for every stabilizer
            # This is correct under the assumption that the default circuit
            # measures them in the order in which they appear
            classical_channels = [
                chan for chan in def_circ.channels if chan.type == "classical"
            ]
            measurement_to_stabilizer_map = {
                c_chan.label: stab
                for c_chan, stab in zip(
                    classical_channels, block.stabilizers, strict=True
                )
            }

            debug_data = validator.is_syndrome_extraction_circuit_valid(
                def_circ, block, measurement_to_stabilizer_map
            )
            self.assertTrue(debug_data.valid)

    def test_default_add_cnot(self):
        """Test that default circuits with an added CNOT don't pass the tests."""
        for block in self.blocks_to_test:
            # Get the default circuit and the data qubit to channel mapping
            def_circ = self.get_block_default_sec(block)
            circ_data_channels = {
                qub: next(chan for chan in def_circ.channels if chan.label == str(qub))
                for qub in block.data_qubits
            }
            # Get the measurement channel for every stabilizer
            # This is correct under the assumption that the default circuit
            # measures them in the order in which they appear
            classical_channels = [
                chan for chan in def_circ.channels if chan.type == "classical"
            ]
            measurement_to_stabilizer_map = {
                c_chan.label: stab
                for c_chan, stab in zip(
                    classical_channels, block.stabilizers, strict=True
                )
            }

            # append an extra CNOT gate between 2 data qubits
            extra_op = Circuit(
                "CNOT",
                channels=[
                    circ_data_channels[block.data_qubits[0]],
                    circ_data_channels[block.data_qubits[1]],
                ],
            )
            def_circ = Circuit(def_circ.name, def_circ.circuit + ((extra_op,),))

            debug_data = validator.is_syndrome_extraction_circuit_valid(
                def_circ, block, measurement_to_stabilizer_map
            )

            # invalid
            self.assertFalse(debug_data.valid)
            # but the stabilizers were correctly measured!
            self.assertTrue(debug_data.checks.stabilizers_measured.valid)

    def test_default_add_log_operation(self):
        """Test that default circuits with an added logical operation
        don't pass the tests because only the LogicalState was altered."""
        for block in self.blocks_to_test:
            # Get the default circuit and the data qubit to channel mapping
            def_circ = self.get_block_default_sec(block)
            circ_data_channels = {
                qub: next(chan for chan in def_circ.channels if chan.label == str(qub))
                for qub in block.data_qubits
            }

            # Get the measurement channel for every stabilizer
            # This is correct under the assumption that the default circuit
            # measures them in the order in which they appear
            classical_channels = [
                chan for chan in def_circ.channels if chan.type == "classical"
            ]
            measurement_to_stabilizer_map = {
                c_chan.label: stab
                for c_chan, stab in zip(
                    classical_channels, block.stabilizers, strict=True
                )
            }

            # find a logical operator
            log_operator = block.logical_z_operators[0]
            # apply the logical operator in the end of the circuit
            # skip the first character (sign)
            extra_ops = tuple(
                (Circuit(p, channels=[circ_data_channels[qub]]),)
                for qub, p in zip(
                    log_operator.data_qubits, log_operator.pauli, strict=True
                )
            )

            def_circ = Circuit(def_circ.name, def_circ.circuit + extra_ops)

            debug_data = validator.is_syndrome_extraction_circuit_valid(
                def_circ, block, measurement_to_stabilizer_map
            )

            # make sure that it is invalid only because the logical state was altered
            self.assertFalse(debug_data.valid)
            self.assertFalse(debug_data.checks.logical_operators.valid)
            self.assertTrue(debug_data.checks.code_stabilizers.valid)
            self.assertTrue(debug_data.checks.stabilizers_measured.valid)

    def test_default_add_code_destabilizer(self):
        """Test that default circuits with an added code destabilizer operation
        don't pass the tests because only the CodeStabilizers were altered."""
        which_destab = 0
        for block in self.blocks_to_test:
            # Get the default circuit and the data qubit to channel mapping
            def_circ = self.get_block_default_sec(block)
            circ_data_channels = {
                qub: next(chan for chan in def_circ.channels if chan.label == str(qub))
                for qub in block.data_qubits
            }

            c_channels = [
                chan for chan in def_circ.channels if chan.type == "classical"
            ]
            # Get the measurement channel for every stabilizer
            # This is correct under the assumption that the default circuit
            # measures them in the order in which they appear
            measurement_to_stabilizer_map = {
                c_chan.label: stab
                for c_chan, stab in zip(c_channels, block.stabilizers, strict=True)
            }

            # find a destabilizing operator
            destab_operator_str = str(block.destabarray[which_destab])

            # apply the destabilizer operator in the end of the circuit
            # skip the first character (sign)
            extra_ops = tuple(
                (Circuit(p, channels=[circ_data_channels[block.data_qubits[i]]]),)
                for i, p in enumerate(destab_operator_str[1:])
                if p != "_"
            )
            def_circ = Circuit(def_circ.name, def_circ.circuit + extra_ops)

            debug_data = validator.is_syndrome_extraction_circuit_valid(
                def_circ, block, measurement_to_stabilizer_map
            )

            # make sure that it is invalid only because the code stabilizers
            # were altered
            self.assertFalse(debug_data.valid)
            self.assertTrue(debug_data.checks.logical_operators.valid)
            self.assertTrue(debug_data.checks.stabilizers_measured.valid)
            self.assertFalse(debug_data.checks.code_stabilizers.valid)

            # Destabilizer destabilizes the following stabilizers in the code:
            # Destabarray destabilizes the reduced_stabarray so we need to find which
            # stabilizers in the original stabarray contain the stabilizer that is
            # destabilized by the destabilizer with index which_destab.
            # (we need to use bookkeeping_inv to find the correct stabilizers)
            stabs_removed = tuple(
                block.stabilizers[idx]
                for idx in range(len(block.bookkeeping))
                if block.bookkeeping_inv[idx, which_destab]
            )
            self.assertEqual(
                (
                    debug_data.checks.code_stabilizers.output
                ).stabilizers_with_incorrect_parity,
                stabs_removed,
            )

    def test_multiple_stabilizer_measurement(self):
        """Test that default circuits with multiple measurements of the same stabilizer
        still pass the tests."""
        for block in self.blocks_to_test:
            # Get the default circuit and the data qubit to channel mapping
            def_circ = self.get_block_default_sec(block)

            ancilla_channels = [
                chan
                for chan in def_circ.channels
                if chan.type != "classical"
                and chan.label not in [str(qub) for qub in block.data_qubits]
            ]
            # First stabilizer measurement channels
            first_classical_channels = [
                chan for chan in def_circ.channels if chan.type == "classical"
            ]
            anc_channel_labels = [
                str(a_qubit)
                for stab in block.stabilizers
                for a_qubit in stab.ancilla_qubits
            ]

            # reset the ancillas
            _ = [chan for chan in def_circ.channels if chan.label in anc_channel_labels]
            reset_gates = tuple(
                (Circuit("reset", channels=[anc]),) for anc in ancilla_channels
            )

            # Second stabilizer measurement channels
            second_classical_channels = [
                Channel("classical", f"c_{anc}_1") for anc in ancilla_channels
            ]
            def_circ_2 = def_circ.clone(
                [
                    chan
                    for chan in def_circ.channels
                    if chan not in first_classical_channels
                ]
                + second_classical_channels
            )

            # construct the final circuit
            # NOTE the reset gates are defined individually, whereas validated circuit
            # comprises a list of lists of circuits
            final_circ = Circuit(
                "double_stabilizer_measurement",
                def_circ.circuit + reset_gates + def_circ_2.circuit,
            )

            # Get the measurement indexes for every stabilizer
            # This is correct under the assumption that the default circuit
            # measures them in the order in which they appear
            measurement_to_stabilizer_map = {
                chan.label: stab
                for chan, stab in zip(
                    first_classical_channels, block.stabilizers, strict=True
                )
            } | {
                chan.label: stab
                for chan, stab in zip(
                    second_classical_channels, block.stabilizers, strict=True
                )
            }

            debug_data = validator.is_syndrome_extraction_circuit_valid(
                final_circ, block, measurement_to_stabilizer_map
            )

            # assert that the circuit is valid
            self.assertTrue(debug_data.valid)


if __name__ == "__main__":
    unittest.main()
