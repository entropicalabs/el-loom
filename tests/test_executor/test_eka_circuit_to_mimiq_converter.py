"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

import unittest
import random

import mimiqcircuits as mc

from loom.executor import convert_circuit_to_mimiq
from loom.executor.eka_circuit_to_mimiq_converter import (
    MEASUREMENT_OPERATIONS_MAP,
    SINGLE_QUBIT_GATE_OPERATIONS_MAP,
    RESET_OPERATIONS_MAP,
    TWO_QUBIT_GATE_OPERATIONS_MAP,
    CLASSICALLY_CONTROLLED_OPERATIONS_SQ_MAP,
    CLASSICALLY_CONTROLLED_OPERATIONS_TQ_MAP,
    ALL_OPERATIONS_MAP,
)
from loom.eka import (
    Circuit,
    Channel,
    ChannelType,
    Eka,
    Lattice,
    Block,
    Stabilizer,
    PauliOperator,
)
from loom.eka.operations import MeasureBlockSyndromes
from loom.interpreter import interpret_eka


# pylint: disable=too-many-instance-attributes, duplicate-code
class TestEkaCircuitToMimiqConverter(unittest.TestCase):
    """Test the conversion of EKA circuits to MIMIQ circuits."""

    def setUp(self):
        # Create channels and a mock measureop for the tests
        self.n_qubits = 10
        self.q_channels = [Channel("quantum", f"q{i}") for i in range(self.n_qubits)]
        self.c_channels = [Channel("classical", f"c{i}") for i in range(self.n_qubits)]

        # A circuit with all single qubit operations
        self.sq_circuit = Circuit(
            name="Single qubit operations",
            circuit=[
                [Circuit(name=op_name, channels=[random.choice(self.q_channels)])]
                for op_name in SINGLE_QUBIT_GATE_OPERATIONS_MAP.keys()
            ],
        )
        # A circuit with all two qubit operations
        self.tq_circuit = Circuit(
            name="Two qubit operations",
            circuit=[
                [Circuit(name=op_name, channels=random.sample(self.q_channels, 2))]
                for op_name in TWO_QUBIT_GATE_OPERATIONS_MAP.keys()
            ],
        )
        # A circuit with all measurement operations
        self.m_circuit = Circuit(
            name="Measurement operations",
            circuit=[
                [
                    Circuit(
                        name=op_name,
                        channels=[
                            random.choice(self.q_channels),
                            random.choice(self.c_channels),
                        ],
                    )
                ]
                for op_name in MEASUREMENT_OPERATIONS_MAP.keys()
            ],
        )
        # A circuit with all classically controlled operations
        self.cc_circuit = Circuit(
            name="Classically controlled operations",
            circuit=[
                Circuit(
                    op_name,
                    channels=[random.choice(self.q_channels), self.c_channels[0]],
                )
                for op_name in CLASSICALLY_CONTROLLED_OPERATIONS_SQ_MAP.keys()
            ]
            + [
                Circuit(
                    op_name,
                    channels=[
                        *random.sample(self.q_channels, 2),
                        self.c_channels[0],
                    ],
                )
                for op_name in CLASSICALLY_CONTROLLED_OPERATIONS_TQ_MAP.keys()
            ],
        )
        # A circuit with all reset operations
        self.r_circuit = Circuit(
            name="Reset operations",
            circuit=[
                [
                    Circuit(
                        op_name,
                        channels=[random.choice(self.q_channels)],
                    )
                ]
                for op_name in RESET_OPERATIONS_MAP.keys()
            ],
        )

        # Mock MeasureStabilizerSyndrome Operation for Rotated Surface Code
        lattice = Lattice.square_2d((4, 4))
        repc = Block(
            stabilizers=(
                Stabilizer(
                    pauli="ZZ",
                    data_qubits=((0, 0, 0), (1, 0, 0)),
                    ancilla_qubits=((0, 0, 1),),
                ),
                Stabilizer(
                    pauli="ZZ",
                    data_qubits=((1, 0, 0), (2, 0, 0)),
                    ancilla_qubits=((1, 0, 1),),
                ),
            ),
            logical_x_operators=[
                PauliOperator(
                    pauli="XXX", data_qubits=((0, 0, 0), (1, 0, 0), (2, 0, 0))
                )
            ],
            logical_z_operators=[PauliOperator(pauli="Z", data_qubits=((0, 0, 0),))],
            unique_label="q1",
        )
        input_eka = Eka(
            lattice,
            blocks=[repc],
            operations=[MeasureBlockSyndromes(repc.unique_label, n_cycles=3)],
        )
        self.meas_circ = interpret_eka(input_eka).final_circuit

    def get_non_parallelized_instruction_list(
        self,
        register_dict: dict[str, dict[int, Channel]],
        eka_circuit: Circuit,
    ) -> list[mc.Instruction]:
        """
        Get the list of instructions for a non-parallelized eka circuit.
        """
        instruction_list = []
        for eka_op in eka_circuit.circuit:
            # Find corresponding operation in the mapping
            mimiq_op = ALL_OPERATIONS_MAP.get(eka_op[0].name)
            # Find the indices of the channels in the register_dict
            qubit_idxs = tuple(
                idx
                for eka_chan in eka_op[0].channels
                for idx, chan in register_dict["quantum"].items()
                if chan == eka_chan
            )
            bit_idxs = tuple(
                idx
                for eka_chan in eka_op[0].channels
                for idx, chan in register_dict["classical"].items()
                if chan == eka_chan
            )

            # Create the expected instruction and add it to the list
            if isinstance(mimiq_op, tuple):
                # If the operation is a tuple, it means it has multiple operations
                for op in mimiq_op:
                    instruction_list += [mc.Instruction(op, qubit_idxs, bit_idxs)]
            else:
                instruction_list += [mc.Instruction(mimiq_op, qubit_idxs, bit_idxs)]

        return instruction_list

    def test_single_qubit_operations(self):
        """Test that all single qubit operations in the mapping of EKA to MIMIQ are
        mapped to operations that can be executed."""

        # Convert the circuit
        mimiq_circuit, register_dict = convert_circuit_to_mimiq(self.sq_circuit)
        # Get the expected instruction list
        expected_instr_list = self.get_non_parallelized_instruction_list(
            register_dict, self.sq_circuit
        )
        self.assertEqual(mimiq_circuit.instructions, expected_instr_list)

    def test_two_qubit_operations(self):
        """Test that all two qubit operations in the mapping of EKA to MIMIQ are
        mapped to operations that can be executed."""

        # Convert the circuit
        mimiq_circuit, register_dict = convert_circuit_to_mimiq(self.tq_circuit)
        # Get the expected instruction list
        expected_instr_list = self.get_non_parallelized_instruction_list(
            register_dict, self.tq_circuit
        )
        self.assertEqual(mimiq_circuit.instructions, expected_instr_list)

    def test_measurement_operations(self):
        """Test that all measurement operations in the mapping of EKA to MIMIQ are
        mapped to operations that can be executed."""

        # Convert the circuit
        mimiq_circuit, register_dict = convert_circuit_to_mimiq(self.m_circuit)
        # Get the expected instruction list
        expected_instr_list = self.get_non_parallelized_instruction_list(
            register_dict, self.m_circuit
        )
        self.assertEqual(mimiq_circuit.instructions, expected_instr_list)

    def test_classically_controlled_operations(self):
        """Test that all classically controlled operations in the mapping of EKA to
        MIMIQ are mapped to operations that can be executed."""

        # Convert the circuit
        mimiq_circuit, register_dict = convert_circuit_to_mimiq(self.cc_circuit)
        # Get the expected instruction list
        expected_instr_list = self.get_non_parallelized_instruction_list(
            register_dict, self.cc_circuit
        )
        self.assertEqual(mimiq_circuit.instructions, expected_instr_list)

    def test_reset_operations(self):
        """Test that all reset operations in the mapping of EKA to MIMIQ are
        mapped to operations that can be executed."""

        # Convert the circuit
        mimiq_circuit, register_dict = convert_circuit_to_mimiq(self.r_circuit)
        # Get the expected instruction list
        expected_instr_list = self.get_non_parallelized_instruction_list(
            register_dict, self.r_circuit
        )
        self.assertEqual(mimiq_circuit.instructions, expected_instr_list)

    def test_parallelization(self):
        """Test that all operations in the mapping of EKA to MIMIQ are mapped to
        operations that can be executed in parallel."""

        q_channels = random.sample(self.q_channels, 10)
        parallel_circuit = Circuit(
            name="Parallel x operations",
            circuit=[[Circuit("x", channels=[q_chan]) for q_chan in q_channels]],
        )

        # Convert the circuit
        mimiq_circuit, register_dict = convert_circuit_to_mimiq(parallel_circuit)

        # Check if the circuit has been converted to a single parallel instruction
        self.assertEqual(len(mimiq_circuit.instructions), 1)

        # Construct the expected instruction
        qubit_idxs = tuple(
            idx
            for eka_op in parallel_circuit.circuit[0]
            for idx, chan in register_dict["quantum"].items()
            if eka_op.channels[0] == chan
        )
        n_repeats = len(qubit_idxs)
        expected_instruction = mc.Instruction(
            mc.Parallel(n_repeats, mc.GateX()), qubit_idxs
        )

        self.assertEqual(mimiq_circuit.instructions[0], expected_instruction)

    def test_non_parallelization(self):
        """Test that all operations in the mapping of EKA to MIMIQ are mapped to
        operations that can be executed in parallel."""

        non_parallel_circuit = Circuit(
            name="Parallel measurement operations",
            circuit=[
                [
                    Circuit("measurement", channels=[q_chan, c_chan])
                    for q_chan, c_chan in zip(
                        self.q_channels, self.c_channels, strict=True
                    )
                ]
            ],
        )

        # Convert the circuit
        mimiq_circuit, _ = convert_circuit_to_mimiq(non_parallel_circuit)

        # Check that the instructions have not been parallelized because they do not
        # correspond to a mimiqcircuits.Gate operation
        self.assertEqual(len(mimiq_circuit.instructions), len(self.q_channels))

    def test_register_dict_mappings(self):
        """Test that registers conversion in the mapping of EKA to MIMIQ are always
        sorted."""
        circuit_list = [
            self.sq_circuit,
            self.tq_circuit,
            self.m_circuit,
            self.cc_circuit,
            self.r_circuit,
            self.meas_circ,
        ]
        for circuit in circuit_list:
            expected_register_dict = {
                "quantum": dict(
                    enumerate(
                        sorted(
                            [
                                chan
                                for chan in circuit.channels
                                if chan.type != ChannelType.CLASSICAL
                            ],
                            key=lambda x: x.label,
                        )
                    )
                ),
                "classical": dict(
                    enumerate(
                        sorted(
                            [
                                chan
                                for chan in circuit.channels
                                if chan.type == ChannelType.CLASSICAL
                            ],
                            key=lambda x: x.label,
                        )
                    )
                ),
            }
            _, register_dict = convert_circuit_to_mimiq(circuit)

            self.assertEqual(register_dict, expected_register_dict)


if __name__ == "__main__":
    unittest.main()
