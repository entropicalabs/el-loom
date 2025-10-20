"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

import unittest

from loom.eka import (
    Stabilizer,
    PauliOperator,
    Lattice,
    SyndromeCircuit,
    Circuit,
    Channel,
)

from loom_five_qubit_perfect_code.code_factory import FiveQubitPerfectCode


class TestFiveQubitPerfectCode(unittest.TestCase):
    """
    Test the functionalities of the FiveQubitPerfectCode class, which is a
    subclass of the Block class.
    """

    def setUp(self):
        """Define the generic properties of the five qubit perfect code"""

        # Define the stabilizers
        # NOTE: All of the stabilizers are labelled with "XZZX" but with different data
        # qubits. This is as the stabilizer class does not accept "I" as a stabilizer,
        # and simply dropping the "I" label will throw a non-commuting validation error.
        self.stabilizers = [
            Stabilizer(
                pauli="XZZX",  # XZZXI
                data_qubits=[(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3)],
                ancilla_qubits=[(0, 0, 5)],
            ),
            Stabilizer(
                pauli="XZZX",  # IXZZX
                data_qubits=[(0, 0, 1), (0, 0, 2), (0, 0, 3), (0, 0, 4)],
                ancilla_qubits=[(0, 0, 6)],
            ),
            Stabilizer(
                pauli="XXZZ",  # XIXZZ
                data_qubits=[(0, 0, 0), (0, 0, 2), (0, 0, 3), (0, 0, 4)],
                ancilla_qubits=[(0, 0, 7)],
            ),
            Stabilizer(
                pauli="ZXXZ",  # ZXIXZ
                data_qubits=[(0, 0, 0), (0, 0, 1), (0, 0, 3), (0, 0, 4)],
                ancilla_qubits=[(0, 0, 8)],
            ),
        ]

        # Define the logical operators
        self.logical_x_operator = PauliOperator(
            pauli="X" * 5,
            data_qubits=[(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3), (0, 0, 4)],
        )
        self.logical_z_operator = PauliOperator(
            pauli="Z" * 5,
            data_qubits=[(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3), (0, 0, 4)],
        )

        # Define the syndrome extraction circuits
        d_channels = [Channel(label=f"d{i}") for i in range(4)]
        a_channels = [Channel(label=f"a{i}", type="quantum") for i in range(1)]
        c_channels = [Channel(label=f"c{i}", type="classical") for i in range(1)]

        self.syndrome_circuits = {
            "XZZXI": SyndromeCircuit(
                pauli="XZZX",
                name="XZZXI_syndrome_extraction",
                circuit=Circuit(
                    name="XZZXI_syndrome_extraction",
                    circuit=[
                        [Circuit("Reset_0", channels=a_channels)],
                        [Circuit("H", channels=a_channels)],
                        [Circuit("CX", channels=[a_channels[0], d_channels[0]])],
                        [Circuit("CZ", channels=[a_channels[0], d_channels[1]])],
                        [Circuit("CZ", channels=[a_channels[0], d_channels[2]])],
                        [Circuit("CX", channels=[a_channels[0], d_channels[3]])],
                        [],  # Identity
                        [],
                        [],
                        [],
                        [Circuit("H", channels=a_channels)],
                        [
                            Circuit(
                                "Measurement", channels=[a_channels[0], c_channels[0]]
                            )
                        ],
                    ],
                ),
            ),
            "IXZZX": SyndromeCircuit(
                pauli="XZZX",
                name="IXZZX_syndrome_extraction",
                circuit=Circuit(
                    name="IXZZX_syndrome_extraction",
                    circuit=[
                        [Circuit("Reset_0", channels=a_channels)],
                        [Circuit("H", channels=a_channels)],
                        [],
                        [],  # Identity
                        [Circuit("CX", channels=[a_channels[0], d_channels[0]])],
                        [Circuit("CZ", channels=[a_channels[0], d_channels[1]])],
                        [Circuit("CZ", channels=[a_channels[0], d_channels[2]])],
                        [Circuit("CX", channels=[a_channels[0], d_channels[3]])],
                        [],
                        [],
                        [Circuit("H", channels=a_channels)],
                        [
                            Circuit(
                                "Measurement", channels=[a_channels[0], c_channels[0]]
                            )
                        ],
                    ],
                ),
            ),
            "XIXZZ": SyndromeCircuit(
                pauli="XXZZ",
                name="XIXZZ_syndrome_extraction",
                circuit=Circuit(
                    name="XIXZZ_syndrome_extraction",
                    circuit=[
                        [Circuit("Reset_0", channels=a_channels)],
                        [Circuit("H", channels=a_channels)],
                        [],
                        [],
                        [Circuit("CX", channels=[a_channels[0], d_channels[0]])],
                        [],  # Identity
                        [Circuit("CX", channels=[a_channels[0], d_channels[1]])],
                        [Circuit("CZ", channels=[a_channels[0], d_channels[2]])],
                        [Circuit("CZ", channels=[a_channels[0], d_channels[3]])],
                        [],
                        [Circuit("H", channels=a_channels)],
                        [
                            Circuit(
                                "Measurement", channels=[a_channels[0], c_channels[0]]
                            )
                        ],
                    ],
                ),
            ),
            "ZXIXZ": SyndromeCircuit(
                pauli="ZXXZ",
                name="ZXIXZ_syndrome_extraction",
                circuit=Circuit(
                    name="ZXIXZ_syndrome_extraction",
                    circuit=[
                        [Circuit("Reset_0", channels=a_channels)],
                        [Circuit("H", channels=a_channels)],
                        [],
                        [],
                        [],
                        [Circuit("CZ", channels=[a_channels[0], d_channels[0]])],
                        [Circuit("CX", channels=[a_channels[0], d_channels[1]])],
                        [],  # Identity
                        [Circuit("CX", channels=[a_channels[0], d_channels[2]])],
                        [Circuit("CZ", channels=[a_channels[0], d_channels[3]])],
                        [Circuit("H", channels=a_channels)],
                        [
                            Circuit(
                                "Measurement", channels=[a_channels[0], c_channels[0]]
                            )
                        ],
                    ],
                ),
            ),
        }

        # Define the stabilizer to circuit mapping
        stabilizer_paulis = ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"]
        self.stabilizer_to_circuit = {
            stab.uuid: self.syndrome_circuits[stabilizer_paulis[index]].uuid
            for index, stab in enumerate(self.stabilizers)
        }

        # Define the five qubit perfect code
        self.position = (2, 1)
        self.five_qubit_perfect_code = FiveQubitPerfectCode.create(
            lattice=Lattice.poly_2d(n=5, anc=4),
            unique_label="q1",
            position=self.position,
        )

    def test_syndrome_extraction_circuit_generation(self):
        """Test the correct generation of the syndrome extraction circuit"""

        # Stabilizer types to be considered
        stabilizer_paulis = ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"]

        # Check circuits are correct
        for pauli in stabilizer_paulis:
            expected_syndrome_circuit = self.syndrome_circuits[pauli]
            syndrome_circuit = (
                FiveQubitPerfectCode.generate_syndrome_extraction_circuits(pauli)
            )

            self.assertEqual(syndrome_circuit, expected_syndrome_circuit)

    def test_five_qubit_perfect_code_creation(self):
        """Test the correct creation of the five qubit perfect code datablock"""

        # Define Block manually
        manual_code = FiveQubitPerfectCode(
            unique_label="q1",
            stabilizers=self.stabilizers,
            logical_x_operators=[self.logical_x_operator],
            logical_z_operators=[self.logical_z_operator],
            syndrome_circuits=list(self.syndrome_circuits.values()),
            stabilizer_to_circuit=self.stabilizer_to_circuit,
        ).shift(self.position)

        self.assertEqual(self.five_qubit_perfect_code, manual_code)


if __name__ == "__main__":
    unittest.main()
