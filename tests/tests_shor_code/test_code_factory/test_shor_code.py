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

from loom_shor_code.code_factory import ShorCode

# pylint: disable=duplicate-code


class TestShorCode(unittest.TestCase):
    """
    Test the functionalities of the ShorCode class, which is a
    subclass of the Block class.
    """

    def setUp(self):
        """Define the generic properties of the Shor code"""

        # Define the stabilizers
        self.stabilizers = [
            Stabilizer(
                pauli="X" * 6,
                data_qubits=[(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)],
                ancilla_qubits=[(0, 1)],
            ),
            Stabilizer(
                pauli="X" * 6,
                data_qubits=[(3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0)],
                ancilla_qubits=[(1, 1)],
            ),
            Stabilizer(
                pauli="ZZ",
                data_qubits=[(0, 0), (1, 0)],
                ancilla_qubits=[(2, 1)],
            ),
            Stabilizer(
                pauli="ZZ",
                data_qubits=[(1, 0), (2, 0)],
                ancilla_qubits=[(3, 1)],
            ),
            Stabilizer(
                pauli="ZZ",
                data_qubits=[(3, 0), (4, 0)],
                ancilla_qubits=[(4, 1)],
            ),
            Stabilizer(
                pauli="ZZ",
                data_qubits=[(4, 0), (5, 0)],
                ancilla_qubits=[(5, 1)],
            ),
            Stabilizer(
                pauli="ZZ",
                data_qubits=[(6, 0), (7, 0)],
                ancilla_qubits=[(6, 1)],
            ),
            Stabilizer(
                pauli="ZZ",
                data_qubits=[(7, 0), (8, 0)],
                ancilla_qubits=[(7, 1)],
            ),
        ]

        # Define the logical operators
        self.logical_x_operator = PauliOperator(
            pauli="X" * 9,
            data_qubits=[(i, 0) for i in range(9)],
        )
        self.logical_z_operator = PauliOperator(
            pauli="Z" * 9,
            data_qubits=[(i, 0) for i in range(9)],
        )

        # Define the syndrome extraction circuits
        d_channels = [Channel(label=f"d{i}") for i in range(6)]
        a_channels = [Channel(label=f"a{i}", type="quantum") for i in range(1)]
        c_channels = [Channel(label=f"c{i}", type="classical") for i in range(1)]

        self.syndrome_circuits = {
            "ZZ": SyndromeCircuit(
                pauli="ZZ",
                name="ZZ_syndrome_extraction",
                circuit=Circuit(
                    name="ZZ_syndrome_extraction",
                    circuit=[
                        [Circuit("Reset_0", channels=a_channels)],
                        [Circuit("H", channels=a_channels)],
                        [],
                        [],
                        [],
                        [],
                        [],
                        [],
                        [Circuit("CZ", channels=[a_channels[0], d_channels[0]])],
                        [Circuit("CZ", channels=[a_channels[0], d_channels[1]])],
                        [Circuit("H", channels=a_channels)],
                        [Circuit("Measurement", channels=a_channels + c_channels)],
                    ],
                ),
            ),
            "X"
            * 6: SyndromeCircuit(
                pauli="X" * 6,
                name="XXXXXX_syndrome_extraction",
                circuit=Circuit(
                    name="XXXXXX_syndrome_extraction",
                    circuit=[
                        [Circuit("Reset_0", channels=a_channels)],
                        [Circuit("H", channels=a_channels)],
                        [Circuit("CX", channels=[a_channels[0], d_channels[0]])],
                        [Circuit("CX", channels=[a_channels[0], d_channels[1]])],
                        [Circuit("CX", channels=[a_channels[0], d_channels[2]])],
                        [Circuit("CX", channels=[a_channels[0], d_channels[3]])],
                        [Circuit("CX", channels=[a_channels[0], d_channels[4]])],
                        [Circuit("CX", channels=[a_channels[0], d_channels[5]])],
                        [],
                        [],
                        [Circuit("H", channels=a_channels)],
                        [Circuit("Measurement", channels=a_channels + c_channels)],
                    ],
                ),
            ),
        }

        # Define the stabilizer to circuit mapping
        self.stabilizer_to_circuit = {
            stab.uuid: self.syndrome_circuits[stab.pauli].uuid
            for stab in self.stabilizers
        }

        # Define the Shor code
        self.position = (2,)
        self.shor_code = ShorCode.create(
            lattice=Lattice.linear(),
            unique_label="q1",
            position=self.position,
        )

    def test_syndrome_extraction_circuit_generation(self):
        """Test the correct generation of the syndrome extraction circuit"""

        # Stabilizer types to be considered
        stabilizer_paulis = ["X" * 6, "Z" * 2]

        # Check circuits are correct
        for pauli in stabilizer_paulis:
            expected_syndrome_circuit = self.syndrome_circuits[pauli]
            syndrome_circuit = ShorCode.generate_syndrome_extraction_circuits(pauli)
            self.assertEqual(syndrome_circuit, expected_syndrome_circuit)

    def test_shor_code_creation(self):
        """Test the correct creation of the Shor Code datablock"""

        # Define Block manually
        manual_code = ShorCode(
            unique_label="q1",
            stabilizers=self.stabilizers,
            logical_x_operators=[self.logical_x_operator],
            logical_z_operators=[self.logical_z_operator],
            syndrome_circuits=[
                self.syndrome_circuits["X" * 6],
                self.syndrome_circuits["ZZ"],
            ],
            stabilizer_to_circuit=self.stabilizer_to_circuit,
        ).shift(self.position)

        self.assertEqual(self.shor_code, manual_code)


if __name__ == "__main__":
    unittest.main()
