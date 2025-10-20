"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

import unittest
import logging

from loom.eka import (
    Stabilizer,
    PauliOperator,
    Lattice,
    SyndromeCircuit,
    Circuit,
    Channel,
)

from loom_steane_code.code_factory import SteaneCode

# Set up logging
logging.getLogger().setLevel(logging.DEBUG)


# pylint: disable=duplicate-code
class TestSteaneCode(unittest.TestCase):
    """
    Test the functionalities of the SteaneCode class, which is a
    subclass of the Block class.
    """

    def setUp(self):
        """Define the generic properties of the Steane code"""
        # Define the stabilizers
        self.stabilizers = [
            Stabilizer(
                pauli="XXXX",
                data_qubits=[(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)],
                ancilla_qubits=[(0, 0, 1)],
            ),
            Stabilizer(
                pauli="XXXX",
                data_qubits=[(1, 0, 0), (1, 1, 0), (2, 1, 0), (2, 0, 0)],
                ancilla_qubits=[(1, 0, 1)],
            ),
            Stabilizer(
                pauli="XXXX",
                data_qubits=[(2, 1, 0), (1, 2, 0), (0, 1, 0), (1, 1, 0)],
                ancilla_qubits=[(2, 0, 1)],
            ),
            Stabilizer(
                pauli="ZZZZ",
                data_qubits=[(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)],
                ancilla_qubits=[(0, 1, 1)],
            ),
            Stabilizer(
                pauli="ZZZZ",
                data_qubits=[(1, 0, 0), (1, 1, 0), (2, 1, 0), (2, 0, 0)],
                ancilla_qubits=[(0, 2, 1)],
            ),
            Stabilizer(
                pauli="ZZZZ",
                data_qubits=[(2, 1, 0), (1, 2, 0), (0, 1, 0), (1, 1, 0)],
                ancilla_qubits=[(0, 3, 1)],
            ),
        ]

        # Define the logical operators
        self.logical_x_operator = PauliOperator(
            pauli="XXX",
            data_qubits=[
                (0, 0, 0),
                (1, 0, 0),
                (2, 0, 0),
            ],
        )
        self.logical_z_operator = PauliOperator(
            pauli="ZZZ",
            data_qubits=[
                (0, 0, 0),
                (1, 0, 0),
                (2, 0, 0),
            ],
        )

        # Define the syndrome extraction circuits
        d_channels = [Channel(label=f"d{i}") for i in range(4)]
        a_channels = [Channel(label=f"a{i}", type="quantum") for i in range(1)]
        c_channels = [Channel(label=f"c{i}", type="classical") for i in range(1)]

        self.syndrome_circuits = {
            "ZZZZ": SyndromeCircuit(
                pauli="Z" * 4,
                name="ZZZZ_syndrome_extraction",
                circuit=Circuit(
                    name="ZZZZ_syndrome_extraction",
                    circuit=[
                        [Circuit("Reset_0", channels=a_channels)],
                        [Circuit("H", channels=a_channels)],
                        [],
                        [],
                        [],
                        [],
                        [Circuit("CZ", channels=[a_channels[0], d_channels[0]])],
                        [Circuit("CZ", channels=[a_channels[0], d_channels[1]])],
                        [Circuit("CZ", channels=[a_channels[0], d_channels[2]])],
                        [Circuit("CZ", channels=[a_channels[0], d_channels[3]])],
                        [Circuit("H", channels=a_channels)],
                        [Circuit("Measurement", channels=a_channels + c_channels)],
                    ],
                ),
            ),
            "XXXX": SyndromeCircuit(
                pauli="X" * 4,
                name="XXXX_syndrome_extraction",
                circuit=Circuit(
                    name="XXXX_syndrome_extraction",
                    circuit=[
                        [Circuit("Reset_0", channels=a_channels)],
                        [Circuit("H", channels=a_channels)],
                        [Circuit("CX", channels=[a_channels[0], d_channels[0]])],
                        [Circuit("CX", channels=[a_channels[0], d_channels[1]])],
                        [Circuit("CX", channels=[a_channels[0], d_channels[2]])],
                        [Circuit("CX", channels=[a_channels[0], d_channels[3]])],
                        [],
                        [],
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

        # Define the Steane code
        self.position = (2, 1)
        self.steane_code = SteaneCode.create(
            lattice=Lattice.square_2d(),
            unique_label="q1",
            position=self.position,
        )

    def test_syndrome_extraction_circuit_generation(self):
        """Test the correct generation of the syndrome extraction circuit"""

        # Stabilizer types to be considered
        stabilizer_paulis = ["X" * 4, "Z" * 4]

        # Check circuits are correct
        for pauli in stabilizer_paulis:
            expected_syndrome_circuit = self.syndrome_circuits[pauli]
            syndrome_circuit = SteaneCode.generate_syndrome_extraction_circuits(pauli)
            self.assertEqual(syndrome_circuit, expected_syndrome_circuit)

    def test_steane_code_creation(self):
        """Test the correct creation of the Steane Code datablock"""

        # Define Block manually
        manual_code = SteaneCode(
            unique_label="q1",
            stabilizers=self.stabilizers,
            logical_x_operators=[self.logical_x_operator],
            logical_z_operators=[self.logical_z_operator],
            syndrome_circuits=[
                self.syndrome_circuits["XXXX"],
                self.syndrome_circuits["ZZZZ"],
            ],
            stabilizer_to_circuit=self.stabilizer_to_circuit,
        ).shift(self.position)

        self.assertEqual(self.steane_code, manual_code)


if __name__ == "__main__":
    unittest.main()
