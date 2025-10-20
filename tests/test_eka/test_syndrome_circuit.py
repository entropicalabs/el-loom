"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

import unittest
import logging

from loom.eka import SyndromeCircuit, Circuit, Channel
from loom.eka.utilities import loads, dumps


class TestSyndromeCircuit(unittest.TestCase):
    """
    Test for the SyndromeCircuit class.
    """

    def setUp(self):
        # Set to ERROR to avoid cluttering the test output
        logging.getLogger().setLevel(logging.ERROR)
        self.formatter = lambda lvl, msg: f"{lvl}:loom.eka.syndrome_circuit:{msg}"

    def test_logging__eq__(self):
        """
        Test that the logging in __eq__ method of SyndromeCircuit works as expected.

        Note: There is no way to test logging formatting, so we only check that the
        logging is called
        """
        with self.assertLogs("loom.eka.syndrome_circuit", level="DEBUG") as cm:
            self.assertNotEqual(
                SyndromeCircuit(pauli="XZ"), SyndromeCircuit(pauli="XY")
            )
        self.assertEqual(
            [
                self.formatter(
                    "INFO", "The two circuits measure different Pauli strings."
                ),
                self.formatter("DEBUG", "XZ != XY\n"),
            ],
            cm.output,
        )

        with self.assertLogs("loom.eka.syndrome_circuit", level="DEBUG") as cm:
            self.assertNotEqual(
                SyndromeCircuit(name="test1", pauli="XYZ"),
                SyndromeCircuit(name="test2", pauli="XYZ"),
            )

        self.assertEqual(
            [
                self.formatter("INFO", "The two circuits have different names."),
                self.formatter("DEBUG", "test1 != test2\n"),
            ],
            cm.output,
        )

        with self.assertLogs("loom.eka.syndrome_circuit", level="DEBUG") as cm:
            self.assertNotEqual(
                SyndromeCircuit(
                    name="test1",
                    pauli="XYZ",
                    circuit=Circuit("h", channels=Channel(type="quantum", label="d0")),
                ),
                SyndromeCircuit(
                    name="test1",
                    pauli="XYZ",
                    circuit=Circuit("x", channels=Channel(type="quantum", label="d0")),
                ),
            )
        # There are additional messages from Circuit, which we do not check here
        self.assertEqual(
            [
                self.formatter(
                    "INFO", "The two circuits have different circuit instructions."
                )
            ],
            cm.output,
        )

    def test_load_dump(self):
        """
        Test that the load and dump functions work correctly.
        """
        syndrome_circuit = SyndromeCircuit(name="test", pauli="XYZ")
        loaded_syndrome_circuit = loads(SyndromeCircuit, dumps(syndrome_circuit))
        self.assertEqual(loaded_syndrome_circuit, syndrome_circuit)

    def test_invalid_attribute(self):
        """
        Test that an error is raised when the SyndromeCircuit attribute is invalid.
        """
        # Test that an error is raised when the name field is empty
        with self.assertRaises(ValueError) as cm:
            SyndromeCircuit(name="", pauli="XXX")
        self.assertIn("String should have at least 1 character", str(cm.exception))

        # Test that an error is raised when the Pauli field is empty
        with self.assertRaises(ValueError) as cm:
            SyndromeCircuit(name="syndrome_circuit", pauli="")
        self.assertIn("String should have at least 1 character", str(cm.exception))

        # Test that an error is raised when the Pauli operator is invalid.
        with self.assertRaises(ValueError) as cm:
            SyndromeCircuit(name="test", pauli="XYZW")
        self.assertIn(
            "Invalid pauli: W. Must be one of ['X', 'Y', 'Z'].", str(cm.exception)
        )

    def test_syndrome_circuit(self):
        """
        Test that checks that the syndrome circuit has been properly created.
        The final syndrome circuit, Circuit object, should have the following structure:
        - Hadamard gate on the ancilla qubit
        - CU gate between the ancilla qubit and the data qubit, where U depends
          on the pauli being measured.
        - Hadamard gate on the ancilla qubit
        - Measurement gate on the ancilla qubit
        The gates should also be ticked appropriately.
        i.e. the Hadamard gate should be ticked at 0, the CNOT gates at 1 to n+1, where
        n is the number of CNOT gates, and the Measurement gate at the last tick, n+2.
        """
        syndrome_circuit = SyndromeCircuit(name="test", pauli="XYZ")

        self.assertEqual(len(syndrome_circuit.circuit.circuit), 7)
        self.assertEqual(syndrome_circuit.name, "test")

        sample_channels = [
            Channel(type="quantum", label="d0"),
            Channel(type="quantum", label="d1"),
            Channel(type="quantum", label="d2"),
            Channel(type="quantum", label="a0"),
            Channel(type="classical", label="c0"),
        ]
        reset0 = Circuit("Reset_0", channels=[sample_channels[3]])
        hadamard = Circuit("h", channels=[sample_channels[3]])
        cx = Circuit("cx", channels=[sample_channels[3], sample_channels[0]])
        cy = Circuit("cy", channels=[sample_channels[3], sample_channels[1]])
        cz = Circuit("cz", channels=[sample_channels[3], sample_channels[2]])
        hadamard2 = Circuit("h", channels=[sample_channels[3]])
        measurement = Circuit(
            "measurement", channels=[sample_channels[3], sample_channels[4]]
        )
        sample_circuit = Circuit(
            name="test",
            circuit=[
                [reset0],
                [hadamard],
                [cx],
                [cy],
                [cz],
                [hadamard2],
                [measurement],
            ],
            channels=sample_channels,
        )
        remapped_circuit = sample_circuit.clone(syndrome_circuit.circuit.channels)

        self.assertEqual(syndrome_circuit.circuit.circuit, remapped_circuit.circuit)


if __name__ == "__main__":
    unittest.main()
