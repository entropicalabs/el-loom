"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

import unittest
import random

import pennylane as qml
from catalyst import qjit

from loom.executor import convert_circuit_to_pennylane
from loom.executor.eka_circuit_to_pennylane_converter import (
    MEASUREMENT_OPERATIONS_MAP,
    SINGLE_QUBIT_GATE_OPERATIONS_MAP,
    TWO_QUBIT_GATE_OPERATIONS_MAP,
    CLASSICALLY_CONTROLLED_OPERATIONS_SQ_MAP,
    CLASSICALLY_CONTROLLED_OPERATIONS_TQ_MAP,
    RESET_OPERATIONS_MAP,
)
from loom.eka import Circuit, Channel


class TestEkaCircuitToPennylaneConverter(unittest.TestCase):
    """Test the EKA to PennyLane converter."""

    def setUp(self):
        # Create channels and a mock measureop for the tests
        self.n_qubits = 10
        self.q_channels = [Channel("quantum", f"q{i}") for i in range(self.n_qubits)]
        self.c_channels = [Channel("classical", f"c{i}") for i in range(self.n_qubits)]

    def test_all_operations_are_mapped(self):
        """Test that all operations in the mapping of EKA to PennyLane are mapped to
        operations that can be executed."""
        for is_catalyst in [True]:  # False removed due to issue with Pennylane update
            # A circuit with all single qubit operations
            sq_circuit = Circuit(
                name="Single qubit operations",
                circuit=[
                    Circuit(name=op_name, channels=[random.choice(self.q_channels)])
                    for op_name in SINGLE_QUBIT_GATE_OPERATIONS_MAP.keys()
                ],
            )
            # A circuit with all two qubit operations
            tq_circuit = Circuit(
                name="Two qubit operations",
                circuit=[
                    Circuit(name=op_name, channels=random.sample(self.q_channels, 2))
                    for op_name in TWO_QUBIT_GATE_OPERATIONS_MAP.keys()
                ],
            )
            # A circuit with all measurement operations
            m_circuit = Circuit(
                name="Measurement operations",
                circuit=[
                    Circuit(
                        name=op_name,
                        channels=[
                            random.choice(self.q_channels),
                            random.choice(self.c_channels),
                        ],
                    )
                    for op_name in MEASUREMENT_OPERATIONS_MAP(is_catalyst).keys()
                ],
            )
            r_circuit = Circuit(
                name="Reset operations",
                circuit=[
                    Circuit(
                        name=op_name,
                        channels=[random.choice(self.q_channels)],
                    )
                    for op_name in RESET_OPERATIONS_MAP(is_catalyst).keys()
                ],
            )

            # Full circuit
            full_circuit = Circuit(
                name="Full circuit",
                circuit=[sq_circuit, tq_circuit, m_circuit, r_circuit],
            )

            # Convert the circuit
            pennylane_circuit, qml_register = convert_circuit_to_pennylane(
                full_circuit, is_catalyst
            )

            # Run the circuit and check that it runs without errors
            dev = qml.device("lightning.qubit", wires=len(qml_register), shots=1)
            qjit()(qml.qnode(dev)(pennylane_circuit))()

    def test_simple_measurement(self):
        """Test that converted PennyLane circuits return the expected results for
        simple measurements.
        """

        m_circuit = Circuit(
            name="Measure qubits 0 and 1 after applying X on qubit 1",
            circuit=[
                Circuit("X", channels=[self.q_channels[1]]),
                Circuit("measure", channels=[self.q_channels[0], self.c_channels[0]]),
                Circuit("measure", channels=[self.q_channels[1], self.c_channels[1]]),
            ],
        )
        expected_mres = {"c0": False, "c1": True}

        pennylane_circuit, qml_register = convert_circuit_to_pennylane(m_circuit, True)

        # Run the circuit
        dev = qml.device("lightning.qubit", wires=len(qml_register), shots=1)
        mres = qjit()(qml.qnode(dev)(pennylane_circuit))()

        self.assertEqual(mres, expected_mres)

    def test_x_y_measurement(self):
        """Test that converted PennyLane circuits return the expected results for
        X and Y gate measurements.
        """
        # Measure qubit 0 in X after applying H on it
        # Measure qubit 1 in X after applying X and H on it
        # Measure qubit 2 in Y after applying H and phase on it
        # Measure qubit 3 in Y after applying X, H and phase on it
        m_circuit = Circuit(
            name="Measure qubit 0 in X after applying H on it and measure qubit 1 in Y",
            circuit=
            # Flip qubits 1 and 3
            [Circuit("x", channels=[self.q_channels[i]]) for i in [1, 3]]
            # Apply hadamard on all qubits
            + [Circuit("h", channels=[self.q_channels[i]]) for i in [0, 1, 2, 3]]
            # Apply phase on qubits 2 and 3
            + [Circuit("phase", channels=[self.q_channels[i]]) for i in [2, 3]]
            # Measure qubits 0 and 1 in X and qubits 2 and 3 in Y
            + [
                Circuit("measure_x", channels=[self.q_channels[i], self.c_channels[i]])
                for i in [0, 1]
            ]
            # Measure qubits 2 and 3 in Y
            + [
                Circuit("measure_y", channels=[self.q_channels[i], self.c_channels[i]])
                for i in [2, 3]
            ],
        )
        expected_mres = {"c0": False, "c1": True, "c2": False, "c3": True}

        # Convert the circuit
        pennylane_circuit, qml_register = convert_circuit_to_pennylane(
            m_circuit, is_catalyst=True
        )

        # Run the circuit
        dev = qml.device("lightning.qubit", wires=len(qml_register), shots=1)
        mres = qjit()(qml.qnode(dev)(pennylane_circuit))()

        self.assertEqual(mres, expected_mres)

    def test_classically_controlled_ops_error(self):
        """Test that an error is raised if a classically controlled operation is
        not preceded by a measurement operation on the control channel.
        """
        error_circ = Circuit(
            name="Classically controlled operation without measurement",
            circuit=[
                Circuit(
                    "classically_controlled_x",
                    channels=[self.c_channels[0], self.q_channels[0]],
                )
            ],
        )
        pennylane_circuit, qml_register = convert_circuit_to_pennylane(
            error_circ, is_catalyst=True
        )
        dev = qml.device("lightning.qubit", wires=len(qml_register), shots=1)

        with self.assertRaises(KeyError) as context:
            qjit()(qml.qnode(dev)(pennylane_circuit))()

        self.assertIn(
            "classically_controlled_x operation is classically controlled but the "
            "control channel c0 could not be found in the measurements dictionary.",
            str(context.exception),
        )

    def test_all_classically_controlled_ops(self):
        """Test that classically controlled operations are executed correctly."""

        classically_controlled_circuit = Circuit(
            name="Classically controlled operations",
            circuit=[
                Circuit("measure_x", channels=[self.q_channels[0], self.c_channels[0]]),
            ]
            + [
                Circuit(
                    op_name,
                    channels=[self.c_channels[0], self.q_channels[0]],
                )
                for op_name in CLASSICALLY_CONTROLLED_OPERATIONS_SQ_MAP.keys()
            ]
            + [
                Circuit(
                    op_name,
                    channels=[
                        self.c_channels[0],
                        self.q_channels[0],
                        self.q_channels[1],
                    ],
                )
                for op_name in CLASSICALLY_CONTROLLED_OPERATIONS_TQ_MAP.keys()
            ],
        )

        # - 1 Convert and run the circuit with catalyst
        pennylane_circuit, qml_register = convert_circuit_to_pennylane(
            classically_controlled_circuit, is_catalyst=True
        )
        dev = qml.device("lightning.qubit", wires=len(qml_register), shots=1)
        _ = qjit()(qml.qnode(dev)(pennylane_circuit))()

        # - 2 Convert and run the circuit without catalyst
        pennylane_circuit, qml_register = convert_circuit_to_pennylane(
            classically_controlled_circuit, is_catalyst=False
        )

        @qml.qnode(qml.device("default.qubit", wires=len(qml_register), shots=1))
        def circ_func():
            pennylane_circuit()
            # Return the measurement result of the first qubit for the circuit to compile
            return qml.sample(qml.measure(0))

        circ_func()

    def test_classically_controlled_op_example(self):
        """Test a simple example of a classically controlled operation."""

        # Measure qubit 0 in X to get a random bit
        # Apply X on qubit 1 if the bit is 1
        # Measure qubit 1 in X
        # The two measurements should be equal
        m_circuit = Circuit(
            name="Classically controlled operation example",
            circuit=[
                Circuit("measure_x", channels=[self.q_channels[0], self.c_channels[0]]),
                Circuit(
                    "classically_controlled_x",
                    channels=[self.c_channels[0], self.q_channels[1]],
                ),
                Circuit("measure", channels=[self.q_channels[1], self.c_channels[1]]),
            ],
        )

        # - 1 Convert and run the circuit with catalyst
        pennylane_circuit, qml_register = convert_circuit_to_pennylane(
            m_circuit, is_catalyst=True
        )
        dev = qml.device("lightning.qubit", wires=len(qml_register), shots=1)
        mres = qjit()(qml.qnode(dev)(pennylane_circuit))()

        # Check that the measurements are equal
        self.assertEqual(mres["c0"], mres["c1"])

        # - 2 Convert and run the circuit without catalyst
        pennylane_circuit, qml_register = convert_circuit_to_pennylane(
            m_circuit, is_catalyst=False
        )

        @qml.qnode(qml.device("default.qubit", wires=len(qml_register), shots=1))
        def circ_func():
            mres = pennylane_circuit()
            # Return the measurement results of c0 and c1 for the circuit to compile
            return [qml.sample(mres["c0"]), qml.sample(mres["c1"])]

        c_0, c_1 = circ_func()
        self.assertEqual(c_0, c_1)


if __name__ == "__main__":
    unittest.main()
