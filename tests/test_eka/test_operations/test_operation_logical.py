"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

import unittest
from loom.eka.utilities import SingleQubitPauliEigenstate, loads, dumps
from loom.eka.operations import (
    Operation,
    Reset,
    CNOT,
    Hadamard,
    Phase,
    PhaseInverse,
    X,
    Y,
    Z,
    T,
)


class TestLogicalOperation(unittest.TestCase):
    """Test cases for logical operations."""

    def test_creation(self):
        """Test the creation of logical operations."""
        # Test the creation of Reset
        reset = Reset(
            target_qubit="q",
            state=SingleQubitPauliEigenstate.ZERO,
        )
        self.assertEqual(reset.target_qubit, "q")
        self.assertEqual(reset.state, SingleQubitPauliEigenstate.ZERO)
        self.assertEqual(reset.__class__.__name__, "Reset")
        # Test the loads/dumps both using the right class and the abstract base class
        self.assertEqual(reset, loads(Reset, dumps(reset)))
        self.assertEqual(reset, loads(Operation, dumps(reset)))

        # Test the creation of CNOT
        cnot = CNOT(
            target_qubit="t",
            control_qubit="c",
        )
        self.assertEqual(cnot.target_qubit, "t")
        self.assertEqual(cnot.control_qubit, "c")
        self.assertEqual(cnot.__class__.__name__, "CNOT")
        # Test the loads/dumps both using the right class and the abstract base class
        self.assertEqual(cnot, loads(CNOT, dumps(cnot)))
        self.assertEqual(cnot, loads(Operation, dumps(cnot)))

        # Test the creation of Hadamard
        hadamard = Hadamard(target_qubit="q")
        self.assertEqual(hadamard.target_qubit, "q")
        self.assertEqual(hadamard.__class__.__name__, "Hadamard")
        # Test the loads/dumps both using the right class and the abstract base class
        self.assertEqual(hadamard, loads(Hadamard, dumps(hadamard)))
        self.assertEqual(hadamard, loads(Operation, dumps(hadamard)))

        # Test the creation of Phase
        phase = Phase(target_qubit="q")
        self.assertEqual(phase.target_qubit, "q")
        self.assertEqual(phase.__class__.__name__, "Phase")
        # Test the loads/dumps both using the right class and the abstract base class
        self.assertEqual(phase, loads(Phase, dumps(phase)))
        self.assertEqual(phase, loads(Operation, dumps(phase)))

        # Test the creation of PhaseInverse
        phase_inverse = PhaseInverse(target_qubit="q")
        self.assertEqual(phase_inverse.target_qubit, "q")
        self.assertEqual(phase_inverse.__class__.__name__, "PhaseInverse")
        # Test the loads/dumps both using the right class and the abstract base class
        self.assertEqual(phase_inverse, loads(PhaseInverse, dumps(phase_inverse)))
        self.assertEqual(phase_inverse, loads(Operation, dumps(phase_inverse)))

        # Test the creation of X
        x = X(target_qubit="q")
        self.assertEqual(x.target_qubit, "q")
        self.assertEqual(x.__class__.__name__, "X")
        # Test the loads/dumps both using the right class and the abstract base class
        self.assertEqual(x, loads(X, dumps(x)))
        self.assertEqual(x, loads(Operation, dumps(x)))

        # Test the creation of Y
        y = Y(target_qubit="q")
        self.assertEqual(y.target_qubit, "q")
        self.assertEqual(y.__class__.__name__, "Y")
        # Test the loads/dumps both using the right class and the abstract base class
        self.assertEqual(y, loads(Y, dumps(y)))
        self.assertEqual(y, loads(Operation, dumps(y)))

        # Test the creation of Z
        z = Z(target_qubit="q")
        self.assertEqual(z.target_qubit, "q")
        self.assertEqual(z.__class__.__name__, "Z")
        # Test the loads/dumps both using the right class and the abstract base class
        self.assertEqual(z, loads(Z, dumps(z)))
        self.assertEqual(z, loads(Operation, dumps(z)))

        # Test the creation of T
        t = T(target_qubit="q")
        self.assertEqual(t.target_qubit, "q")
        self.assertEqual(t.__class__.__name__, "T")
        # Test the loads/dumps both using the right class and the abstract base class
        self.assertEqual(t, loads(T, dumps(t)))
        self.assertEqual(t, loads(Operation, dumps(t)))


if __name__ == "__main__":
    unittest.main()
