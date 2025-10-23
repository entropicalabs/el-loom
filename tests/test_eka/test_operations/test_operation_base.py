"""
Copyright 2024 Entropica Labs Pte Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""

import unittest
from loom.eka.operations import (
    Operation,
    MeasureStabilizerSyndrome,
    MeasureObservable,
)
from loom.eka import Stabilizer, PauliOperator
from loom.eka.utilities import loads, dumps


class TestBaseOperation(unittest.TestCase):
    """Unit tests for base operations."""

    def test_creation(self):
        """Test the creation of base operations."""
        # Test the creation of MeasureStabilizerSyndrome
        stab = Stabilizer(pauli="XX", data_qubits=((0, 0), (0, 1)))
        read_stab = MeasureStabilizerSyndrome(stabilizer=stab)
        self.assertEqual(read_stab.stabilizer, stab)
        self.assertEqual(read_stab.__class__.__name__, "MeasureStabilizerSyndrome")
        # Test the loads/dumps both using the right class and the abstract base class
        self.assertEqual(read_stab, loads(MeasureStabilizerSyndrome, dumps(read_stab)))
        self.assertEqual(read_stab, loads(Operation, dumps(read_stab)))

        # Test the creation of MeasureObservable
        obs = PauliOperator(pauli="X", data_qubits=((0, 0),))
        read_obs = MeasureObservable(
            observable=obs,
        )
        self.assertEqual(read_obs.observable, obs)
        self.assertEqual(read_obs.__class__.__name__, "MeasureObservable")
        # Test the loads/dumps both using the right class and the abstract base class
        self.assertEqual(read_obs, loads(MeasureObservable, dumps(read_obs)))
        self.assertEqual(read_obs, loads(Operation, dumps(read_obs)))


if __name__ == "__main__":
    unittest.main()
