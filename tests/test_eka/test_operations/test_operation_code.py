"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

import unittest


from loom.eka.utilities import (
    SingleQubitPauliEigenstate,
    Direction,
    Orientation,
    ResourceState,
    loads,
    dumps,
)
from loom.eka.operations import (
    Grow,
    Shrink,
    Merge,
    Split,
    MeasureLogicalX,
    MeasureLogicalY,
    MeasureLogicalZ,
    ResetAllDataQubits,
    ResetAllAncillaQubits,
    Operation,
    StateInjection,
)


# pylint: disable=protected-access
class TestCodeOperation(unittest.TestCase):
    """
    Test the creation of the logical operator measurement and qubit reset
    operations.
    """

    def test_measure_logical_x(self):
        """
        Test the creation of a logical X measurement operation"""

        meas_log_x = MeasureLogicalX(input_block_name="q1")
        self.assertEqual(meas_log_x.input_block_name, "q1")
        self.assertEqual(meas_log_x.logical_qubit, 0)
        self.assertEqual(
            meas_log_x.__class__.__name__,
            "MeasureLogicalX",
        )
        self.assertEqual(meas_log_x._inputs, ("q1",))
        self.assertEqual(meas_log_x._outputs, ("q1",))
        # Test the loads/dumps both using the right class and the abstract base class
        self.assertEqual(meas_log_x, loads(MeasureLogicalX, dumps(meas_log_x)))
        self.assertEqual(meas_log_x, loads(Operation, dumps(meas_log_x)))

    def test_measure_logical_y(self):
        """
        Test the creation of a logical Y measurement operation
        """

        meas_log_y = MeasureLogicalY(input_block_name="q1")
        self.assertEqual(meas_log_y.input_block_name, "q1")
        self.assertEqual(meas_log_y.logical_qubit, 0)
        self.assertEqual(
            meas_log_y.__class__.__name__,
            "MeasureLogicalY",
        )
        self.assertEqual(meas_log_y._inputs, ("q1",))
        self.assertEqual(meas_log_y._outputs, ("q1",))
        # Test the loads/dumps both using the right class and the abstract base class
        self.assertEqual(meas_log_y, loads(MeasureLogicalY, dumps(meas_log_y)))
        self.assertEqual(meas_log_y, loads(Operation, dumps(meas_log_y)))

    def test_measure_logical_z(self):
        """
        Test the creation of a logical Z measurement operation"""

        meas_log_z = MeasureLogicalZ(input_block_name="q1")
        self.assertEqual(meas_log_z.input_block_name, "q1")
        self.assertEqual(meas_log_z.logical_qubit, 0)
        self.assertEqual(
            meas_log_z.__class__.__name__,
            "MeasureLogicalZ",
        )
        self.assertEqual(meas_log_z._inputs, ("q1",))
        self.assertEqual(meas_log_z._outputs, ("q1",))
        # Test the loads/dumps both using the right class and the abstract base class
        self.assertEqual(meas_log_z, loads(MeasureLogicalZ, dumps(meas_log_z)))
        self.assertEqual(meas_log_z, loads(Operation, dumps(meas_log_z)))

    def test_reset_all_data_qubits(self):
        """
        Test the creation of a logical reset operation"""

        # Test the creation of a logical reset operation
        for state in SingleQubitPauliEigenstate:
            logical_reset = ResetAllDataQubits(input_block_name="q1", state=state)
            self.assertEqual(logical_reset.input_block_name, "q1")
            self.assertEqual(logical_reset.__class__.__name__, "ResetAllDataQubits")
            self.assertEqual(logical_reset._inputs, ("q1",))
            self.assertEqual(logical_reset._outputs, ("q1",))
            self.assertEqual(logical_reset.state, state)
            # Test the loads/dumps both using the right class and the abstract base class
            self.assertEqual(
                logical_reset, loads(ResetAllDataQubits, dumps(logical_reset))
            )
            self.assertEqual(logical_reset, loads(Operation, dumps(logical_reset)))

    def test_reset_all_ancilla_qubits(self):
        """
        Test the creation of an ancilla reset operation"""
        # Test the ancilla reset operation
        for state in SingleQubitPauliEigenstate:
            ancilla_reset = ResetAllAncillaQubits(input_block_name="q1", state=state)
            self.assertEqual(ancilla_reset.input_block_name, "q1")
            self.assertEqual(ancilla_reset.__class__.__name__, "ResetAllAncillaQubits")
            self.assertEqual(ancilla_reset._inputs, ("q1",))
            self.assertEqual(ancilla_reset._outputs, ("q1",))
            self.assertEqual(ancilla_reset.state, state)
            # Test the loads/dumps both using the right class and the abstract base class
            self.assertEqual(
                ancilla_reset, loads(ResetAllAncillaQubits, dumps(ancilla_reset))
            )
            self.assertEqual(ancilla_reset, loads(Operation, dumps(ancilla_reset)))

    def test_grow(self):
        """Test the creation of a Grow operation"""
        grow = Grow(input_block_name="q1", direction=Direction.TOP, length=1)
        self.assertEqual(grow.input_block_name, "q1")
        self.assertEqual(grow.direction, Direction.TOP)
        self.assertEqual(grow.length, 1)
        self.assertEqual(grow.__class__.__name__, "Grow")
        self.assertEqual(grow._inputs, ("q1",))
        self.assertEqual(grow._outputs, ("q1",))
        # Test the loads/dumps both using the right class and the abstract base class
        self.assertEqual(grow, loads(Grow, dumps(grow)))
        self.assertEqual(grow, loads(Operation, dumps(grow)))

        # Test invalid length input
        err_msg_length = "length has to be larger than 0."

        invalid_lengths = [-1, 0]
        for invalid_length in invalid_lengths:
            with self.assertRaises(ValueError) as cm:
                grow = Grow(
                    input_block_name="q1",
                    direction=Direction.TOP,
                    length=invalid_length,
                )
            self.assertIn(err_msg_length, str(cm.exception))

    def test_shrink(self):
        """Test the creation of a Shrink operation"""

        shrink = Shrink(
            input_block_name="q1",
            direction=Direction.TOP,
            length=1,
        )
        self.assertEqual(shrink.input_block_name, "q1")
        self.assertEqual(shrink.direction, Direction.TOP)
        self.assertEqual(shrink.length, 1)
        self.assertEqual(shrink.__class__.__name__, "Shrink")
        self.assertEqual(shrink._inputs, ("q1",))
        self.assertEqual(shrink._outputs, ("q1",))
        # Test the loads/dumps both using the right class and the abstract base class
        self.assertEqual(shrink, loads(Shrink, dumps(shrink)))
        self.assertEqual(shrink, loads(Operation, dumps(shrink)))

        # Test invalid length input
        err_msg_length = "length has to be larger than 0."

        invalid_lengths = [-1, 0]
        for invalid_length in invalid_lengths:
            with self.assertRaises(ValueError) as cm:
                shrink = Shrink(
                    input_block_name="q1",
                    direction=Direction.TOP,
                    length=invalid_length,
                )
            self.assertIn(err_msg_length, str(cm.exception))

    def test_merge(self):
        """Test the creation of a Merge operation"""

        merge = Merge(
            input_blocks_name=["q1", "q2"],
            output_block_name="q3",
            orientation=Orientation.HORIZONTAL,
        )
        self.assertEqual(merge.input_blocks_name, ("q1", "q2"))
        self.assertEqual(merge.output_block_name, "q3")
        self.assertEqual(merge.orientation, Orientation.HORIZONTAL)
        self.assertEqual(merge.__class__.__name__, "Merge")
        self.assertEqual(merge._inputs, ("q1", "q2"))
        self.assertEqual(merge._outputs, ("q3",))
        # Test the loads/dumps both using the right class and the abstract base class
        self.assertEqual(merge, loads(Merge, dumps(merge)))
        self.assertEqual(merge, loads(Operation, dumps(merge)))

    def test_split(self):
        """Test the creation of a Split operation"""

        split = Split(
            input_block_name="q1",
            output_blocks_name=["q2", "q3"],
            orientation=Orientation.VERTICAL,
            split_position=3,
        )
        self.assertEqual(split.input_block_name, "q1")
        self.assertEqual(split.output_blocks_name, ("q2", "q3"))
        self.assertEqual(split.orientation, Orientation.VERTICAL)
        self.assertEqual(split.split_position, 3)
        self.assertEqual(split.__class__.__name__, "Split")
        self.assertEqual(split._inputs, ("q1",))
        self.assertEqual(split._outputs, ("q2", "q3"))
        # Test the loads/dumps both using the right class and the abstract base class
        self.assertEqual(split, loads(Split, dumps(split)))
        self.assertEqual(split, loads(Operation, dumps(split)))

        # Test invalid creation of split
        err_msg_split_position = "split_position has to be larger than 0."

        with self.assertRaises(ValueError) as cm:
            Split(
                input_block_name="q1",
                output_blocks_name=["q2", "q3"],
                orientation=Orientation.VERTICAL,
                split_position=-1,
            )
        self.assertIn(err_msg_split_position, str(cm.exception))

    def test_state_injection(self):
        """Test the creation of a state injection operation"""
        # Test the creation of a state injection operation
        for state in ResourceState:
            state_injection = StateInjection(
                input_block_name="q1", resource_state=state
            )
            self.assertEqual(state_injection.input_block_name, "q1")
            self.assertEqual(state_injection.__class__.__name__, "StateInjection")
            self.assertEqual(state_injection._inputs, ("q1",))
            self.assertEqual(state_injection._outputs, ("q1",))
            self.assertEqual(state_injection.resource_state, state)
            # Test the loads/dumps using the right class and the abstract base class
            self.assertEqual(
                state_injection, loads(StateInjection, dumps(state_injection))
            )
            self.assertEqual(state_injection, loads(Operation, dumps(state_injection)))


if __name__ == "__main__":
    unittest.main()
