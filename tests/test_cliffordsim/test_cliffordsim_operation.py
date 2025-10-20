"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

import unittest
import inspect

from loom.cliffordsim.operations.controlled_operation import (
    ControlledOperation,
    has_ccontrol,
)
from loom.cliffordsim.operations.resize_operation import ResizeOperation
from loom.cliffordsim.operations.measurement_operation import (
    MeasurementOperation,
    Measurement,
)
from loom.cliffordsim.operations.gate_operation import GateOperation
from loom.cliffordsim.operations.datamanipulation_operation import (
    DataManipulationOperation,
)
from loom.cliffordsim.operations.classical_operation import (
    ClassicalOperation,
    ClassicalNOT,
    ClassicalTwoBitOperation,
    ClassicalOR,
    ClassicalAND,
)


class TestCliffordSimOperation(unittest.TestCase):
    """Test the logical restrictions in Operations if they exist."""

    def test_input_exclusion_errors(self):
        """Verify that post init restrictions raise the appropriate errors."""
        test_operations = {
            ClassicalNOT: [],
            ControlledOperation: [ClassicalNOT("testreg", 0)],
        }

        for each_operation, args in test_operations.items():
            ### Case 1: Error raised if only register name provided.
            with self.assertRaises(ValueError):
                each_operation(*args, "testreg")

            ### Case 2: No Error if at least bit order or bit ID is provided.
            each_operation(*args, "testreg", bit_order=0)
            each_operation(*args, "testreg", bit_id="bit_1")

            ### Case 3: Error raised if both provided.
            with self.assertRaises(ValueError):
                each_operation(*args, "testreg", bit_order=0, bit_id="bit_1")

    def test_classicaltwobitoperation(self):
        """
        Test the logical restrictions in ClassicalTwoBitOperation and its subclasses.
        """
        ### Case 1: Error Raised if less than 2 input bits provided
        with self.assertRaises(ValueError):
            ClassicalTwoBitOperation("testreg", input_bit_order=[0], write_bit_order=0)
        # Case 1a: If only input was provided
        with self.assertRaises(ValueError):
            ClassicalTwoBitOperation("testreg", input_bit_order=[0, 1])

        ### Case 2: Proper Initialization of 2-bit Operations
        # Case 2a: With Ordering
        ClassicalTwoBitOperation("testreg", input_bit_order=[0, 1], write_bit_order=0)
        # Case 2b: With IDs
        ClassicalTwoBitOperation(
            "testreg", input_bit_ids=["bit_1", "bit_2"], write_bit_id="bit_1"
        )
        # Case 2c: ID and Order mix
        ClassicalTwoBitOperation(
            "testreg", input_bit_ids=["bit_1", "bit_2"], write_bit_order=0
        )
        ClassicalTwoBitOperation(
            "testreg", input_bit_order=[0, 1], write_bit_id="bit_1"
        )
        # Case 2d: Different Registers
        op = ClassicalTwoBitOperation(
            "testreg",
            input_bit_order=[0, 1],
            output_reg_name="testreg_2",
            write_bit_order=0,
        )
        self.assertEqual(op.output_reg_name, "testreg_2")

        ### Case 3: Error Raised if bit order and bit ID is provided
        # Case 3a: For input
        with self.assertRaises(ValueError):
            ClassicalTwoBitOperation(
                "testreg",
                input_bit_order=[0, 1],
                input_bit_ids=["bit_1", "bit_2"],
                write_bit_order=0,
            )
        # Case 3b: For write
        with self.assertRaises(ValueError):
            ClassicalTwoBitOperation(
                "testreg",
                input_bit_order=[0, 1],
                write_bit_order=0,
                write_bit_id="bit_1",
            )

        ### Case 4: If not output register name is provided. The same input register is used.
        op = ClassicalTwoBitOperation(
            "testreg", input_bit_ids=["bit_1", "bit_2"], write_bit_order=0
        )
        self.assertEqual(op.output_reg_name, "testreg")

        ### Case 5: ClassicalOR and ClassicalAND are subclasses of ClassicalTwoBitOperation
        self.assertTrue(issubclass(ClassicalOR, ClassicalTwoBitOperation))
        self.assertTrue(issubclass(ClassicalAND, ClassicalTwoBitOperation))
        self.assertFalse(issubclass(ClassicalNOT, ClassicalTwoBitOperation))

    def test_ccontrol_operations(self):
        """All of these Operations should have the with_ccontrol method. with_ccontrol
        is added using the has_ccontrol decorator.
        """
        self.assertTrue(hasattr(DataManipulationOperation, "with_ccontrol"))
        self.assertTrue(hasattr(ResizeOperation, "with_ccontrol"))
        self.assertTrue(hasattr(MeasurementOperation, "with_ccontrol"))
        self.assertTrue(hasattr(GateOperation, "with_ccontrol"))
        self.assertTrue(hasattr(ClassicalOperation, "with_ccontrol"))

    def test_has_ccontrol_function(self):
        """Test that the input class is returned with a with_ccontrol method and that
        the with_ccontrol method behaves as expected.
        """

        all_modified_operation = [
            DataManipulationOperation,
            ResizeOperation,
            MeasurementOperation,
            GateOperation,
            ClassicalOperation,
        ]
        for each_operation in all_modified_operation:
            mock_obj = unittest.mock.Mock(spec=each_operation)
            method_sig = inspect.signature(has_ccontrol(mock_obj).with_ccontrol)
            input_parameters = list(method_sig.parameters.keys())

            # Check input parameters of with_ccontrol
            self.assertEqual(
                input_parameters, ["self", "reg_name", "bit_order", "bit_id"]
            )

            controlled_op = mock_obj.with_ccontrol(
                mock_obj, reg_name="testreg_name", bit_order=0
            )

            # Check output of with_ccontrol is a ControlledOperation that wraps app_operation
            self.assertTrue(isinstance(controlled_op, ControlledOperation))
            self.assertEqual(controlled_op.app_operation, mock_obj)

    def test_measurement_w_cwrite(self):
        """
        Tests the appropriate Errors are raised when the user doesn't correctly
        specify information about the output bit for a Measurement that writes to a
        classical register."""
        # Case 1: Both bit order and bit ID are specified.
        with self.assertRaises(ValueError):
            Measurement(
                target_qubit=4, reg_name="testreg_name", bit_order=0, bit_id="bit_0"
            )

        # Case 2: If the register name is specified, bit_order or bit_id must be specified
        with self.assertRaises(ValueError):
            Measurement(target_qubit=4, reg_name="testreg_name")

        # Case 3: Proper Initialization
        Measurement(target_qubit=4, reg_name="testreg_name", bit_order=0)
        Measurement(target_qubit=4, reg_name="testreg_name", bit_id="bit_0")


if __name__ == "__main__":
    unittest.main()
