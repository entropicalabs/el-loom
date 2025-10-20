"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

# pylint: disable=too-many-lines
import unittest
from pydantic import ValidationError

from loom.eka import (
    Block,
    Stabilizer,
    PauliOperator,
)
from loom.eka.utilities import loads, dumps, uuid_error


class TestBlock(unittest.TestCase):  # pylint: disable=too-many-public-methods
    """
    Test for the Block class.
    """

    def setUp(self):
        # Manually construct a simple distance-3 rotated surface code block
        # Note 1: Since uuids are not explicitly assigned to code `Stabilizer`s,
        # they are automatically generated.
        # Note 2: Since the fields `Block.syndrome_circuits` and
        # `Block.stabilizer_to_circuit` are not explicitly assigned, they are
        # automatically populated by validation checks in the `Block` class.
        # pylint: disable=duplicate-code
        self.rotated_surface_code = Block(
            stabilizers=(
                Stabilizer(
                    pauli="ZZZZ",
                    data_qubits=((1, 0, 0), (1, 1, 0), (0, 0, 0), (0, 1, 0)),
                    ancilla_qubits=((1, 1, 1),),
                ),
                Stabilizer(
                    pauli="ZZZZ",
                    data_qubits=((2, 1, 0), (2, 2, 0), (1, 1, 0), (1, 2, 0)),
                    ancilla_qubits=((2, 2, 1),),
                ),
                Stabilizer(
                    pauli="XXXX",
                    data_qubits=((1, 1, 0), (0, 1, 0), (1, 2, 0), (0, 2, 0)),
                    ancilla_qubits=((1, 2, 1),),
                ),
                Stabilizer(
                    pauli="XXXX",
                    data_qubits=((2, 0, 0), (1, 0, 0), (2, 1, 0), (1, 1, 0)),
                    ancilla_qubits=((2, 1, 1),),
                ),
                Stabilizer(
                    pauli="XX",
                    data_qubits=((0, 0, 0), (0, 1, 0)),
                    ancilla_qubits=((0, 1, 1),),
                ),
                Stabilizer(
                    pauli="XX",
                    data_qubits=((2, 1, 0), (2, 2, 0)),
                    ancilla_qubits=((3, 2, 1),),
                ),
                Stabilizer(
                    pauli="ZZ",
                    data_qubits=((2, 0, 0), (1, 0, 0)),
                    ancilla_qubits=((2, 0, 1),),
                ),
                Stabilizer(
                    pauli="ZZ",
                    data_qubits=((1, 2, 0), (0, 2, 0)),
                    ancilla_qubits=((1, 3, 1),),
                ),
            ),
            logical_x_operators=[
                PauliOperator(
                    pauli="XXX", data_qubits=((0, 0, 0), (1, 0, 0), (2, 0, 0))
                )
            ],
            logical_z_operators=[
                PauliOperator(
                    pauli="ZZZ", data_qubits=((0, 0, 0), (0, 1, 0), (0, 2, 0))
                )
            ],
            unique_label="q1",
        )

    def test_creation_valid_block(self):
        """
        Test the creation of a Block object.
        """
        stabilizers = [
            Stabilizer(
                pauli="XX",
                data_qubits=(
                    (0, 0),
                    (1, 0),
                ),
                ancilla_qubits=((3, 1),),
            ),
            Stabilizer(
                pauli="XX",
                data_qubits=(
                    (1, 0),
                    (2, 0),
                ),
                ancilla_qubits=((4, 1),),
            ),
        ]
        logical_x = PauliOperator(pauli="XXX", data_qubits=((0, 0), (1, 0), (2, 0)))
        logical_z = PauliOperator(pauli="ZZZ", data_qubits=((0, 0), (1, 0), (2, 0)))

        block = Block(
            stabilizers=stabilizers,
            logical_x_operators=[logical_x],
            logical_z_operators=[logical_z],
            unique_label="q1",
        )

        self.assertEqual(block.unique_label, "q1")
        self.assertEqual(block.stabilizers, tuple(stabilizers))
        self.assertEqual(block.logical_x_operators, (logical_x,))
        self.assertEqual(block.logical_z_operators, (logical_z,))
        self.assertEqual(block.skip_validation, False)

    # Test of the "before" validators
    # NOTE they are tested in the order they are executed with pydantic (bottom to top
    # in the code)
    def test_validate_non_empty_stabilizers(self):
        """
        Test that an error is raised if the stabilizers list is empty. This is tested
        in _assign_types() before the Block is created.
        """
        with self.assertRaises(ValueError) as cm:
            _ = Block(
                unique_label=self.rotated_surface_code.unique_label,
                stabilizers=[],
                logical_x_operators=self.rotated_surface_code.logical_x_operators,
                logical_z_operators=self.rotated_surface_code.logical_z_operators,
            )
        self.assertIn("List cannot be empty.", str(cm.exception))

    def test_validate_non_empty_log_operators(self):
        """
        Test that an error is raised if the logical_x_operators list is empty. This
        is tested in _assign_types() before the Block is created.
        """
        with self.assertRaises(ValueError) as cm:
            _ = Block(
                unique_label=self.rotated_surface_code.unique_label,
                stabilizers=self.rotated_surface_code.stabilizers,
                logical_x_operators=[],
                logical_z_operators=[],
            )
        self.assertIn("List cannot be empty.", str(cm.exception))

    def test_validate_distinct_stabilizers(self):
        """
        Test that an error is raised if the stabilizers are not distinct. This is
        tested in _validate_distinct_stabilizers() before the Block is created.
        """
        # Ensure that the number of stabilizers is still right but one is repeated
        with self.assertRaises(ValueError) as cm:
            _ = Block(
                unique_label=self.rotated_surface_code.unique_label,
                stabilizers=self.rotated_surface_code.stabilizers[:-1]
                + self.rotated_surface_code.stabilizers[0:1],
                logical_x_operators=self.rotated_surface_code.logical_x_operators,
                logical_z_operators=self.rotated_surface_code.logical_z_operators,
            )
        err_msg = "Value error, Stabilizers must be distinct."
        self.assertIn(err_msg, str(cm.exception))

    def test_validate_distinct_logical_x_operators(self):
        """
        Test that an error is raised if the logical_x_operators are not distinct.
        This is tested in _validate_distinct_logical_x_operators() before the Block is
        created.
        """
        # Add an additional valid stabilizer first to make sure dimensions are correct
        additional_stab = Stabilizer(pauli="Z", data_qubits=((10, 10, 0),))
        # Add an additional valid logical Z first to make sure dimensions are correct
        additional_log_z_operator = PauliOperator(pauli="Z", data_qubits=((10, 10, 0),))
        # Test repeated logical X operators
        with self.assertRaises(ValueError) as cm:
            _ = Block(
                unique_label=self.rotated_surface_code.unique_label,
                stabilizers=self.rotated_surface_code.stabilizers + (additional_stab,),
                logical_x_operators=self.rotated_surface_code.logical_x_operators
                + self.rotated_surface_code.logical_x_operators,  # Repeated logical X operator
                logical_z_operators=self.rotated_surface_code.logical_z_operators
                + (additional_log_z_operator,),
            )
        err_msg = "Logical X operators must be distinct."
        self.assertIn(err_msg, str(cm.exception))

    def test_validate_distinct_logical_z_operators(self):
        """
        Test that an error is raised if the logical_z_operators are not distinct.
        This is tested in _validate_distinct_logical_z_operators() before the Block is
        created.
        """
        # Add an additional stabilizer first to make sure dimensions are correct
        additional_stab = Stabilizer(pauli="Z", data_qubits=((10, 10, 0),))
        # Add an additional logical X first to make sure dimensions are correct
        additional_log_x_operator = PauliOperator(pauli="Z", data_qubits=((10, 10, 0),))
        # Test repeated logical Z operators
        with self.assertRaises(ValueError) as cm:
            _ = Block(
                unique_label=self.rotated_surface_code.unique_label,
                stabilizers=self.rotated_surface_code.stabilizers + (additional_stab,),
                logical_x_operators=self.rotated_surface_code.logical_x_operators
                + (additional_log_x_operator,),
                logical_z_operators=self.rotated_surface_code.logical_z_operators
                + self.rotated_surface_code.logical_z_operators,  # Repeated logical Z operator
            )
        err_msg = "Logical Z operators must be distinct."
        self.assertIn(err_msg, str(cm.exception))

    def test_validate_number_of_logical_operators(self):
        """
        Test that an error is raised if the number of logical X operators is
        not equal to the number of logical Z operators. This is tested in
        _validate_number_of_logical_operators() before the Block is created.
        """
        # Add an additional stabilizer first to make sure dimensions are correct
        additional_stab = Stabilizer(pauli="Z", data_qubits=((10, 10, 0),))
        additional_z_operator = PauliOperator(
            pauli="ZZZ", data_qubits=((1, 0, 0), (1, 1, 0), (1, 2, 0))
        )
        # Test for different number of logical X and Z operators
        with self.assertRaises(ValueError) as cm:
            _ = Block(
                unique_label=self.rotated_surface_code.unique_label,
                stabilizers=self.rotated_surface_code.stabilizers + (additional_stab,),
                logical_x_operators=self.rotated_surface_code.logical_x_operators,
                logical_z_operators=self.rotated_surface_code.logical_z_operators
                + (additional_z_operator,),
            )
        err_msg = (
            "Value error, The number of logical X operators must be equal to "
            "the number of logical Z operators."
        )
        self.assertIn(err_msg, str(cm.exception))

    # Test of the "after" validators
    def test_validate_coordinate_dimensions(self):
        """
        Test that an error is raised if the coordinates of all qubits don't have the
        same dimensions. This is tested in _validate_coordinate_dimensions() after the
        Block is created.
        """
        stabilizers = [
            Stabilizer(
                pauli="XX",
                data_qubits=((0, 0), (1, 0)),
                ancilla_qubits=((3, 0, 1),),  # Different dimensions
            ),
            Stabilizer(
                pauli="XX",
                data_qubits=((1, 0), (2, 0)),
                ancilla_qubits=((4, 1),),
            ),
        ]
        logical_x = PauliOperator(pauli="XXX", data_qubits=((0, 0), (1, 0), (2, 0)))
        logical_z = PauliOperator(pauli="ZZZ", data_qubits=((0, 0), (1, 0), (2, 0)))
        with self.assertRaises(ValueError) as cm:
            _ = Block(
                stabilizers=stabilizers,
                logical_x_operators=[logical_x],
                logical_z_operators=[logical_z],
                unique_label="q1",
            )
        err_msg = "All qubits coordinates must have the same dimension."
        self.assertIn(err_msg, str(cm.exception))

    def test_validate_qubits_included(self):
        """
        Test that an error is raised if the qubits in the logical operators are not
        included in the stabilizers. This is tested in _validate_qubits_included()
        before the Block is created.
        """
        # Replace the usual logical operators by ones outside of the block
        error_log_x = PauliOperator("X", [(10, 10, 0)])
        error_log_z = PauliOperator("Z", [(10, 10, 0)])
        with self.assertRaises(ValueError) as cm:
            _ = Block(
                unique_label=self.rotated_surface_code.unique_label,
                stabilizers=self.rotated_surface_code.stabilizers,
                logical_x_operators=[error_log_x],
                logical_z_operators=[error_log_z],
            )
        err_msg = (
            "Qubits {(10, 10, 0)} are not included in the stabilizers but are used "
            "in the logical operators"
        )
        self.assertIn(err_msg, str(cm.exception))

    def test_validate_dimensional_compatibility(self):
        """
        Test that an error is raised if the number of qubits and stabilizers in the
        Block is not compatible with the number of logical qubits. This is tested in
        _validate_dimensional_compatibility() after the Block is created.
        """
        # Add an additional stabilizer to trigger dimensional compatibility error
        # There are now 9+2 data qubits, 8+1 stabilizers, 1 logical X and 1 logical Z
        additional_stab = Stabilizer(
            pauli="ZZ",
            data_qubits=((10, 10, 0), (10, 11, 0)),
        )
        with self.assertRaises(ValueError) as cm:
            _ = Block(
                unique_label=self.rotated_surface_code.unique_label,
                stabilizers=self.rotated_surface_code.stabilizers + (additional_stab,),
                logical_x_operators=self.rotated_surface_code.logical_x_operators,
                logical_z_operators=self.rotated_surface_code.logical_z_operators,
            )
        err_msg = (
            "The number of qubits and independent stabilizers in the Block is "
            "not compatible with the number of logical qubits."
        )
        self.assertIn(err_msg, str(cm.exception))

    def test_validate_commutation_stabilizers(self):
        """
        Test that an error is raised if the stabilizers do not commute with each
        other. This is tested in _validate_commutation_stabilizers() after the Block is
        created.
        """
        # Add a non-commuting stabilizer
        non_commuting_stab = Stabilizer(pauli="Z", data_qubits=((0, 0, 0),))
        # Make sure that the total number of stabilizers is still correct
        with self.assertRaises(ValueError) as cm:
            _ = Block(
                unique_label=self.rotated_surface_code.unique_label,
                stabilizers=self.rotated_surface_code.stabilizers[:-1]
                + (non_commuting_stab,),
                logical_x_operators=self.rotated_surface_code.logical_x_operators,
                logical_z_operators=self.rotated_surface_code.logical_z_operators,
            )
        err_msg = "Stabilizers must commute with each other"
        self.assertIn(err_msg, str(cm.exception))

    def test_validate_commutation_logical_operators(self):
        """
        Test that an error is raised if the logical operators do not commute with
        each other for both X and Z operators. This is tested in
        _validate_commutation_logical_operators() after the Block is created.
        """
        # Add an additional stabilizer first to make sure dimensions are correct
        additional_stab = Stabilizer(
            pauli="ZZ",
            data_qubits=((10, 10, 0), (10, 11, 0)),
        )
        # Add an additional logical Z first to make sure dimensions are correct
        additional_log_z_operator = PauliOperator(pauli="Z", data_qubits=((10, 10, 0),))
        # Add a non-commuting logical X operator
        non_commuting_logical_x = PauliOperator(pauli="Z", data_qubits=((0, 0, 0),))
        with self.assertRaises(ValueError) as cm:
            _ = Block(
                unique_label=self.rotated_surface_code.unique_label,
                stabilizers=self.rotated_surface_code.stabilizers + (additional_stab,),
                logical_x_operators=self.rotated_surface_code.logical_x_operators
                + (non_commuting_logical_x,),
                logical_z_operators=self.rotated_surface_code.logical_z_operators
                + (additional_log_z_operator,),
            )
        err_msg = "Logical X operators must commute with each other"
        self.assertIn(err_msg, str(cm.exception))

        # Add an additional stabilizer first to make sure dimensions are correct
        additional_stab = Stabilizer(
            pauli="XX",
            data_qubits=((10, 10, 0), (10, 11, 0)),
        )
        # Add an additional logical X first to make sure dimensions are correct
        additional_log_x_operator = PauliOperator(pauli="X", data_qubits=((10, 10, 0),))
        # Add a non-commuting logical Z operator
        non_commuting_logical_z = PauliOperator(pauli="X", data_qubits=((0, 0, 0),))
        with self.assertRaises(ValueError) as cm:
            _ = Block(
                unique_label=self.rotated_surface_code.unique_label,
                stabilizers=self.rotated_surface_code.stabilizers + (additional_stab,),
                logical_x_operators=self.rotated_surface_code.logical_x_operators
                + (additional_log_x_operator,),
                logical_z_operators=self.rotated_surface_code.logical_z_operators
                + (non_commuting_logical_z,),
            )
        err_msg = "Logical Z operators must commute with each other"
        self.assertIn(err_msg, str(cm.exception))

    def test_validate_commutation_stabilizers_logical_operators(self):
        """
        Test that an error is raised if the stabilizers do not commute with the
        logical operators. This is tested in
        _validate_commutation_stabilizers_logical_operators() after the Block is
        created.
        """
        # Test that stabilizers commute with all logical X
        # Add an additional stabilizer first to make sure dimensions are correct
        additional_stab = Stabilizer(
            pauli="ZZ",
            data_qubits=((10, 10, 0), (10, 11, 0)),
        )
        # Add an additional logical Z first to make sure dimensions are correct
        commuting_logical_z = PauliOperator(pauli="Z", data_qubits=((10, 10, 0),))
        # Add a non-commuting logical X operator to trigger the error
        non_commuting_logical_x = PauliOperator(pauli="X", data_qubits=((10, 10, 0),))
        with self.assertRaises(ValueError) as cm:
            _ = Block(
                unique_label=self.rotated_surface_code.unique_label,
                stabilizers=self.rotated_surface_code.stabilizers + (additional_stab,),
                logical_x_operators=self.rotated_surface_code.logical_x_operators
                + (non_commuting_logical_x,),
                logical_z_operators=self.rotated_surface_code.logical_z_operators
                + (commuting_logical_z,),
            )
        err_msg = "Stabilizers must commute with logical X operators"
        self.assertIn(err_msg, str(cm.exception))

        # Test that stabilizers commute with all logical Z
        # Add an additional stabilizer first to make sure dimensions are correct
        additional_stab = Stabilizer(
            pauli="XX",
            data_qubits=((10, 10, 0), (10, 11, 0)),
        )
        # Add an additional logical X first to make sure dimensions are correct
        commuting_logical_x = PauliOperator(pauli="X", data_qubits=((10, 10, 0),))
        # Add a non-commuting logical Z operator to trigger the error
        non_commuting_logical_z = PauliOperator(pauli="Z", data_qubits=((10, 10, 0),))
        with self.assertRaises(ValueError) as cm:
            _ = Block(
                unique_label=self.rotated_surface_code.unique_label,
                stabilizers=self.rotated_surface_code.stabilizers + (additional_stab,),
                logical_x_operators=self.rotated_surface_code.logical_x_operators
                + (commuting_logical_x,),
                logical_z_operators=self.rotated_surface_code.logical_z_operators
                + (non_commuting_logical_z,),
            )
        err_msg = "Stabilizers must commute with logical Z operators"
        self.assertIn(err_msg, str(cm.exception))

    def test_validate_anticommutation_logical_operators_one_to_one(self):
        """
        Test that an error is raised if the logical X and Z operators at the same
        index do not anti-commute. Test that an error is raised if the logical X and Z
        operators at different indices do not commute. This is tested in
        _validate_anticommutation_logical_operators_one_to_one() after the Block is
        created.
        """
        # Test anti-commutation of logical operators at the same index
        # Add an additional stabilizer first to make sure dimensions are correct
        additional_stab = Stabilizer(
            pauli="XX",
            data_qubits=((10, 10, 0), (10, 11, 0)),
        )
        # Add a logical X and logical Z operator that do not anti-commute
        new_logical_x = PauliOperator(pauli="X", data_qubits=((10, 10, 0),))
        new_logical_z = PauliOperator(pauli="X", data_qubits=((10, 11, 0),))
        with self.assertRaises(ValueError) as cm:
            _ = Block(
                unique_label=self.rotated_surface_code.unique_label,
                stabilizers=self.rotated_surface_code.stabilizers + (additional_stab,),
                logical_x_operators=self.rotated_surface_code.logical_x_operators
                + (new_logical_x,),
                logical_z_operators=self.rotated_surface_code.logical_z_operators
                + (new_logical_z,),
            )
        err_msg = (
            "Logical X and Z operators at the same index must anticommute with "
            "each other"
        )
        self.assertIn(err_msg, str(cm.exception))

        # Test commutation relation of logical operators at different indices
        # Add additional stabilizer first to make sure dimensions are correct
        additional_stab = Stabilizer(
            pauli="XXX",
            data_qubits=((0, 10, 0), (1, 10, 0), (2, 10, 0)),
        )
        # Add a logical X and logical Z operator that do not anti-commute
        new_logical_xs = (
            PauliOperator(pauli="X", data_qubits=((0, 10, 0),)),
            PauliOperator(pauli="XX", data_qubits=((1, 10, 0), (2, 10, 0))),
        )
        new_logical_zs = (
            PauliOperator(pauli="ZZ", data_qubits=((0, 10, 0), (1, 10, 0))),
            PauliOperator(pauli="ZZ", data_qubits=((1, 10, 0), (2, 10, 0))),
        )
        with self.assertRaises(ValueError) as cm:
            _ = Block(
                unique_label=self.rotated_surface_code.unique_label,
                stabilizers=self.rotated_surface_code.stabilizers + (additional_stab,),
                logical_x_operators=self.rotated_surface_code.logical_x_operators
                + new_logical_xs,
                logical_z_operators=self.rotated_surface_code.logical_z_operators
                + new_logical_zs,
            )
        err_msg = (
            "Logical X and Z operators at different indices must commute with "
            "each other"
        )
        self.assertIn(err_msg, str(cm.exception))

    def test_block_reducible_stabilizers(self):
        """
        Test that no error is raised when the stabilizers are reducible.
        """
        # Test that the stabilizers are reducible
        # Replace the usual stabilizers by reducible ones
        additional_stab = Stabilizer(
            pauli="ZZXX",
            data_qubits=((0, 2, 0), (1, 2, 0), (2, 1, 0), (2, 2, 0)),
        )
        _ = Block(
            unique_label=self.rotated_surface_code.unique_label,
            stabilizers=self.rotated_surface_code.stabilizers + (additional_stab,),
            logical_x_operators=self.rotated_surface_code.logical_x_operators,
            logical_z_operators=self.rotated_surface_code.logical_z_operators,
        )

    def test_loads_dumps(self):
        """
        Test that the loads and dumps functions work correctly.
        """
        stabilizers = [
            Stabilizer(
                pauli="XX",
                data_qubits=(
                    (0, 0),
                    (1, 0),
                ),
                ancilla_qubits=((3, 1),),
            ),
            Stabilizer(
                pauli="XX",
                data_qubits=(
                    (1, 0),
                    (2, 0),
                ),
                ancilla_qubits=((4, 1),),
            ),
        ]
        logical_x = PauliOperator(pauli="XXX", data_qubits=((0, 0), (1, 0), (2, 0)))
        logical_z = PauliOperator(pauli="ZZZ", data_qubits=((0, 0), (1, 0), (2, 0)))

        block = Block(
            stabilizers=stabilizers,
            logical_x_operators=[logical_x],
            logical_z_operators=[logical_z],
            unique_label="q1",
        )

        block_json = dumps(block)
        loaded_block = loads(Block, block_json)

        self.assertEqual(loaded_block, block)

    def test_qubit_properties(self):
        """
        Test that the data_qubits, ancilla_qubits, and qubits properties work correctly.
        """
        stabilizers = [
            Stabilizer(
                pauli="XX",
                data_qubits=(
                    (0, 0),
                    (1, 0),
                ),
                ancilla_qubits=((3, 1),),
            ),
            Stabilizer(
                pauli="XX",
                data_qubits=(
                    (1, 0),
                    (2, 0),
                ),
                ancilla_qubits=((4, 1),),
            ),
        ]
        logical_x = PauliOperator(pauli="XXX", data_qubits=((0, 0), (1, 0), (2, 0)))
        logical_z = PauliOperator(pauli="ZZZ", data_qubits=((0, 0), (1, 0), (2, 0)))

        block = Block(
            stabilizers=stabilizers,
            logical_x_operators=[logical_x],
            logical_z_operators=[logical_z],
            unique_label="q1",
        )
        # Comparing sets because the order of the qubits is not guaranteed
        self.assertEqual(set(block.data_qubits), set(((0, 0), (1, 0), (2, 0))))
        self.assertEqual(set(block.ancilla_qubits), set(((3, 1), (4, 1))))
        self.assertEqual(
            set(block.qubits), set(((0, 0), (1, 0), (2, 0), (3, 1), (4, 1)))
        )

    def test_shift_function(self):
        """
        Test whether the shift() function correctly shifts a block.
        """
        block_shifted_via_func = self.rotated_surface_code.shift((3, 5))

        # Define the shifted Block manually, to compare with the one created by
        # the function `create()`
        manual_block = Block(
            stabilizers=(
                Stabilizer(
                    pauli="ZZZZ",
                    data_qubits=((4, 5, 0), (4, 6, 0), (3, 5, 0), (3, 6, 0)),
                    ancilla_qubits=((4, 6, 1),),
                    uuid=self.rotated_surface_code.stabilizers[0].uuid,
                ),
                Stabilizer(
                    pauli="ZZZZ",
                    data_qubits=((5, 6, 0), (5, 7, 0), (4, 6, 0), (4, 7, 0)),
                    ancilla_qubits=((5, 7, 1),),
                    uuid=self.rotated_surface_code.stabilizers[1].uuid,
                ),
                Stabilizer(
                    pauli="XXXX",
                    data_qubits=((4, 6, 0), (3, 6, 0), (4, 7, 0), (3, 7, 0)),
                    ancilla_qubits=((4, 7, 1),),
                    uuid=self.rotated_surface_code.stabilizers[2].uuid,
                ),
                Stabilizer(
                    pauli="XXXX",
                    data_qubits=((5, 5, 0), (4, 5, 0), (5, 6, 0), (4, 6, 0)),
                    ancilla_qubits=((5, 6, 1),),
                    uuid=self.rotated_surface_code.stabilizers[3].uuid,
                ),
                Stabilizer(
                    pauli="XX",
                    data_qubits=((3, 5, 0), (3, 6, 0)),
                    ancilla_qubits=((3, 6, 1),),
                    uuid=self.rotated_surface_code.stabilizers[4].uuid,
                ),
                Stabilizer(
                    pauli="XX",
                    data_qubits=((5, 6, 0), (5, 7, 0)),
                    ancilla_qubits=((6, 7, 1),),
                    uuid=self.rotated_surface_code.stabilizers[5].uuid,
                ),
                Stabilizer(
                    pauli="ZZ",
                    data_qubits=((5, 5, 0), (4, 5, 0)),
                    ancilla_qubits=((5, 5, 1),),
                    uuid=self.rotated_surface_code.stabilizers[6].uuid,
                ),
                Stabilizer(
                    pauli="ZZ",
                    data_qubits=((4, 7, 0), (3, 7, 0)),
                    ancilla_qubits=((4, 8, 1),),
                    uuid=self.rotated_surface_code.stabilizers[7].uuid,
                ),
            ),
            logical_x_operators=(
                PauliOperator(
                    pauli="XXX", data_qubits=((3, 5, 0), (4, 5, 0), (5, 5, 0))
                ),
            ),
            logical_z_operators=(
                PauliOperator(
                    pauli="ZZZ", data_qubits=((3, 5, 0), (3, 6, 0), (3, 7, 0))
                ),
            ),
            unique_label="q1",
            syndrome_circuits=self.rotated_surface_code.syndrome_circuits,
            stabilizer_to_circuit=self.rotated_surface_code.stabilizer_to_circuit,
        )
        # self.rotated_surface_code and manual_block are not the same because
        # self.rotated_surface_code starts at (0,0) while manual_block starts
        # at (3,5)
        self.assertNotEqual(self.rotated_surface_code, manual_block)
        # block_shifted_via_func and manual_block both start at (3,5)
        self.assertEqual(block_shifted_via_func, manual_block)

    def test_shift_with_rename(self):
        """
        Test whether the shift() function also correctly renames a block.
        """
        block_shifted_same_label = self.rotated_surface_code.shift((3, 5))
        block_shifted_new_label = self.rotated_surface_code.shift(
            (3, 5), new_label="q3"
        )
        # Check for unique_label values
        self.assertEqual(
            block_shifted_same_label.unique_label,
            self.rotated_surface_code.unique_label,
        )
        self.assertEqual(block_shifted_same_label.unique_label, "q1")
        self.assertEqual(block_shifted_new_label.unique_label, "q3")
        # Check that the class type is not changed.
        self.assertEqual(
            type(self.rotated_surface_code), type(block_shifted_same_label)
        )
        self.assertEqual(type(self.rotated_surface_code), type(block_shifted_new_label))
        # Check that the syndrome circuits are still the same
        self.assertEqual(
            block_shifted_same_label.syndrome_circuits,
            self.rotated_surface_code.syndrome_circuits,
        )
        # Note that the stabilizer to syndrome circuit mapping is different because new
        # Stabilizer objects were created and thus the uuids changed

    def test_shift_function_invalid_input(self):
        """
        Test whether the input validation of shift() works.
        """
        # Position has the wrong dimension
        with self.assertRaises(ValueError):
            _ = self.rotated_surface_code.shift((3, 5, 7, 0))

    def test_rename(self):
        """
        Test the rename() function of a Block.
        """
        block_renamed = self.rotated_surface_code.rename("new_qb")
        # Check that the name changed
        self.assertEqual(block_renamed.unique_label, "new_qb")
        # Check that all other fields are the same
        self.assertEqual(
            block_renamed.stabilizers, self.rotated_surface_code.stabilizers
        )
        self.assertEqual(
            block_renamed.logical_x_operators,
            self.rotated_surface_code.logical_x_operators,
        )
        self.assertEqual(
            block_renamed.logical_z_operators,
            self.rotated_surface_code.logical_z_operators,
        )
        self.assertEqual(
            block_renamed.syndrome_circuits, self.rotated_surface_code.syndrome_circuits
        )
        self.assertEqual(
            block_renamed.stabilizer_to_circuit,
            self.rotated_surface_code.stabilizer_to_circuit,
        )
        # Check that the class type is not changed.
        self.assertEqual(type(self.rotated_surface_code), type(block_renamed))

    def test_block_creation_without_unique_label(self):
        """
        Test whether a `unique_label` is generated automatically if not provided.
        """
        # Test uuid creation for manual creation of the block
        stabilizers = [
            Stabilizer(
                pauli="XX",
                data_qubits=(
                    (0, 0),
                    (1, 0),
                ),
                ancilla_qubits=((3, 1),),
            ),
            Stabilizer(
                pauli="XX",
                data_qubits=(
                    (1, 0),
                    (2, 0),
                ),
                ancilla_qubits=((4, 1),),
            ),
        ]
        logical_x = PauliOperator(pauli="XXX", data_qubits=((0, 0), (1, 0), (2, 0)))
        logical_z = PauliOperator(pauli="ZZZ", data_qubits=((0, 0), (1, 0), (2, 0)))

        block = Block(
            stabilizers=stabilizers,
            logical_x_operators=[logical_x],
            logical_z_operators=[logical_z],
        )
        uuid_error(block.unique_label)

    def test_pauli_charges(self):
        """
        Tests whether the Pauli charges for a distance-3 rotated surface code,
        created manually, are correctly calculated.
        """
        pauli_charges_expected = {
            (0, 1, 0): "Z",
            (1, 2, 0): "X",
            (2, 1, 0): "Z",
            (0, 0, 0): "Y",
            (1, 1, 0): "_",
            (2, 0, 0): "Y",
            (0, 2, 0): "Y",
            (2, 2, 0): "Y",
            (1, 0, 0): "X",
        }
        self.assertEqual(
            self.rotated_surface_code.pauli_charges, pauli_charges_expected
        )

    def test_validation_stab_to_synd_circ_map(self):
        """
        Test whether the validation works that all uuids in the stabilizer to circuit
        map must exist.
        """
        stabilizers = [
            Stabilizer(
                pauli="XX",
                data_qubits=(
                    (0, 0),
                    (1, 0),
                ),
                ancilla_qubits=((3, 1),),
            ),
            Stabilizer(
                pauli="XX",
                data_qubits=(
                    (1, 0),
                    (2, 0),
                ),
                ancilla_qubits=((4, 1),),
            ),
        ]
        logical_x = PauliOperator(pauli="XXX", data_qubits=((0, 0), (1, 0), (2, 0)))
        logical_z = PauliOperator(pauli="ZZZ", data_qubits=((0, 0), (1, 0), (2, 0)))

        with self.assertRaises(ValueError) as cm:
            _ = Block(
                stabilizers=stabilizers,
                logical_x_operators=[logical_x],
                logical_z_operators=[logical_z],
                stabilizer_to_circuit={"wrong_uuid": "test"},
            )
        self.assertIn(
            "Stabilizer with uuid wrong_uuid is not present in the stabilizers.",
            str(cm.exception),
        )

        with self.assertRaises(ValueError) as cm:
            _ = Block(
                stabilizers=stabilizers,
                logical_x_operators=[logical_x],
                logical_z_operators=[logical_z],
                stabilizer_to_circuit={stabilizers[0].uuid: "wrong_uuid"},
            )
        self.assertIn(
            "Syndrome circuit with uuid wrong_uuid is not present in the syndrome circuits",
            str(cm.exception),
        )

    def test_bypass_validation_invalid_block(self):
        """
        Test whether the skip_validation flag works correctly.
        """
        stabilizers = [
            Stabilizer(
                pauli="XX",
                data_qubits=(
                    (0, 0),
                    (1, 0),
                ),
                ancilla_qubits=((3, 1),),
            ),
            Stabilizer(
                pauli="XX",
                data_qubits=(
                    (1, 0),
                    (2, 0),
                ),
                ancilla_qubits=((4, 1),),
            ),
        ]
        logical_x = PauliOperator(pauli="XXX", data_qubits=((0, 0), (1, 0), (2, 0)))
        logical_z = PauliOperator(pauli="XXX", data_qubits=((0, 0), (1, 0), (2, 0)))

        # Test that the Block is created without any issues, even though it should be invalid
        _ = Block(
            stabilizers=stabilizers,
            logical_x_operators=[logical_x],
            logical_z_operators=[logical_z],
            skip_validation=True,
        )
        # Check that the validation fails, when enabled
        with self.assertRaises(ValidationError):
            _ = Block(
                stabilizers=stabilizers,
                logical_x_operators=[logical_x],
                logical_z_operators=[logical_z],
                skip_validation=False,
            )

    def test_bypass_validation_valid_block(self):
        """
        Test whether that a block created with skip_validation flag is initialised
        in the same way as without.
        """
        stabilizers = [
            Stabilizer(
                pauli="XX",
                data_qubits=(
                    (0, 0),
                    (1, 0),
                ),
                ancilla_qubits=((3, 1),),
            ),
            Stabilizer(
                pauli="XX",
                data_qubits=(
                    (1, 0),
                    (2, 0),
                ),
                ancilla_qubits=((4, 1),),
            ),
        ]
        logical_x = PauliOperator(pauli="XXX", data_qubits=((0, 0), (1, 0), (2, 0)))
        logical_z = PauliOperator(pauli="ZZZ", data_qubits=((0, 0), (1, 0), (2, 0)))

        block_without_validation = Block(
            unique_label="block",
            stabilizers=stabilizers,
            logical_x_operators=[logical_x],
            logical_z_operators=[logical_z],
            skip_validation=True,
        )
        block_with_validation = Block(
            unique_label="block",
            stabilizers=stabilizers,
            logical_x_operators=[logical_x],
            logical_z_operators=[logical_z],
        )
        self.assertEqual(block_without_validation, block_with_validation)

    def test_from_blocks_constructor(self):
        """
        Test the from_blocks constructor of the Block class.
        """
        # block_shifted_no_overlap is shifted relative to
        # self.rotated_surface_code by (0,4), so it does not overlap with
        # self.rotated_surface_code
        block_shifted_no_overlap = Block(
            stabilizers=(
                Stabilizer(
                    pauli="ZZZZ",
                    data_qubits=((1, 4, 0), (1, 5, 0), (0, 4, 0), (0, 5, 0)),
                    ancilla_qubits=((1, 5, 1),),
                    uuid=self.rotated_surface_code.stabilizers[0].uuid,
                ),
                Stabilizer(
                    pauli="ZZZZ",
                    data_qubits=((2, 5, 0), (2, 6, 0), (1, 5, 0), (1, 6, 0)),
                    ancilla_qubits=((2, 6, 1),),
                    uuid=self.rotated_surface_code.stabilizers[1].uuid,
                ),
                Stabilizer(
                    pauli="XXXX",
                    data_qubits=((1, 5, 0), (0, 5, 0), (1, 6, 0), (0, 6, 0)),
                    ancilla_qubits=((1, 6, 1),),
                    uuid=self.rotated_surface_code.stabilizers[2].uuid,
                ),
                Stabilizer(
                    pauli="XXXX",
                    data_qubits=((2, 4, 0), (1, 4, 0), (2, 5, 0), (1, 5, 0)),
                    ancilla_qubits=((2, 5, 1),),
                    uuid=self.rotated_surface_code.stabilizers[3].uuid,
                ),
                Stabilizer(
                    pauli="XX",
                    data_qubits=((0, 4, 0), (0, 5, 0)),
                    ancilla_qubits=((0, 5, 1),),
                    uuid=self.rotated_surface_code.stabilizers[4].uuid,
                ),
                Stabilizer(
                    pauli="XX",
                    data_qubits=((2, 5, 0), (2, 6, 0)),
                    ancilla_qubits=((3, 6, 1),),
                    uuid=self.rotated_surface_code.stabilizers[5].uuid,
                ),
                Stabilizer(
                    pauli="ZZ",
                    data_qubits=((2, 4, 0), (1, 4, 0)),
                    ancilla_qubits=((2, 4, 1),),
                    uuid=self.rotated_surface_code.stabilizers[6].uuid,
                ),
                Stabilizer(
                    pauli="ZZ",
                    data_qubits=((1, 6, 0), (0, 6, 0)),
                    ancilla_qubits=((1, 7, 1),),
                    uuid=self.rotated_surface_code.stabilizers[7].uuid,
                ),
            ),
            logical_x_operators=[
                PauliOperator(
                    pauli="XXX", data_qubits=((0, 4, 0), (1, 4, 0), (2, 4, 0))
                )
            ],
            logical_z_operators=[
                PauliOperator(
                    pauli="ZZZ", data_qubits=((0, 4, 0), (0, 5, 0), (0, 6, 0))
                )
            ],
            syndrome_circuits=self.rotated_surface_code.syndrome_circuits,
            stabilizer_to_circuit=self.rotated_surface_code.stabilizer_to_circuit,
            unique_label="q1",
        )

        combined_block = Block.from_blocks(
            [self.rotated_surface_code, block_shifted_no_overlap]
        )

        # Check that the combined block has the same qubits and data qubits as the two
        # individual blocks
        combined_qubits = set(self.rotated_surface_code.qubits) | set(
            block_shifted_no_overlap.qubits
        )
        self.assertEqual(set(combined_block.qubits), combined_qubits)
        combined_data_qubits = set(self.rotated_surface_code.data_qubits) | set(
            block_shifted_no_overlap.data_qubits
        )
        self.assertEqual(set(combined_block.data_qubits), combined_data_qubits)

        # Check that if the two blocks overlap, an exception is raised
        # block_shifted_overlap is shifted relative to self.rotated_surface_code by (1,2), so it
        # overlaps with self.rotated_surface_code
        block_shifted_overlap = Block(
            stabilizers=(
                Stabilizer(
                    pauli="ZZZZ",
                    data_qubits=((2, 2, 0), (2, 3, 0), (1, 2, 0), (1, 3, 0)),
                    ancilla_qubits=((2, 3, 1),),
                    uuid=self.rotated_surface_code.stabilizers[0].uuid,
                ),
                Stabilizer(
                    pauli="ZZZZ",
                    data_qubits=((3, 3, 0), (3, 4, 0), (2, 3, 0), (2, 4, 0)),
                    ancilla_qubits=((3, 4, 1),),
                    uuid=self.rotated_surface_code.stabilizers[1].uuid,
                ),
                Stabilizer(
                    pauli="XXXX",
                    data_qubits=((2, 3, 0), (1, 3, 0), (2, 4, 0), (1, 4, 0)),
                    ancilla_qubits=((2, 4, 1),),
                    uuid=self.rotated_surface_code.stabilizers[2].uuid,
                ),
                Stabilizer(
                    pauli="XXXX",
                    data_qubits=((3, 2, 0), (2, 2, 0), (3, 3, 0), (2, 3, 0)),
                    ancilla_qubits=((3, 3, 1),),
                    uuid=self.rotated_surface_code.stabilizers[3].uuid,
                ),
                Stabilizer(
                    pauli="XX",
                    data_qubits=((1, 2, 0), (1, 3, 0)),
                    ancilla_qubits=((1, 3, 1),),
                    uuid=self.rotated_surface_code.stabilizers[4].uuid,
                ),
                Stabilizer(
                    pauli="XX",
                    data_qubits=((3, 3, 0), (3, 4, 0)),
                    ancilla_qubits=((4, 4, 1),),
                    uuid=self.rotated_surface_code.stabilizers[5].uuid,
                ),
                Stabilizer(
                    pauli="ZZ",
                    data_qubits=((3, 2, 0), (2, 2, 0)),
                    ancilla_qubits=((3, 2, 1),),
                    uuid=self.rotated_surface_code.stabilizers[6].uuid,
                ),
                Stabilizer(
                    pauli="ZZ",
                    data_qubits=((2, 4, 0), (1, 4, 0)),
                    ancilla_qubits=((2, 5, 1),),
                    uuid=self.rotated_surface_code.stabilizers[7].uuid,
                ),
            ),
            logical_x_operators=[
                PauliOperator(
                    pauli="XXX", data_qubits=((1, 2, 0), (2, 2, 0), (3, 2, 0))
                )
            ],
            logical_z_operators=[
                PauliOperator(
                    pauli="ZZZ", data_qubits=((1, 2, 0), (1, 3, 0), (1, 4, 0))
                )
            ],
            syndrome_circuits=self.rotated_surface_code.syndrome_circuits,
            stabilizer_to_circuit=self.rotated_surface_code.stabilizer_to_circuit,
            unique_label="q3",
        )

        with self.assertRaises(ValueError):
            Block.from_blocks([self.rotated_surface_code, block_shifted_overlap])

        # And also if we use the same block twice
        with self.assertRaises(ValueError):
            Block.from_blocks([self.rotated_surface_code, self.rotated_surface_code])

    def test_stabilizers_labels(self):
        """Test the stabilizers_labels property."""

        expected_labels = {}

        for stabilizer in self.rotated_surface_code.stabilizers:
            expected_labels[stabilizer.uuid] = {
                "space_coordinates": stabilizer.ancilla_qubits[0]
            }

        self.assertEqual(self.rotated_surface_code.stabilizers_labels, expected_labels)

    def test_get_stabilizer_label(self):
        """Test the correct extraction of a single stabilizer label."""

        for stabilizer in self.rotated_surface_code.stabilizers:
            expected_label = {"space_coordinates": stabilizer.ancilla_qubits[0]}
            label = self.rotated_surface_code.get_stabilizer_label(stabilizer.uuid)
            self.assertEqual(label, expected_label)


if __name__ == "__main__":
    unittest.main()
