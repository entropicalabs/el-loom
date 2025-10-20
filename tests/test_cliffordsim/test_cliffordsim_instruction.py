"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

import unittest
from unittest.mock import Mock
import numpy as np

from loom.cliffordsim.tableau import Tableau
from loom.cliffordsim.pauli_frame import PauliFrame
from loom.cliffordsim.data_store import DataStore
from loom.cliffordsim.classicalreg import ClassicalRegister
from loom.cliffordsim.moments.instruction import (
    IdentityInstruction,
    IdentityDecorator,
    HadamardDecorator,
    PhaseDecorator,
    CNOTDecorator,
    CZDecorator,
    XDecorator,
    ZDecorator,
    YDecorator,
    MeasurementDecorator,
    DecoratorSelector,
    UpdateTableauDecorator,
    CreatePauliFrameDecorator,
    RecordPauliFrameDecorator,
    ClassicalBitDecorator,
)
from loom.cliffordsim.operations import (
    Identity,
    Hadamard,
    Phase,
    CNOT,
    CZ,
    X,
    Y,
    Z,
    Measurement,
    UpdateTableau,
    CreatePauliFrame,
    RecordPauliFrame,
)


class TestCliffordSimInstruction(unittest.TestCase):
    """
    We test the Instruction classes and their associated functionality.
    """

    def test_identityinstruction(self):
        """The identity instruction should return the same
        Tableau. There should be no changes in its values."""
        test_tab = Tableau(2)
        input_tab = Tableau(2)
        data_store = DataStore()
        iden_inst = IdentityInstruction()
        new_tab, _, _ = iden_inst.apply(input_tab, data_store)

        self.assertTrue(np.array_equal(test_tab.tableau, new_tab.tableau))

    def test_identitydecorator(self):
        """Check for the application of the identity decorator operation."""
        test_tab = Tableau(2)
        input_tab = Tableau(2)
        data_store = DataStore()

        # Apply identity to the first qubit
        iden_inst = IdentityDecorator(IdentityInstruction(), Identity(0))
        new_tab, _, _ = iden_inst.apply(input_tab, data_store)

        self.assertTrue(np.array_equal(test_tab.tableau, new_tab.tableau))

        # Apply identity to both qubits
        iden_inst = IdentityDecorator(
            IdentityDecorator(IdentityInstruction(), Identity(0)), Identity(1)
        )
        new_tab, _, _ = iden_inst.apply(input_tab, data_store)

        self.assertTrue(np.array_equal(test_tab.tableau, new_tab.tableau))

    def test_hadamarddecorator(self):
        """Check for the application of the hadamard decorator operation."""
        test_tab = Tableau(2)
        input_tab = Tableau(2)
        data_store = DataStore()
        hadamard_inst = HadamardDecorator(
            XDecorator(IdentityInstruction(), X(0)), Hadamard(0)
        )
        new_tab, _, _ = hadamard_inst.apply(input_tab, data_store)

        # Columns X and Z are swapped
        self.assertFalse(np.array_equal(test_tab.z[:, 0], new_tab.z[:, 0]))
        self.assertFalse(np.array_equal(test_tab.x[:, 0], new_tab.x[:, 0]))
        self.assertTrue(np.array_equal(test_tab.x[:, 0], new_tab.z[:, 0]))
        self.assertTrue(np.array_equal(test_tab.z[:, 0], new_tab.x[:, 0]))

        # Column R = R XOR (X and Z)
        self.assertFalse(np.array_equal(test_tab.r, new_tab.r))
        # Manually apply change to R from H decorator
        test_tab.r ^= test_tab.x[:, 0] & test_tab.z[:, 0]
        # Manually apply change to R from X decorator
        # Since H swaps the X and Z Columns, the X decorator XOR from the
        # original X Column instead of the Z Column
        test_tab.r ^= test_tab.x[:, 0]
        self.assertTrue(np.array_equal(test_tab.r, new_tab.r))

    def test_phasedecorator(self):
        """Check for the application of the phase decorator operation."""
        test_tab = Tableau(2)
        input_tab = Tableau(2)
        data_store = DataStore()
        phase_inst = XDecorator(PhaseDecorator(IdentityInstruction(), Phase(0)), X(0))
        new_tab, _, _ = phase_inst.apply(input_tab, data_store)

        # Column Z = Z XOR X
        self.assertFalse(np.array_equal(test_tab.z, new_tab.z))
        self.assertTrue(
            np.array_equal(new_tab.z[:, 0], (test_tab.z[:, 0] ^ test_tab.x[:, 0]))
        )

        # Column R = R XOR (X and Z)
        self.assertFalse(np.array_equal(test_tab.r, new_tab.r))
        # Manually apply the X Decorator
        test_tab.r ^= test_tab.z[:, 0]
        # Manually apply the Phase Decorator
        test_tab.r ^= test_tab.x[:, 0] & test_tab.z[:, 0]
        self.assertTrue(np.array_equal(test_tab.r, new_tab.r))

    def test_xdecorator(self):
        """Check for the application of the X decorator operation."""
        test_tab = Tableau(2)
        input_tab = Tableau(2)
        data_store = DataStore()
        x_inst = XDecorator(IdentityInstruction(), X(0))
        new_tab, _, _ = x_inst.apply(input_tab, data_store)

        # Column R = R XOR Z
        self.assertFalse(np.array_equal(test_tab.r, new_tab.r))
        test_tab.r ^= test_tab.z[:, 0]
        self.assertTrue(np.array_equal(test_tab.r, new_tab.r))

    def test_zdecorator(self):
        """Check for the application of the Z decorator operation."""
        test_tab = Tableau(2)
        input_tab = Tableau(2)
        data_store = DataStore()
        z_inst = ZDecorator(IdentityInstruction(), Z(0))
        new_tab, _, _ = z_inst.apply(input_tab, data_store)

        # Column R = R XOR X
        self.assertFalse(np.array_equal(test_tab.r, new_tab.r))
        test_tab.r ^= test_tab.x[:, 0]
        self.assertTrue(np.array_equal(test_tab.r, new_tab.r))

    def test_ydecorator(self):
        """Check for the application of the Y decorator operation."""
        test_tab = Tableau(2)
        input_tab = Tableau(2)
        data_store = DataStore()
        y_inst = YDecorator(IdentityInstruction(), Y(0))
        new_tab, _, _ = y_inst.apply(input_tab, data_store)

        # Column R = R XOR X XOR Z
        self.assertFalse(np.array_equal(test_tab.r, new_tab.r))
        test_tab.r ^= test_tab.x[:, 0] ^ test_tab.z[:, 0]
        self.assertTrue(np.array_equal(test_tab.r, new_tab.r))

    def test_cnotdecorator(self):
        """Check for the application of the CNOT decorator operation."""
        test_tab = Tableau(3)
        input_tab = Tableau(3)
        data_store = DataStore()
        cnot_inst = XDecorator(CNOTDecorator(IdentityInstruction(), CNOT(0, 1)), X(0))
        new_tab, _, _ = cnot_inst.apply(input_tab, data_store)

        self.assertFalse(np.array_equal(test_tab.r, new_tab.r))

        # Manually apply the X Decorator
        test_tab.r ^= test_tab.z[:, 0]

        # Where a and b are the control and target qubits respectively
        # Column R = R XOR (X[a] and Z[b] and (X[b] XOR Z[a] XOR [1, 1, ...]))
        self.assertTrue(
            np.array_equal(
                new_tab.r,
                (
                    test_tab.r
                    ^ test_tab.x[:, 0]
                    & test_tab.z[:, 1]
                    & (test_tab.x[:, 1] ^ test_tab.z[:, 0] ^ [1] * 6)
                ),
            )
        )

        # Column X[b] = X[b] XOR X[a]
        self.assertTrue(
            np.array_equal(new_tab.x[:, 1], (test_tab.x[:, 1] ^ test_tab.x[:, 0]))
        )

        # Column Z[a] = Z[a] XOR Z[b]
        self.assertTrue(
            np.array_equal(new_tab.z[:, 0], (test_tab.z[:, 0] ^ test_tab.z[:, 1]))
        )

    def test_czdecorator(self):
        """Check for the application of the CZ decorator operation."""
        test_tab = Tableau(3)
        input_tab = Tableau(3)
        data_store = DataStore()
        cz_inst = XDecorator(CZDecorator(IdentityInstruction(), CZ(0, 1)), X(0))
        new_tab, _, _ = cz_inst.apply(input_tab, data_store)

        self.assertFalse(np.array_equal(test_tab.r, new_tab.r))

        # Manually apply the X Decorator
        test_tab.r ^= test_tab.z[:, 0]

        # Where a and b are the control and target qubits respectively
        # Column R = R XOR (X[a] and X[b] and (Z[b] XOR Z[a] XOR [1, 1, ...]))
        self.assertTrue(
            np.array_equal(
                new_tab.r,
                (
                    test_tab.r
                    ^ test_tab.x[:, 0]
                    & test_tab.x[:, 1]
                    & (test_tab.z[:, 1] ^ test_tab.z[:, 0] ^ [1] * 6)
                ),
            )
        )

        # Column Z[b] = Z[b] XOR X[a]
        self.assertTrue(
            np.array_equal(new_tab.z[:, 1], (test_tab.z[:, 1] ^ test_tab.x[:, 0]))
        )

        # Column Z[a] = Z[a] XOR X[b]
        self.assertTrue(
            np.array_equal(new_tab.z[:, 0], (test_tab.z[:, 0] ^ test_tab.x[:, 1]))
        )

    def test_measurementdecorator(self):
        """Check for the application of the Measurement decorator operation."""
        input_tab = Tableau(1)
        data_store = DataStore()
        data_store.set_time_step(0)
        meas_inst = HadamardDecorator(
            MeasurementDecorator(IdentityInstruction(), Measurement(0)),
            Hadamard(0),
        )
        new_tab, _, _ = meas_inst.apply(input_tab, data_store)

        # When we measure in Z-basis, the qubit must be in at least +Z or -Z
        stab_set = new_tab.stabilizer_set
        self.assertTrue(any(each_value in stab_set for each_value in ["+Z", "-Z"]))

    def test_measurement_decorator_2(self):
        """
        Check that the measurements are recorded appropriately.
        """
        input_tab = Tableau(1)
        data_store = DataStore()
        # TimeStep of the Measurement Operation
        data_store.set_time_step(1)
        meas_dec = MeasurementDecorator(IdentityInstruction(), Measurement(0))
        meas_inst = XDecorator(
            meas_dec,
            X(0),
        )
        _ = meas_inst.apply(input_tab, data_store)

        data_store_mea = data_store.measurements

        self.assertEqual(data_store_mea["time_step"], [1])
        self.assertEqual(
            data_store_mea["1"][meas_dec.measurement_id]["measurement_result"],
            1,
        )
        self.assertEqual(
            data_store_mea["1"][meas_dec.measurement_id]["is_random"],
            False,
        )

    def test_measurementdecorator_bias(self):
        """Check for the biased application of the Measurement decorator operation."""
        for bias in [0, 1]:
            input_tab = Tableau(1)
            data_store = DataStore()
            data_store.set_time_step(1)
            meas_decorator = MeasurementDecorator(
                IdentityInstruction(), Measurement(0, bias=bias)
            )
            total_decorator = HadamardDecorator(
                meas_decorator,
                Hadamard(0),
            )
            _, _, _ = total_decorator.apply(input_tab, data_store)

            # The result should be equal to the bias
            data_store_mea = data_store.measurements

            self.assertEqual(data_store_mea["time_step"], [1])
            self.assertEqual(
                data_store_mea["1"][meas_decorator.measurement_id][
                    "measurement_result"
                ],
                bias,
            )

    def test_measurement_decorator_bases(self):
        """Check that the measurement decorator works with different bases."""
        # Define the bases to be tested
        bases = ["X", "Y", "Z"]

        # Define the decorators/operators to set the state to 0 in the respective bases
        # The operators need to be given in the reverse order of application
        deterministic_result_decs0 = [
            [
                (HadamardDecorator, Hadamard(0)),
            ],
            [(PhaseDecorator, Phase(0)), (HadamardDecorator, Hadamard(0))],
            [],
        ]

        for basis, det_res_dec0 in zip(bases, deterministic_result_decs0, strict=True):
            input_tab = Tableau(1)
            data_store = DataStore()
            data_store.set_time_step(1)
            meas_decorator = MeasurementDecorator(
                IdentityInstruction(), Measurement(0, basis=basis)
            )
            # Apply the decorators
            total_decorator = meas_decorator
            for det_res_dec in det_res_dec0:
                decorator, operator = det_res_dec
                total_decorator = decorator(total_decorator, operator)
            _, _, _ = total_decorator.apply(input_tab, data_store)

            # The result should be 0 in time step 1
            data_store_mea = data_store.measurements
            self.assertEqual(data_store_mea["time_step"], [1])
            self.assertEqual(
                data_store_mea["1"][meas_decorator.measurement_id][
                    "measurement_result"
                ],
                0,
            )

    def test_updatetableaudecorator(self):
        """Check that the Tableau is updated correctly."""
        input_tab_w_scratch = Tableau(1)
        data_store = DataStore()
        new_tableau = np.array([[0, 1, 0], [1, 0, 0]])

        self.assertFalse(np.array_equal(new_tableau, input_tab_w_scratch.tableau))

        ut_dec = UpdateTableauDecorator(
            IdentityInstruction(), UpdateTableau(new_tableau)
        )
        new_tab_w_scratch, new_data_store, _ = ut_dec.apply(
            input_tab_w_scratch, data_store
        )

        self.assertEqual(data_store, new_data_store)  # No Changes here.
        self.assertTrue(np.array_equal(new_tab_w_scratch.tableau, new_tableau))
        # The tableau in Tableau is modified in-place
        self.assertTrue(
            np.array_equal(new_tab_w_scratch.tableau, input_tab_w_scratch.tableau)
        )

    def test_createpauliframedecorator(self):
        """Check that the PauliFrame is correctly created
        when applied in the right direction."""
        new_pff = PauliFrame.from_string("X", direction="forward")
        new_pfb = PauliFrame.from_string("X", direction="backward")
        data_store = DataStore()

        # Check that the PauliFrame is correctly created in the right direction
        # Forward PauliFrame is created during forward propagation
        cpf_dec = CreatePauliFrameDecorator(
            IdentityInstruction(), CreatePauliFrame(new_pff)
        )
        pfs, data_store, _ = cpf_dec.apply_pf([], data_store)
        self.assertEqual(pfs, [new_pff])
        # Backward PauliFrame is created during backward propagation
        cpf_dec_back = CreatePauliFrameDecorator(
            IdentityInstruction(), CreatePauliFrame(new_pfb)
        )
        pfs_back, data_store, _ = cpf_dec_back.apply_pf_back([], data_store)
        self.assertEqual(pfs_back, [new_pfb])

        # Check that the PauliFrame is not created in the wrong direction
        # Forward PauliFrame is not created during backward propagation
        pfs, data_store, _ = cpf_dec.apply_pf_back([], data_store)
        self.assertEqual(pfs, [])
        # Backward PauliFrame is not created during forward propagation
        pfs_back, data_store, _ = cpf_dec_back.apply_pf([], data_store)
        self.assertEqual(pfs_back, [])

    def test_recordpauliframedecorator(self):
        """Check that the PauliFrame is correctly recorded
        when applied in the right direction."""
        pff = PauliFrame.from_string("X", direction="forward")
        pfb = PauliFrame.from_string("X", direction="backward")

        correct_record_forward = {
            "time_step": [1],
            "1": {pff.id: {"initial_pauli_frame": pff, "recorded_pauli_frame": pff}},
        }
        correct_record_backward = {
            "time_step": [1],
            "1": {pfb.id: {"initial_pauli_frame": pfb, "recorded_pauli_frame": pfb}},
        }
        empty_record = {"time_step": []}

        # Check that the PauliFrame is correctly recorded in the right direction
        # and not recorded in the wrong direction

        # Forward PauliFrame is recorded during forward propagation
        rpf_dec = RecordPauliFrameDecorator(
            IdentityInstruction(), RecordPauliFrame(pff)
        )
        data_store = DataStore()
        data_store.set_time_step(1)
        pfs, data_store, _ = rpf_dec.apply_pf([pff], data_store)
        self.assertEqual(pfs, [pff])  # Check that the PFs are not modified

        self.assertEqual(data_store.pf_records["forward"], correct_record_forward)
        self.assertEqual(data_store.pf_records["backward"], empty_record)

        # Forward RecordPauliFrame is not recorded during backward propagation
        data_store = DataStore()
        data_store.set_time_step(1)
        pfs, data_store, _ = rpf_dec.apply_pf_back([pff], data_store)
        self.assertEqual(pfs, [pff])  # Check that the PFs are not modified
        self.assertEqual(data_store.pf_records["forward"], empty_record)
        self.assertEqual(data_store.pf_records["backward"], empty_record)

        # Backward PauliFrame is recorded during backward propagation
        rpf_dec_back = RecordPauliFrameDecorator(
            IdentityInstruction(), RecordPauliFrame(pfb)
        )
        data_store = DataStore()
        data_store.set_time_step(1)
        pfs, data_store, _ = rpf_dec_back.apply_pf_back([pfb], data_store)
        self.assertEqual(pfs, [pfb])  # Check that the PFs are not modified
        self.assertEqual(data_store.pf_records["backward"], correct_record_backward)
        self.assertEqual(data_store.pf_records["forward"], empty_record)
        # Backward RecordPauliFrame is not recorded during forward propagation
        data_store = DataStore()
        data_store.set_time_step(1)
        pfs, data_store, _ = rpf_dec_back.apply_pf([pfb], data_store)
        self.assertEqual(pfs, [pfb])  # Check that the PFs are not modified
        self.assertEqual(data_store.pf_records["forward"], empty_record)
        self.assertEqual(data_store.pf_records["backward"], empty_record)

    def test_measurement_from_pframe(self):
        """Check that measurement Instruction works with Pauliframe Instructions
        `cases_to_check` = {mbasis_str: {pauli_frame_str: expected_meas_result}}
        """
        cases_to_check = {
            "Z": {"X": 1, "Y": 1, "Z": 0},
            "X": {"X": 0, "Y": 1, "Z": 1},
            "Y": {"X": 1, "Y": 0, "Z": 1},
        }

        for mbasis, cases in cases_to_check.items():
            for pauli_type, expected_meas_result in cases.items():
                pff = PauliFrame.from_string(pauli_type, direction="forward")
                input_tab = Tableau(1)
                data_store = DataStore()

                # instructions to be applied in reverse, Measurement first then CreatePauliFrame.
                data_store.set_time_step(1)
                meas_dec = MeasurementDecorator(
                    IdentityInstruction(), Measurement(0, basis=mbasis)
                )
                _, _, _ = meas_dec.apply(input_tab, data_store)

                cpf_dec = CreatePauliFrameDecorator(meas_dec, CreatePauliFrame(pff))
                _, data_store, _ = cpf_dec.apply_pf([], data_store)

                data_store_mea = data_store.measurements
                self.assertEqual(
                    data_store_mea["1"][meas_dec.measurement_id]["flip_results"][
                        pff.id
                    ],
                    expected_meas_result,
                )

    def test_inputoperationtypecheck(self):
        """Check that the Decorator only accepts the Operation Object of a
        similar type in its initialization."""
        wrong_op_mock = Mock()
        wrong_op_mock.__class__.__name__ = "Wrong"
        wrong_input_operation = ["string", 0, 0.0, None, wrong_op_mock]
        for each_attr in DecoratorSelector:
            # Wrong initialization inputs into input_operation.
            for each_wrong_input in wrong_input_operation:
                with self.assertRaises(TypeError):
                    each_attr.value(IdentityInstruction(), each_wrong_input)

            # Right initialization input into input_operation.
            right_op_mock = Mock()
            right_op_mock.__class__.__name__ = each_attr.value.__name__[:-9]
            each_attr.value(IdentityInstruction(), right_op_mock)

    # pylint: disable=protected-access
    def test_classicalbitdecorator(self):
        """Checks if the private methods in ClassicalBitDecorator return the expected
        outputs.
        """
        # Create Decorator
        instruction = unittest.mock.Mock(spec=IdentityInstruction)
        mock_op = unittest.mock.MagicMock()
        mock_op.__class__.__name__ = "ClassicalBit"
        test_dec = ClassicalBitDecorator(instruction, mock_op)

        # Create Classical Register
        classical_reg = ClassicalRegister("testreg_name", 2, ["bit_0", "bit_1"])

        # Fetch by bit_order
        self.assertEqual(
            (0, 0, "bit_0"), test_dec._get_bit_info(classical_reg, 0, None)
        )
        self.assertEqual(
            (0, 1, "bit_1"), test_dec._get_bit_info_w_bit_order(classical_reg, 1)
        )

        # Fetch by bit_id
        self.assertEqual(
            (0, 0, "bit_0"), test_dec._get_bit_info(classical_reg, None, "bit_0")
        )
        self.assertEqual(
            (0, 1, "bit_1"), test_dec._get_bit_info_w_bit_id(classical_reg, "bit_1")
        )


if __name__ == "__main__":
    unittest.main()
