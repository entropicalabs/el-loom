"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

# pylint: disable=too-many-lines
import uuid
import unittest
import numpy as np

from loom.cliffordsim.engine import Engine
from loom.cliffordsim.operations import (
    Hadamard,
    Phase,
    PhaseInv,
    CNOT,
    CZ,
    CY,
    X,
    Y,
    Z,
    Measurement,
    SWAP,
    AddQubit,
    DeleteQubit,
    Reset,
    UpdateTableau,
    RecordPauliFrame,
    CreateClassicalRegister,
    RecordClassicalRegister,
    ClassicalNOT,
    ClassicalOR,
    ClassicalAND,
)
from loom.cliffordsim.pauli_frame import PauliFrame
from loom.cliffordsim.exceptions import (
    EngineRunError,
    TableauSizeError,
    InvalidTableauError,
    ClassicalRegisterError,
    ClassicalOperationError,
)
from loom.cliffordsim.tableau import compare_stabilizer_set
from loom.cliffordsim.classicalreg import ClassicalRegisterSnapshot, ClassicalRegister
from loom.eka.utilities import (
    is_tableau_valid,
    is_subset_of_stabarray,
    SignedPauliOp,
    StabArray,
)

# pylint: disable=import-error, wrong-import-order
from utilities import random_list_of_gates


class TestCliffordSimEngine(
    unittest.TestCase
):  # pylint: disable=too-many-public-methods
    """
    We test the CliffordSim Engine and its associated functionality.
    """

    def test_initialization(self):
        """Tests initialized state."""
        nqubits = np.random.randint(1, 10)
        cliffordsim_engine = Engine([], nqubits)

        init_stab_set = {
            "+" + i * "_" + "Z" + (nqubits - i - 1) * "_" for i in range(nqubits)
        }

        self.assertEqual(cliffordsim_engine.stabilizer_set, init_stab_set)

    def test_invalid_initialization(self):
        """Test that invalid inputs raise an error"""
        nqubits = 5

        # test initializing with non-operation list
        cops = ["Not a cliffordsim operation"]
        with self.assertRaises(TypeError):
            _ = Engine(cops, nqubits)

        # test initialization with record but no creation of PF
        pf = PauliFrame.from_string("XXXXX")
        cops = [RecordPauliFrame(pf)]
        with self.assertRaises(ValueError):
            _ = Engine(cops, nqubits)

    def test_stabilizer_set(self):
        """Tests that the stabilizer set gives the correct strings."""
        operation_list = [Hadamard(0), Hadamard(1), Phase(1), X(2)]
        cliffordsim_engine = Engine(operation_list, 3)

        init_stab_set = {"+Z__", "+_Z_", "+__Z"}
        self.assertEqual(
            init_stab_set,
            cliffordsim_engine.stabilizer_set,
            "Incorrect initial stabilizer set.",
        )

        # apply some gates
        cliffordsim_engine.run()

        corr_stab_set = {"+X__", "+_Y_", "-__Z"}
        self.assertEqual(
            corr_stab_set,
            cliffordsim_engine.stabilizer_set,
            "Incorrect stabilizer set.",
        )

    def test_engine_run(self):
        """The Engine initializes the tableau of the right size and state.
        Checks that the state evolves correctly using bell pairs.
        """

        operation_list = [Hadamard(0), CNOT(0, 1)]
        nqubits = 2
        seed = 5

        cliffordsim_engine = Engine(operation_list, nqubits, seed)

        self.assertEqual(cliffordsim_engine.stabilizer_set, {"+_Z", "+Z_"})

        cliffordsim_engine.run()

        self.assertEqual(cliffordsim_engine.stabilizer_set, {"+XX", "+ZZ"})

        operation_list.append(X(0))

        cliffordsim_engine = Engine(operation_list, nqubits, seed)
        cliffordsim_engine.run()

        self.assertEqual(cliffordsim_engine.stabilizer_set, {"+XX", "-ZZ"})

        operation_list.append(Hadamard(1))

        cliffordsim_engine = Engine(operation_list, nqubits, seed)
        cliffordsim_engine.run()

        self.assertEqual(cliffordsim_engine.stabilizer_set, {"+XZ", "-ZX"})

        operation_list.extend([X(0), Phase(0), Phase(0)])

        cliffordsim_engine = Engine(operation_list, nqubits, seed)
        cliffordsim_engine.run()

        self.assertEqual(cliffordsim_engine.stabilizer_set, {"-XZ", "+ZX"})

    def test_sq_gates(self):
        """Tests serially the application of single qubit gates."""
        operation_list = []
        cliffordsim_engine = Engine(operation_list, 1)

        self.assertTrue(cliffordsim_engine.stabilizer_set == {"+Z"})

        operation_list.append(Hadamard(0))
        cliffordsim_engine = Engine(operation_list, 1)
        cliffordsim_engine.run()
        self.assertTrue(cliffordsim_engine.stabilizer_set == {"+X"})

        operation_list.append(Phase(0))
        cliffordsim_engine = Engine(operation_list, 1)
        cliffordsim_engine.run()
        self.assertTrue(cliffordsim_engine.stabilizer_set == {"+Y"})

        operation_list.append(Phase(0))
        cliffordsim_engine = Engine(operation_list, 1)
        cliffordsim_engine.run()
        self.assertTrue(cliffordsim_engine.stabilizer_set == {"-X"})

        operation_list.append(X(0))
        cliffordsim_engine = Engine(operation_list, 1)
        cliffordsim_engine.run()
        self.assertTrue(cliffordsim_engine.stabilizer_set == {"-X"})

        operation_list.append(Hadamard(0))
        cliffordsim_engine = Engine(operation_list, 1)
        cliffordsim_engine.run()
        self.assertTrue(cliffordsim_engine.stabilizer_set == {"-Z"})

    def test_phase_correctness(self):
        """Testing correctness of phase transformations with known circuit
        outputs.

        +Z -apply-> HSSSH -final-> +Y
        +Y -apply-> S -final-> -X
        """
        operation_list = [
            Hadamard(0),
            Phase(0),
            Phase(0),
            Phase(0),
            Hadamard(0),
            Phase(0),
        ]
        nqubits = 1
        seed = 5

        cliffordsim_engine = Engine(operation_list, nqubits, seed)

        cliffordsim_engine.run()

        self.assertEqual(cliffordsim_engine.stabilizer_set, {"-X"})

    def test_x_equivalence(self):
        """
        Check for the following equivalence:

        H_0 P_0 P_0 H_0 == X_0
        H_0 Z_0 H_0 == X_0
        """

        operation_list = [X(0)]
        operation_list_2 = [Hadamard(0), Phase(0), Phase(0), Hadamard(0)]
        operation_list_3 = [Hadamard(0), Z(0), Hadamard(0)]
        nqubits = 1
        seed = 5

        cliffordsim_engine = Engine(operation_list, nqubits, seed)
        cliffordsim_engine_2 = Engine(operation_list_2, nqubits, seed)
        cliffordsim_engine_3 = Engine(operation_list_3, nqubits, seed)

        cliffordsim_engine.run()
        cliffordsim_engine_2.run()
        cliffordsim_engine_3.run()

        self.assertEqual(
            cliffordsim_engine.stabilizer_set, cliffordsim_engine_2.stabilizer_set
        )
        self.assertEqual(
            cliffordsim_engine.stabilizer_set, cliffordsim_engine_3.stabilizer_set
        )

    def test_i_equivalence(self):
        """
        Check for the following equivalence:

        X_0 X_0 == I_0
        Y_0 Y_0 == I_0
        Z_0 Z_0 == I_0
        H_0 H_0 == I_0
        """
        operation_list = []
        operation_list_2 = [X(0), X(0)]
        operation_list_3 = [Y(0), Y(0)]
        operation_list_4 = [Z(0), Z(0)]
        operation_list_5 = [Hadamard(0), Hadamard(0)]
        nqubits = 1
        seed = 5

        cliffordsim_engine = Engine(operation_list, nqubits, seed)
        cliffordsim_engine_2 = Engine(operation_list_2, nqubits, seed)
        cliffordsim_engine_3 = Engine(operation_list_3, nqubits, seed)
        cliffordsim_engine_4 = Engine(operation_list_4, nqubits, seed)
        cliffordsim_engine_5 = Engine(operation_list_5, nqubits, seed)

        cliffordsim_engine.run()
        cliffordsim_engine_2.run()
        cliffordsim_engine_3.run()
        cliffordsim_engine_4.run()
        cliffordsim_engine_5.run()

        self.assertEqual(
            cliffordsim_engine.stabilizer_set, cliffordsim_engine_2.stabilizer_set
        )
        self.assertEqual(
            cliffordsim_engine.stabilizer_set, cliffordsim_engine_3.stabilizer_set
        )
        self.assertEqual(
            cliffordsim_engine.stabilizer_set, cliffordsim_engine_4.stabilizer_set
        )
        self.assertEqual(
            cliffordsim_engine.stabilizer_set, cliffordsim_engine_5.stabilizer_set
        )

    def test_cx_cz_equivalence(self):
        """
        Check for the following equivalence:

        CNOT_0_1 == H_1 CZ_0_1 H_1
        """

        operation_list = [X(0), CNOT(0, 1)]
        operation_list_2 = [X(0), Hadamard(1), CZ(0, 1), Hadamard(1)]
        nqubits = 2
        seed = 5

        cliffordsim_engine = Engine(operation_list, nqubits, seed)
        cliffordsim_engine_2 = Engine(operation_list_2, nqubits, seed)

        cliffordsim_engine.run()
        cliffordsim_engine_2.run()

        self.assertEqual(
            cliffordsim_engine.stabilizer_set, cliffordsim_engine_2.stabilizer_set
        )

    def test_phaseinv(self):
        "Test that Phase followed by PhaseInv has no effect"
        nqubits = 5

        for _ in range(3):
            gates = random_list_of_gates(5, 20)

            gates_w_phase_first = gates.copy()
            for i in range(nqubits):
                gates_w_phase_first.append(Phase(i))
                gates_w_phase_first.append(PhaseInv(i))

            gates_w_phaseinv_first = gates.copy()
            for i in range(nqubits):
                gates_w_phaseinv_first.append(PhaseInv(i))
                gates_w_phaseinv_first.append(Phase(i))

            # define engines
            cat_eng0 = Engine(gates, nqubits)
            cat_eng1 = Engine(gates_w_phase_first, nqubits)
            cat_eng2 = Engine(gates_w_phaseinv_first, nqubits)
            # run engines
            cat_eng0.run()
            cat_eng1.run()
            cat_eng2.run()
            # make sure that the stabilizer sets match in all 3 cases
            self.assertTrue(
                compare_stabilizer_set(
                    cat_eng0.tableau_w_scratch,
                    cat_eng1.tableau_w_scratch,
                )
            )
            self.assertTrue(
                compare_stabilizer_set(
                    cat_eng1.tableau_w_scratch,
                    cat_eng2.tableau_w_scratch,
                )
            )

    def test_cx_cy_equivalence_random(self):
        """
        Check for the following equivalence:

        CY_0_1 == H_1 S_1^-1 CNOT_0_1 S_1 H_1
        """
        nqubits = 2

        for _ in range(20):
            op_list = random_list_of_gates(2, 20)

            op_list_cx = op_list + [
                Hadamard(1),
                Phase(1),
                CNOT(0, 1),
                PhaseInv(1),
                Hadamard(1),
            ]

            op_list_cy = op_list + [CY(0, 1)]

            cliffordsim_eng_cx = Engine(op_list_cx, nqubits)
            cliffordsim_eng_cy = Engine(op_list_cy, nqubits)

            cliffordsim_eng_cx.run()
            cliffordsim_eng_cy.run()

            self.assertEqual(
                cliffordsim_eng_cx.stabilizer_set,
                cliffordsim_eng_cy.stabilizer_set,
            )

    def test_swap_triple_cnot_equivalence(self):
        """
        Check for the following equivalence:

        SWAP_0_1 == CNOT_0_1 CNOT_1_0 CNOT_0_1
        """
        # some generic operations to generate a 2 qubit state
        base_ops = [Hadamard(0), X(1), CNOT(0, 1), Hadamard(1), Phase(0)]

        operation_list = base_ops + [SWAP(0, 1)]
        operation_list_2 = base_ops + [CNOT(0, 1), CNOT(1, 0), CNOT(0, 1)]
        nqubits = 2
        seed = 5

        cliffordsim_engine = Engine(operation_list, nqubits, seed)
        cliffordsim_engine_2 = Engine(operation_list_2, nqubits, seed)

        cliffordsim_engine.run()
        cliffordsim_engine_2.run()

        self.assertEqual(
            cliffordsim_engine.stabilizer_set, cliffordsim_engine_2.stabilizer_set
        )

    def test_engine_single_measurement(self):
        """
        Check that the measurements are recorded appropriately.

        M_0 == 0
        X_0 M_0 == 1
        H_0 M_0 == 0 OR 1
        """
        nqubits = 1
        seed = 5

        meas_op = Measurement(0)
        operation_list = [meas_op]

        cliffordsim_engine = Engine(operation_list, nqubits, seed)
        cliffordsim_engine.run()
        data_store_mea = cliffordsim_engine.data_store.measurements
        self.assertEqual(data_store_mea["time_step"], [0])
        # check measurement result
        self.assertEqual(
            # result from engine
            data_store_mea["0"][meas_op.label]["measurement_result"],
            # expected result
            0,
        )
        # check random flag
        self.assertEqual(
            # result from engine
            data_store_mea["0"][meas_op.label]["is_random"],
            # expected result
            False,
        )

        operation_list = [X(0), meas_op]

        cliffordsim_engine = Engine(operation_list, nqubits, seed)
        cliffordsim_engine.run()
        data_store_mea = cliffordsim_engine.data_store.measurements
        self.assertEqual(data_store_mea["time_step"], [1])
        # check measurement result
        self.assertEqual(
            # result from engine
            data_store_mea["1"][meas_op.label]["measurement_result"],
            # expected result
            1,
        )
        # check random flag
        self.assertEqual(
            # result from engine
            data_store_mea["1"][meas_op.label]["is_random"],
            # expected result
            False,
        )

        operation_list = [Hadamard(0), meas_op]

        cliffordsim_engine = Engine(operation_list, nqubits, seed)
        cliffordsim_engine.run()
        data_store_mea = cliffordsim_engine.data_store.measurements
        self.assertEqual(data_store_mea["time_step"], [1])
        self.assertEqual(
            list(data_store_mea.values())[1][meas_op.label]["is_random"],
            True,
        )

    def test_engine_multiple_measurement(self):
        """Check that the measurements are recorded appropriately.

        M_0 X_1 M_1 == 01
        M_0 H_1 M_1 == 0{0 OR 1}
        """
        nqubits = 2
        seed = 5

        meas_op = Measurement(0)
        meas_op_2 = Measurement(1)

        operation_list = [meas_op, X(1), meas_op_2]
        output_result = {
            "time_step": [0, 2],
            "0": {meas_op.label: {"measurement_result": 0, "is_random": False}},
            "2": {meas_op_2.label: {"measurement_result": 1, "is_random": False}},
        }

        cliffordsim_engine = Engine(operation_list, nqubits, seed)
        cliffordsim_engine.run()
        data_store_mea = cliffordsim_engine.data_store.measurements
        self.assertEqual(data_store_mea["time_step"], [0, 2])

        # already asserted that records will be on time steps [0,2]
        # test whether bitstring is expected
        measurement_bitstring = [
            data_store_mea[str(timestep)][m_op.label]["measurement_result"]
            for timestep, m_op in zip([0, 2], [meas_op, meas_op_2], strict=True)
        ]
        expected_bitstring = [
            output_result[str(timestep)][m_op.label]["measurement_result"]
            for timestep, m_op in zip([0, 2], [meas_op, meas_op_2], strict=True)
        ]
        self.assertEqual(measurement_bitstring, expected_bitstring)

        # check whether randomness result is expected
        measurement_randomflag = [
            data_store_mea[str(timestep)][m_op.label]["is_random"]
            for timestep, m_op in zip([0, 2], [meas_op, meas_op_2], strict=True)
        ]
        expected_randomflag = [
            output_result[str(timestep)][m_op.label]["is_random"]
            for timestep, m_op in zip([0, 2], [meas_op, meas_op_2], strict=True)
        ]
        self.assertEqual(measurement_randomflag, expected_randomflag)

        operation_list = [meas_op, Hadamard(1), meas_op_2]
        output_result = {
            "time_step": [0, 2],
            "0": {meas_op.label: {"measurement_result": 0, "is_random": False}},
            "2": {meas_op_2.label: {"measurement_result": 1, "is_random": True}},
        }

        cliffordsim_engine = Engine(operation_list, nqubits, seed)
        cliffordsim_engine.run()
        self.assertEqual(
            cliffordsim_engine.data_store.measurements["time_step"], [0, 2]
        )
        engine_result = list(cliffordsim_engine.data_store.measurements.values())
        self.assertEqual(
            engine_result[1][meas_op.label]["measurement_result"],
            0,
        )
        self.assertEqual(
            engine_result[1][meas_op.label]["is_random"],
            False,
        )
        self.assertEqual(
            engine_result[2][meas_op_2.label]["is_random"],
            True,
        )

    def test_non_existent_qubit(self):
        """Check if a proper error is raised when an Operation for a qubit that
        doesn't exist is specified."""
        nqubits = 2
        seed = 5

        operation_list = [Hadamard(0), CNOT(0, 1), X(2)]

        cliffordsim_engine = Engine(operation_list, nqubits, seed)
        with self.assertRaises(EngineRunError):
            cliffordsim_engine.run()

    def test_parallel_measurements(self):
        """As measurement operations are non-local, the order with respect to
        other operations that occur in the same time slice can affect the final
        output. This test should check that the ordering, for measurement
        operations + other gate operations are not violated."""
        nqubits = 2
        seed = 5

        operation_list = [
            Hadamard(0),
            CNOT(0, 1),
            Measurement(0),
            Measurement(1),
        ]
        operation_list_2 = [
            Hadamard(0),
            CNOT(0, 1),
            Measurement(1),
            Measurement(0),
        ]

        cliffordsim_engine = Engine(operation_list, nqubits, seed)
        cliffordsim_engine.run()

        cliffordsim_engine_2 = Engine(operation_list_2, nqubits, seed)
        cliffordsim_engine_2.run()

        # The order of measurements are important.
        # Thus, they should be in their own Moment.
        self.assertNotEqual(
            cliffordsim_engine.stabilizer_set, cliffordsim_engine_2.stabilizer_set
        )

        # Measurements should be parallelized in their own time slice
        cliffordsim_engines_list = [cliffordsim_engine, cliffordsim_engine_2]
        for each_cliffordsim_engine in cliffordsim_engines_list:
            for (
                each_set_p_ops
            ) in each_cliffordsim_engine.moment_queue.parallelized_operations:
                for each_op in each_set_p_ops:
                    if each_op.name == "Measurement":
                        self.assertEqual(len(each_set_p_ops), 1)

    def test_parallel_gate_and_measurement(self):
        """Checking for the same stabilizer set output.
        Both circuits are similar, but the ordering of the Operations created
        within CliffordSim for these 2 circuits are different. However, for this
        example, both stabilizer sets should be the same.
        """
        nqubits = 2
        seed = 5

        meas_op_0 = Measurement(0)
        meas_op_1 = Measurement(1)

        operation_list = [Hadamard(0), CNOT(0, 1), X(0), meas_op_1, meas_op_0]
        operation_list_2 = [
            Hadamard(0),
            CNOT(0, 1),
            meas_op_1,
            X(0),
            meas_op_0,
        ]

        cliffordsim_engine = Engine(operation_list, nqubits, seed, parallelize=True)
        parallel_op = cliffordsim_engine.moment_queue.parallelized_operations
        cliffordsim_engine.run()

        cliffordsim_engine_2 = Engine(operation_list_2, nqubits, seed, parallelize=True)
        parallel_op_2 = cliffordsim_engine_2.moment_queue.parallelized_operations
        cliffordsim_engine_2.run()

        # The order of measurements are important.
        # Thus, they should be in their own Moment.
        self.assertNotEqual(parallel_op, parallel_op_2)

        # In this example, both stabilizer sets should be the same
        self.assertEqual(
            cliffordsim_engine.stabilizer_set, cliffordsim_engine_2.stabilizer_set
        )

    def test_qubit_addition_simple(self):
        """Check if simply adding a qubit works as intended."""
        nqubits = 1
        seed = 5

        operation_list = [AddQubit(1)]

        cliffordsim_engine = Engine(operation_list, nqubits, seed)
        # make sure that this runs
        cliffordsim_engine.run()

        self.assertEqual(cliffordsim_engine.tableau_w_scratch.nqubits, nqubits + 1)

        stab_set = {"+_Z", "+Z_"}
        self.assertEqual(stab_set, cliffordsim_engine.tableau_w_scratch.stabilizer_set)

    def test_qubit_addition_complicated(self):
        """Check if simply adding a qubit works as intended."""
        nqubits = 2
        seed = 5

        operation_list = [
            Hadamard(0),
            CNOT(0, 1),
            AddQubit(1),
            Hadamard(1),
            Phase(1),
        ]

        cliffordsim_engine = Engine(operation_list, nqubits, seed)
        # make sure that this runs
        cliffordsim_engine.run()

        stab_set = {"+Z_Z", "+X_X", "+_Y_"}
        self.assertEqual(stab_set, cliffordsim_engine.tableau_w_scratch.stabilizer_set)

    def test_qubit_deletion_bell_pair(self):
        """Tests that after the deletion of an unused qubit the
        stabilizers are correct. The rest of the qubits are a bell pair."""
        nqubits = 3

        operation_list = [Hadamard(0), CNOT(0, 2), DeleteQubit(1)]

        cliffordsim_engine = Engine(operation_list, nqubits)
        cliffordsim_engine.run()

        stab_set = {"+ZZ", "+XX"}
        self.assertEqual(stab_set, cliffordsim_engine.tableau_w_scratch.stabilizer_set)

    def test_qubit_deletion_random(self):
        """Tests that after a random circuit and then deleting a qubit,
        the stabilizers and destabilizers generate the pauli group."""

        rand_gen = np.random.default_rng()
        nqubits = 5
        for _ in range(5):
            iter_seed = rand_gen.integers(10**5)
            operation_list = random_list_of_gates(nqubits, 20, iter_seed)

            # delete a qubit
            qubit_to_delete = rand_gen.integers(nqubits)
            operation_list += [DeleteQubit(qubit_to_delete)]

            cliffordsim_engine = Engine(operation_list, nqubits, iter_seed)
            cliffordsim_engine.run()

            self.assertTrue(
                is_tableau_valid(cliffordsim_engine.tableau_w_scratch.tableau)
            )

    def test_reset(self):
        """Tests that after a random circuit and a reset on qubit k, the operator that
        describes the reset state is a stabilizer.
        """
        nqubits = 5
        n_ops = 100

        operation_list = random_list_of_gates(nqubits, n_ops)

        # Go through all possible reset states
        for state in ["0", "1", "+", "-", "+i", "-i"]:
            # Go through all qubits
            for qub_to_reset in range(nqubits):
                # Reset one qubit
                op_list = operation_list + [Reset(qub_to_reset, state=state)]

                cliffordsim_engine = Engine(op_list, nqubits)
                cliffordsim_engine.run()

                match state:
                    case "0":
                        p, sign = "Z", "+"
                    case "1":
                        p, sign = "Z", "-"
                    case "+":
                        p, sign = "X", "+"
                    case "-":
                        p, sign = "X", "-"
                    case "+i":
                        p, sign = "Y", "+"
                    case "-i":
                        p, sign = "Y", "-"

                # Construct the stabilizer that should exist from the reset
                resulting_stab = (
                    sign + "_" * qub_to_reset + p + "_" * (nqubits - qub_to_reset - 1)
                )
                pauli_op = SignedPauliOp.from_string(resulting_stab)

                output_stabarray = StabArray(
                    cliffordsim_engine.tableau_w_scratch.stabilizer_array
                )

                pauli_op_exists = is_subset_of_stabarray(pauli_op, output_stabarray)

                self.assertTrue(pauli_op_exists)

    def test_udpatetableau(self):
        """Tests for Tableau Updating function during runtime."""
        # Check that the tableau states are exactly what they should be.
        # Check that the measurement results are as expected.
        ### Case 0: UpdateTableau Operation does not type check and auto convert.
        input_tableau = [[0, 1, 0], [1, 0, 0]]
        self.assertFalse(
            isinstance(UpdateTableau(input_tableau, False).tableau, np.ndarray)
        )
        self.assertTrue(
            np.array_equal(
                UpdateTableau(np.array(input_tableau), 5).tableau,
                np.array(input_tableau),
            )
        )

        ### Case 1: Update Tableau at the start. (i.e. Change initial tableau state)
        base_operation_list = []
        change_operation_list = [UpdateTableau(np.array(input_tableau))]
        test_eng = Engine(base_operation_list, nqubits=1)
        test_eng.run()
        change_eng = Engine(change_operation_list, nqubits=1)
        change_eng.run()

        self.assertFalse(
            np.array_equal(test_eng.tableau_w_scratch.tableau, np.array(input_tableau))
        )
        self.assertFalse(
            np.array_equal(
                test_eng.tableau_w_scratch.tableau,
                change_eng.tableau_w_scratch.tableau,
            )
        )

        ### Case 2: Update Tableau after some gate applications.
        base_operation_list = [UpdateTableau(np.array(input_tableau))]
        change_operation_list = [
            Hadamard(0),
            X(0),
            UpdateTableau(np.array(input_tableau)),
        ]
        test_eng = Engine(base_operation_list, nqubits=1)
        test_eng.run()
        change_eng = Engine(change_operation_list, nqubits=1)
        change_eng.run()

        self.assertTrue(
            np.array_equal(
                test_eng.tableau_w_scratch.tableau,
                change_eng.tableau_w_scratch.tableau,
            )
        )

        ### Case 3: Change of Tableau size at runtime using update.
        # (Print Warning Message. NFA.)
        base_operation_list = [UpdateTableau(np.array(input_tableau))]
        larger_tableau = [
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
        ]
        change_operation_list = [UpdateTableau(np.array(larger_tableau))]
        test_eng = Engine(base_operation_list, nqubits=1)
        test_eng.run()
        change_eng = Engine(change_operation_list, nqubits=1)

        with self.assertRaises(TableauSizeError):
            change_eng.run()

        ### Case 4: Tableau Validation works for right and wrong tableau.
        # (Should raise error if wrong tableau.)
        valid_operation_list = [UpdateTableau(np.array(input_tableau))]
        wrong_operation_list = [UpdateTableau(input_tableau)]
        test_eng = Engine(valid_operation_list, nqubits=1)
        test_eng.run()
        change_eng = Engine(wrong_operation_list, nqubits=1)

        with self.assertRaises(InvalidTableauError):
            change_eng.run()

        ### Case 5: What happens if wrong tableau but no validation(?)
        wrong_operation_list = [UpdateTableau(input_tableau, False)]
        change_eng = Engine(wrong_operation_list, nqubits=1)

        with self.assertRaises(InvalidTableauError):
            change_eng.run()

    def test_createclassicalregister_case1(self):
        """Tests that the classical register can be created successfully.

        The classical register is initialized with only 1 bit by default.
        """
        ### Case 1: Create 1 Classical Register (Default).
        base_operation_list = []
        new_operation_list = [CreateClassicalRegister("testreg_name")]
        test_eng = Engine(base_operation_list, nqubits=1)
        test_eng.run()
        new_eng = Engine(new_operation_list, nqubits=1)
        new_eng.run()

        # No Classical Register exists in the registry of the Engine
        self.assertTrue(not test_eng.registry)

        # At least 1 Classical Register exists in the registry
        self.assertEqual(len(new_eng.registry), 1)
        self.assertEqual(
            list(new_eng.registry.keys()), ["testreg_name"]
        )  # Register name check
        self.assertEqual(
            new_eng.registry["testreg_name"].no_of_bits, 1
        )  # Default no of bits in a register
        self.assertTrue(
            uuid.UUID(new_eng.registry["testreg_name"].bit_ids[0])
        )  # Valid UUID of automatic generated bit ID
        cbit_id = new_eng.registry["testreg_name"].bit_ids[0]
        self.assertEqual(
            new_eng.registry["testreg_name"].id_bit_reg[cbit_id], 0
        )  # Initial state of bit is 0

    def test_createclassicalregister_case2(self):
        """
        Tests custom initialization options, no_of_bits and bit_ids, of the classical
        register.

        The classical register should be initialized with no_of_bits, number of bits if
        it is specified.

        The CreateClassicalRegister Operation returns an error if the user tries to
        specify fewer bit IDs than the total number of bits in the register.

        The bits within the classical register should be initialized with bit IDs that
        are UUID version 4 compatible if no bit IDs were provided by the user.
        """
        ### Case 2: Create 1 "Custom" No of Bits Classical Register
        base_operation_list = [CreateClassicalRegister("testreg_name", no_of_bits=3)]
        custom_bit_ids = ["bit1", "bit2", "bit3"]
        new_operation_list = [
            CreateClassicalRegister(
                "newreg_name_w_ids", no_of_bits=3, bit_ids=custom_bit_ids
            )
        ]
        test_eng = Engine(base_operation_list, nqubits=1)
        test_eng.run()
        new_eng = Engine(new_operation_list, nqubits=1)
        new_eng.run()

        # The number of bit ids, if provided, must be equal to the number of bits of the Register.
        with self.assertRaises(ValueError):
            _ = [
                CreateClassicalRegister(
                    "errreg_name_w_ids", no_of_bits=3, bit_ids=["bit1"]
                )
            ]

        # Classical Register has only 3 bits. And bit IDs are all UUID valid.
        self.assertEqual(list(test_eng.registry.keys()), ["testreg_name"])
        self.assertEqual(test_eng.registry["testreg_name"].no_of_bits, 3)
        self.assertTrue(
            [
                uuid.UUID(each_uuid, version=4)
                for each_uuid in test_eng.registry["testreg_name"].bit_ids
            ]
        )
        self.assertEqual(test_eng.registry["testreg_name"].bit_reg, [0, 0, 0])

        # Classical Register has only 3 bits. And bit IDs are custom.
        self.assertEqual(list(new_eng.registry.keys()), ["newreg_name_w_ids"])
        self.assertEqual(new_eng.registry["newreg_name_w_ids"].no_of_bits, 3)
        self.assertEqual(
            list(new_eng.registry["newreg_name_w_ids"].bit_ids), custom_bit_ids
        )
        self.assertEqual(new_eng.registry["newreg_name_w_ids"].bit_reg, [0, 0, 0])

    def test_createclassicalregister_case3(self):
        """
        Tests that multiple classical registers, with different initialization options,
        can be created.
        """
        ### Case 3: Create Multiple Classical Registers
        base_operation_list = [CreateClassicalRegister("testreg_name")]
        custom_bit_ids = ["bit1", "bit2", "bit3"]
        new_operation_list = [
            CreateClassicalRegister("newreg_name"),
            CreateClassicalRegister(
                "newreg_name_w_ids", no_of_bits=3, bit_ids=custom_bit_ids
            ),
        ]
        test_eng = Engine(base_operation_list, nqubits=1)
        test_eng.run()
        new_eng = Engine(new_operation_list, nqubits=1)
        new_eng.run()

        # Only 1 Classical Register in test engine.
        self.assertEqual(len(test_eng.registry), 1)
        self.assertEqual(test_eng.registry["testreg_name"].no_of_bits, 1)
        self.assertTrue(
            [
                uuid.UUID(each_uuid, version=4)
                for each_uuid in test_eng.registry["testreg_name"].bit_ids
            ]
        )
        self.assertEqual(test_eng.registry["testreg_name"].bit_reg, [0])

        # 2 Classical Registers in the registry.
        self.assertEqual(len(new_eng.registry), 2)
        self.assertEqual(
            list(new_eng.registry.keys()), ["newreg_name", "newreg_name_w_ids"]
        )
        self.assertEqual(
            [each_register.no_of_bits for each_register in new_eng.registry.values()],
            [1, 3],
        )
        self.assertEqual(
            list(new_eng.registry["newreg_name_w_ids"].bit_ids), custom_bit_ids
        )

    def test_createclassicalregister_case3a(self):
        """
        Tests that an error is raised if a classical register of the same name is
        created.
        """
        ### Case 3a: Adding 2 Classical Registers with the same name. A printout should
        # be observed.
        custom_bit_ids = ["bit1", "bit2", "bit3"]
        err_operation_list = [
            CreateClassicalRegister("newreg_name"),
            CreateClassicalRegister(
                "newreg_name", no_of_bits=3, bit_ids=custom_bit_ids
            ),
        ]
        err_eng = Engine(err_operation_list, nqubits=1)
        with self.assertRaises(ClassicalRegisterError):
            err_eng.run()

    def test_recordclassicalregister_case1(self):
        """Test that checks the classical register is recorded properly within the
        DataStore.

        NOTE: Classical registers are stored as ClassicalRegisterSnapshot
        within the DataStore and have to be "restored".
        """
        ### Case 1: Record an existing ClassicalRegister.
        base_operation_list = [CreateClassicalRegister("testreg_name")]
        new_operation_list = [
            CreateClassicalRegister("testreg_name"),
            RecordClassicalRegister("testreg_name"),
        ]
        test_eng = Engine(base_operation_list, nqubits=1)
        test_eng.run()
        new_eng = Engine(new_operation_list, nqubits=1)
        new_eng.run()

        # No Classical Register exists in the DataStore of the Engine. Only exists in
        # Engine registry.
        self.assertTrue(len(test_eng.registry), 1)
        self.assertTrue(test_eng.data_store.cr_records == {"time_step": []})

        # Classical Register found in both DataStore and Engine registry.
        self.assertEqual(len(new_eng.registry), 1)
        self.assertEqual(list(new_eng.registry.keys()), ["testreg_name"])

        # Check for Snapshot of Classical Register in DataStore
        creg_snapshot = new_eng.data_store.cr_records["1"]["testreg_name"]
        self.assertTrue(isinstance(creg_snapshot, ClassicalRegisterSnapshot))
        # Restore Classical Register Snapshot should be equal to the Classical Register
        # at the time the snapshot was taken.
        self.assertEqual(
            new_eng.registry["testreg_name"], ClassicalRegister.restore(creg_snapshot)
        )

    def test_recordclassicalregister_case2(self):
        """
        Test checks that, when multiple classical registers are present, the correctly
        name classical register is recorded.
        """
        ### Case 2: Record the right register
        test_operation_list = [
            CreateClassicalRegister("c2_testreg_name"),
            CreateClassicalRegister("c2_testreg_name_1"),
            RecordClassicalRegister("c2_testreg_name"),
        ]
        test_eng = Engine(test_operation_list, nqubits=1)
        test_eng.run()

        # 2 Registers in registry. Only 1 snapshot in DataStore.
        self.assertEqual(len(test_eng.registry), 2)
        self.assertEqual(
            list(test_eng.registry.keys()), ["c2_testreg_name", "c2_testreg_name_1"]
        )

        self.assertEqual(
            len(test_eng.data_store.cr_records), 2
        )  # time_step is one of the keys in this dict.
        creg_snapshot = test_eng.data_store.cr_records["2"]["c2_testreg_name"]
        self.assertTrue(isinstance(creg_snapshot, ClassicalRegisterSnapshot))

    def test_recordclassicalregister_case3(self):
        """
        Test checks that, when multiple classical registers are present, multiple
        registers can be recorded and stored within the DataStore.
        """
        ### Case 3: Multiple Recorded Registers
        test_operation_list = [
            CreateClassicalRegister("c3_testreg_name"),
            CreateClassicalRegister("c3_testreg_name_1"),
            RecordClassicalRegister("c3_testreg_name"),
            RecordClassicalRegister("c3_testreg_name_1"),
            RecordClassicalRegister("c3_testreg_name"),
        ]
        test_eng = Engine(test_operation_list, nqubits=1)
        test_eng.run()

        # 2 Registers in registry. 3 snapshots in DataStore.
        self.assertEqual(len(test_eng.registry), 2)
        self.assertEqual(
            list(test_eng.registry.keys()), ["c3_testreg_name", "c3_testreg_name_1"]
        )

        self.assertEqual(
            len(test_eng.data_store.cr_records), 4
        )  # time_step is one of the keys in this dict.
        creg_snapshot = test_eng.data_store.cr_records["2"]["c3_testreg_name"]
        creg_snapshot_1 = test_eng.data_store.cr_records["3"]["c3_testreg_name_1"]
        creg_snapshot_2 = test_eng.data_store.cr_records["4"]["c3_testreg_name"]
        self.assertTrue(isinstance(creg_snapshot, ClassicalRegisterSnapshot))
        self.assertTrue(isinstance(creg_snapshot_1, ClassicalRegisterSnapshot))
        self.assertTrue(isinstance(creg_snapshot_2, ClassicalRegisterSnapshot))

        # pylint: disable=fixme
        ### TODO: Case 4: Proper state recording for register mutated over time.

    def test_createclassicalnot(self):
        """
        Tests that a classical NOT operation is performed successfully on a classical
        register.
        """
        ### Case 1: NOT Operation performed successfully on the specified bit, selected
        # by order. We also check that the change in 1 register does not affect another.
        base_operation_list = [
            CreateClassicalRegister("testreg_name", no_of_bits=3),
            CreateClassicalRegister("ghostreg_name"),
        ]
        new_operation_list = [
            CreateClassicalRegister("testreg_name", no_of_bits=3),
            CreateClassicalRegister("ghostreg_name"),
            ClassicalNOT("testreg_name", bit_order=0),
            ClassicalNOT("testreg_name", bit_order=2),
        ]
        test_eng = Engine(base_operation_list, nqubits=1)
        test_eng.run()
        new_eng = Engine(new_operation_list, nqubits=1)
        new_eng.run()

        self.assertEqual(test_eng.registry["testreg_name"].bit_reg, [0, 0, 0])
        self.assertEqual(test_eng.registry["ghostreg_name"].bit_reg, [0])
        self.assertEqual(new_eng.registry["testreg_name"].bit_reg, [1, 0, 1])
        self.assertEqual(new_eng.registry["ghostreg_name"].bit_reg, [0])

        ### Case 2: NOT Operation performed successfully on the specified bit, selected
        # by bit ID.
        new_operation_list = [
            CreateClassicalRegister(
                "c2_testreg_name", no_of_bits=3, bit_ids=["bit_1", "bit_2", "bit_3"]
            ),
            CreateClassicalRegister("c2_ghostreg_name"),
            ClassicalNOT("c2_testreg_name", bit_id="bit_1"),
            ClassicalNOT("c2_testreg_name", bit_id="bit_2"),
            ClassicalNOT("c2_ghostreg_name", bit_order=0),
        ]
        new_eng = Engine(new_operation_list, nqubits=1)
        new_eng.run()

        self.assertEqual(new_eng.registry["c2_testreg_name"].bit_reg, [1, 1, 0])
        self.assertEqual(new_eng.registry["c2_ghostreg_name"].bit_reg, [1])

        ### Case 3: Error raised when bit defined for Operation does not exist.
        # Case 3a: With bit order
        err_operation_list = [
            CreateClassicalRegister(
                "c3a_testreg_name", no_of_bits=1, bit_ids=["bit_1"]
            ),
            ClassicalNOT("c3a_testreg_name", bit_order=1),
        ]

        with self.assertRaises(ClassicalOperationError) as err_msg:
            err_eng = Engine(err_operation_list, nqubits=1)
            err_eng.run()

        self.assertEqual(
            str(err_msg.exception),
            "The selected bit_order, 1, does not exist or is not valid in"
            " c3a_testreg_name. There are only 1 bits in the register.",
        )

        # Case 3b: With bit ID
        err_operation_list = [
            CreateClassicalRegister(
                "c3b_testreg_name", no_of_bits=1, bit_ids=["bit_1"]
            ),
            ClassicalNOT("c3b_testreg_name", bit_id="bit_2"),
        ]

        with self.assertRaises(ClassicalOperationError) as err_msg:
            err_eng = Engine(err_operation_list, nqubits=1)
            err_eng.run()

        self.assertEqual(
            str(err_msg.exception),
            "The selected bit_id, bit_2, does not exist or is not valid in"
            f" c3b_testreg_name. The only available bit IDs in this register are: {['bit_1']}",
        )


class TestCliffordSimEngineClassicalOp(unittest.TestCase):
    """
    Tests for the CliffordSim Engine when performing classical operations on classical
    registers.
    """

    def setUp(self):
        """
        Creates an operation list that initializes a classical register with 3 bits with
        the first bit flipped.
        """
        self.operation_list = [
            CreateClassicalRegister("testreg_name", no_of_bits=3),
            ClassicalNOT("testreg_name", bit_order=0),
        ]

    def test_classicalor_case1(self):
        """
        Test checks that the output of the OR operation is written on the right bit in
        the same register.
        By default, if the name of the output classical register is not provided, it is
        assumed that the output of the operation is written within the same classical
        register.
        """
        self.operation_list.append(
            ClassicalOR("testreg_name", input_bit_order=[0, 1], write_bit_order=2)
        )
        eng = Engine(self.operation_list, nqubits=1)
        eng.run()

        self.assertEqual(eng.registry["testreg_name"].bit_reg, [1, 0, 1])

    def test_classicalor_case1a(self):
        """
        Test checks that the output from the classical operation can be written over an
        input bit.
        """
        self.operation_list.append(
            ClassicalOR("testreg_name", input_bit_order=[0, 1], write_bit_order=1)
        )
        eng = Engine(self.operation_list, nqubits=1)
        eng.run()

        self.assertEqual(eng.registry["testreg_name"].bit_reg, [1, 1, 0])

    def test_classicalor_case1b(self):
        """
        Test checks that for a complicated set of operations, the output on the
        classical register is as expected.

        Expected evolution: (# is the snapshot)
        [0, 0, 0] -> [1, 0, 0] -> [1, 1, 0] -#-> [1, 1, 0] -> [1, 1, 1]
        """
        extend_operation_list = [
            ClassicalOR("testreg_name", input_bit_order=[0, 1], write_bit_order=1),
            RecordClassicalRegister("testreg_name"),
            ClassicalOR("testreg_name", input_bit_order=[0, 2], write_bit_order=1),
            ClassicalOR("testreg_name", input_bit_order=[0, 1], write_bit_order=2),
        ]
        self.operation_list.extend(extend_operation_list)
        eng = Engine(self.operation_list, nqubits=1)
        eng.run()

        self.assertEqual(eng.registry["testreg_name"].bit_reg, [1, 1, 1])

        mid_run_ss = eng.data_store.cr_records["3"]["testreg_name"]
        restore_cr = ClassicalRegister.restore(mid_run_ss)

        self.assertEqual(restore_cr.bit_reg, [1, 1, 0])

    def test_classicalor_case2a(self):
        """Test checks that an error is raised if the target bit does not exist.
        Non-existant write bit order.
        """
        self.operation_list.append(
            ClassicalOR("testreg_name", input_bit_order=[0, 1], write_bit_order=3)
        )
        eng = Engine(self.operation_list, nqubits=3)

        with self.assertRaises(ClassicalOperationError) as err_msg:
            eng.run()

        self.assertEqual(
            str(err_msg.exception),
            "The selected bit_order, 3, does not exist or is not valid in testreg_name."
            " There are only 3 bits in the register.",
        )

    def test_classicalor_case2b(self):
        """
        Test checks that an error is raised if the target bit does not exist.
        Non-existant write bit ID.
        """

        self.operation_list.append(
            ClassicalOR("testreg_name", input_bit_order=[0, 1], write_bit_id="bit_2")
        )
        eng = Engine(self.operation_list, nqubits=3)

        with self.assertRaises(ClassicalOperationError) as err_msg:
            eng.run()

        self.assertEqual(
            str(err_msg.exception),
            "The selected bit_id, bit_2, does not exist or is not valid in "
            "testreg_name. The only available bit IDs in this register are: "
            f"{eng.registry['testreg_name'].bit_ids}",
        )

    def test_classicalor_case2c(self):
        """Test checks that an error is raised if the target bit does not exist.
        Non-existant output register.
        """
        self.operation_list.append(
            ClassicalOR(
                "testreg_name",
                input_bit_order=[0, 1],
                output_reg_name="non-existant_reg",
                write_bit_id=0,
            )
        )
        eng = Engine(self.operation_list, nqubits=3)

        with self.assertRaises(ClassicalRegisterError) as err_msg:
            eng.run()

        self.assertEqual(
            str(err_msg.exception),
            "An error has occured when trying to select the register, non-existant_reg.",
        )

    def test_classicalor_case2d(self):
        """Test checks that an error is raised if the target bit does not exist.
        Non-existant input bit order.
        """
        self.operation_list.append(
            ClassicalOR("testreg_name", input_bit_order=[4, 5], write_bit_order=0)
        )
        eng = Engine(self.operation_list, nqubits=3)

        with self.assertRaises(ClassicalOperationError) as err_msg:
            eng.run()

        self.assertEqual(
            str(err_msg.exception),
            "The selected bit_order, 4, does not exist or is not valid in testreg_name."
            " There are only 3 bits in the register.",
        )

    def test_classicalor_case2e(self):
        """Test checks that an error is raised if the target bit does not exist.
        Non-existant input bit IDs.
        """
        self.operation_list.append(
            ClassicalOR(
                "testreg_name", input_bit_ids=["bit_1", "bit_2"], write_bit_order=0
            )
        )
        eng = Engine(self.operation_list, nqubits=3)

        with self.assertRaises(ClassicalOperationError) as err_msg:
            eng.run()

        self.assertEqual(
            str(err_msg.exception),
            "The selected bit_id, bit_1, does not exist or is not valid in "
            "testreg_name. The only available bit IDs in this "
            f"register are: {eng.registry['testreg_name'].bit_ids}",
        )

    def test_classicalor_case3(self):
        """Test checks that we can record the output on another classical register."""
        operation_list = [
            CreateClassicalRegister(
                "testreg_name", no_of_bits=2, bit_ids=["bit_1", "bit_2"]
            ),
            CreateClassicalRegister("secondreg_name"),
            ClassicalNOT("testreg_name", bit_id="bit_1"),
            ClassicalOR(
                "testreg_name",
                input_bit_ids=["bit_1", "bit_2"],
                output_reg_name="secondreg_name",
                write_bit_order=0,
            ),
        ]
        eng = Engine(operation_list, nqubits=3)
        eng.run()

        self.assertEqual(eng.registry["testreg_name"].bit_reg, [1, 0])
        self.assertEqual(eng.registry["testreg_name"].id_bit_reg["bit_1"], 1)
        self.assertEqual(eng.registry["secondreg_name"].bit_reg, [1])

    def test_classicaland_case1(self):
        """Test checks that the output of the AND operation is written on the
        right bit in the same register.
        By default, if the name of the output classical register is not provided, it is
        assumed that the output of the operation is written within the same classical
        register.
        """
        extend_operation_list = [
            ClassicalNOT("testreg_name", bit_order=1),
            ClassicalAND("testreg_name", input_bit_order=[0, 1], write_bit_order=2),
        ]
        self.operation_list.extend(extend_operation_list)
        eng = Engine(self.operation_list, nqubits=1)
        eng.run()

        self.assertEqual(eng.registry["testreg_name"].bit_reg, [1, 1, 1])

    def test_classicaland_case1a(self):
        """
        Test checks that the output from the classical operation can be written over an
        input bit.
        """
        self.operation_list.append(
            ClassicalAND("testreg_name", input_bit_order=[0, 1], write_bit_order=0)
        )
        eng = Engine(self.operation_list, nqubits=1)
        eng.run()

        self.assertEqual(eng.registry["testreg_name"].bit_reg, [0, 0, 0])

    def test_classicaland_case1b(self):
        """Test checks that for a complicated set of operations, the output on the
        classical register is as expected.

        Expected evolution: (# is the snapshot)
        [0, 0, 0] -> [1, 0, 0] -> [1, 0, 0] -#-> [1, 0, 0] -> [0, 0, 0]
        """
        extend_operation_list = [
            ClassicalAND("testreg_name", input_bit_order=[0, 1], write_bit_order=1),
            RecordClassicalRegister("testreg_name"),
            ClassicalAND("testreg_name", input_bit_order=[0, 2], write_bit_order=1),
            ClassicalAND("testreg_name", input_bit_order=[0, 1], write_bit_order=0),
        ]
        self.operation_list.extend(extend_operation_list)
        eng = Engine(self.operation_list, nqubits=1)
        eng.run()

        self.assertEqual(eng.registry["testreg_name"].bit_reg, [0, 0, 0])

        mid_run_ss = eng.data_store.cr_records["3"]["testreg_name"]
        restore_cr = ClassicalRegister.restore(mid_run_ss)

        self.assertEqual(restore_cr.bit_reg, [1, 0, 0])

    def test_classicaland_case2a(self):
        """
        Test checks that an error is raised if the target bit does not exist.
        Non-existant write bit order.
        """
        self.operation_list.append(
            ClassicalAND("testreg_name", input_bit_order=[0, 1], write_bit_order=3)
        )
        eng = Engine(self.operation_list, nqubits=3)

        with self.assertRaises(ClassicalOperationError) as err_msg:
            eng.run()

        self.assertEqual(
            str(err_msg.exception),
            "The selected bit_order, 3, does not exist or is not valid in testreg_name."
            " There are only 3 bits in the register.",
        )

    def test_classicaland_case2b(self):
        """
        Test checks that an error is raised if the target bit does not exist.
        Non-existant write bit ID.
        """
        self.operation_list.append(
            ClassicalAND("testreg_name", input_bit_order=[0, 1], write_bit_id="bit_2")
        )
        eng = Engine(self.operation_list, nqubits=3)

        with self.assertRaises(ClassicalOperationError) as err_msg:
            eng.run()

        self.assertEqual(
            str(err_msg.exception),
            "The selected bit_id, bit_2, does not exist or is not valid in testreg_name."
            " The only available bit IDs in this register"
            f" are: {eng.registry['testreg_name'].bit_ids}",
        )

    def test_classicaland_case2c(self):
        """Test checks that an error is raised if the target bit does not exist.
        Non-existant output register.
        """
        self.operation_list.append(
            ClassicalAND(
                "testreg_name",
                input_bit_order=[0, 1],
                output_reg_name="non-existant_reg",
                write_bit_id=0,
            )
        )
        eng = Engine(self.operation_list, nqubits=3)

        with self.assertRaises(ClassicalRegisterError) as err_msg:
            eng.run()

        self.assertEqual(
            str(err_msg.exception),
            "An error has occured when trying to select the register, non-existant_reg.",
        )

    def test_classicaland_case2d(self):
        """
        Test checks that an error is raised if the target bit does not exist.
        Non-existant input bit order.
        """
        self.operation_list.append(
            ClassicalAND("testreg_name", input_bit_order=[4, 5], write_bit_order=0)
        )
        eng = Engine(self.operation_list, nqubits=3)

        with self.assertRaises(ClassicalOperationError) as err_msg:
            eng.run()

        self.assertEqual(
            str(err_msg.exception),
            "The selected bit_order, 4, does not exist or is not valid in testreg_name."
            " There are only 3 bits in the register.",
        )

    def test_classicaland_case2e(self):
        """
        Test checks that an error is raised if the target bit does not exist.
        Non-existant input bit IDs.
        """
        self.operation_list.append(
            ClassicalAND(
                "testreg_name", input_bit_ids=["bit_1", "bit_2"], write_bit_order=0
            )
        )
        eng = Engine(self.operation_list, nqubits=3)

        with self.assertRaises(ClassicalOperationError) as err_msg:
            eng.run()

        self.assertEqual(
            str(err_msg.exception),
            "The selected bit_id, bit_1, does not exist or is not valid in"
            " testreg_name. The only available bit IDs in this register"
            f" are: {eng.registry['testreg_name'].bit_ids}",
        )

    def test_classicaland_case3(self):
        """
        Test checks that we can record the output on another classical register.
        """
        operation_list = [
            CreateClassicalRegister(
                "testreg_name", no_of_bits=2, bit_ids=["bit_1", "bit_2"]
            ),
            CreateClassicalRegister("secondreg_name"),
            ClassicalNOT("testreg_name", bit_id="bit_1"),
            ClassicalNOT("testreg_name", bit_id="bit_2"),
            ClassicalAND(
                "testreg_name",
                input_bit_ids=["bit_1", "bit_2"],
                output_reg_name="secondreg_name",
                write_bit_order=0,
            ),
        ]
        eng = Engine(operation_list, nqubits=3)
        eng.run()

        self.assertEqual(eng.registry["testreg_name"].bit_reg, [1, 1])
        self.assertEqual(eng.registry["testreg_name"].id_bit_reg["bit_1"], 1)
        self.assertEqual(eng.registry["secondreg_name"].bit_reg, [1])


class TestCliffordSimEngineControlledOp(unittest.TestCase):
    """
    Tests for the CliffordSim Engine when performing controlled operations on
    classical registers.
    """

    # pylint: disable=no-member
    def test_controlled_gate_op(self):
        """Test checks that gate operations are properly applied when controlled."""
        operation_list = [
            CreateClassicalRegister("testreg_name", 2),
            ClassicalNOT("testreg_name", 0),  # Apply NOT on bit 0
            X(0).with_ccontrol("testreg_name", 0),  # Apply Controlled X on qubit 0
            X(1).with_ccontrol("testreg_name", 1),
            CNOT(0, 2).with_ccontrol("testreg_name", 0),
            Measurement(0, label="meas_0"),
            Measurement(1, label="meas_1"),
            Measurement(2, label="meas_2"),
        ]
        eng = Engine(operation_list, nqubits=3)
        eng.run()

        # 4 entries, including time_step
        self.assertEqual(4, len(eng.data_store.measurements))
        # Expected Output
        self.assertEqual(
            1, eng.data_store.measurements["5"]["meas_0"]["measurement_result"]
        )
        self.assertEqual(
            0, eng.data_store.measurements["6"]["meas_1"]["measurement_result"]
        )
        self.assertEqual(
            1, eng.data_store.measurements["7"]["meas_2"]["measurement_result"]
        )
        # Expected Classical Register State
        self.assertEqual(eng.registry["testreg_name"].bit_reg, [1, 0])

    def test_controlled_class_op(self):
        """Test checks that classical operations can be controlled."""
        operation_list = [
            CreateClassicalRegister("testreg_name", 2),
            ClassicalNOT("testreg_name", 0),  # Apply NOT on bit 0
            ClassicalNOT("testreg_name", 1).with_ccontrol(
                "testreg_name", 0
            ),  # Apply NOT on bit 1 only if bit 9 is 1.
            X(0).with_ccontrol("testreg_name", 1),
            Measurement(0, label="meas_0"),
        ]
        eng = Engine(operation_list, nqubits=1)
        eng.run()

        # 2 entries, including time_step
        self.assertEqual(2, len(eng.data_store.measurements))
        # Expected Output
        self.assertEqual(
            1, eng.data_store.measurements["4"]["meas_0"]["measurement_result"]
        )
        # Expected Classical Register State
        self.assertEqual(eng.registry["testreg_name"].bit_reg, [1, 1])

    def test_controlled_measurement_op(self):
        """
        Test checks that the measurement operation can successfully write onto the
        Classical Register and apply a controlled operation with that output bit.
        """
        operation_list = [
            CreateClassicalRegister("testreg_name", 1),
            Hadamard(0),
            Measurement(
                0, label="meas_0", reg_name="testreg_name", bit_order=0
            ),  # Measurement qubit 0 and write the output to bit 0 of testreg_name
            X(0).with_ccontrol("testreg_name", 0),
            Measurement(0, label="meas_0_e"),
        ]
        # Repeat for more randomness in measurement output.
        for _ in range(10):
            eng = Engine(operation_list, nqubits=1)
            eng.run()

            # 3 entries, including time_step
            self.assertEqual(3, len(eng.data_store.measurements))
            # Expected Output
            self.assertEqual(
                0, eng.data_store.measurements["4"]["meas_0_e"]["measurement_result"]
            )

    def test_controlled_measurement_op_2(self):
        """Test checks that the measurement operation can be classically controlled."""
        operation_list = [
            CreateClassicalRegister("testreg_name", 1),
            Hadamard(0),
            Measurement(
                0, label="meas_0", reg_name="testreg_name", bit_order=0
            ),  # Measurement qubit 0 and write the output to bit 0 of testreg_name
            Measurement(1, label="meas_0_e").with_ccontrol("testreg_name", 0),
        ]
        # Repeat for more randomness in measurement output.
        for _ in range(10):
            eng = Engine(operation_list, nqubits=2)
            eng.run()

            # 3 entries, including time_step
            if len(eng.data_store.measurements) == 3:
                # Measured 1
                self.assertEqual(
                    1, eng.data_store.measurements["2"]["meas_0"]["measurement_result"]
                )
                # Classically Controlled Measurement Triggers as Expected Output
                self.assertEqual(
                    0,
                    eng.data_store.measurements["3"]["meas_0_e"]["measurement_result"],
                )
            elif len(eng.data_store.measurements) == 2:
                # Measured 0, no second measurement
                self.assertEqual(
                    0, eng.data_store.measurements["2"]["meas_0"]["measurement_result"]
                )

    def test_controlled_datamanipulation_op(self):
        """
        Test checks that the measurement operation can successfully write onto the
        Classical Register and apply a controlled operation with that output bit.
        """
        operation_list = [
            CreateClassicalRegister("testreg_name", 2),
            ClassicalNOT("testreg_name", 0),
            CreateClassicalRegister("createdreg_name", 1).with_ccontrol(
                "testreg_name", 0
            ),
            CreateClassicalRegister("createdreg2_name", 1).with_ccontrol(
                "testreg_name", 1
            ),
        ]
        eng = Engine(operation_list, nqubits=1)
        eng.run()

        # Only 2 registers exists
        self.assertEqual(list(eng.registry.keys()), ["testreg_name", "createdreg_name"])

    def test_controlled_resize_op(self):
        """Test checks that the new qubit is initialized correctly when conditioned."""
        operation_list = [
            CreateClassicalRegister("testreg_name", 1),
            ClassicalNOT("testreg_name", 0),
            AddQubit(1).with_ccontrol("testreg_name", 0),
            X(1),
            Measurement(0, label="meas_0"),
            Measurement(1, label="meas_1"),
        ]
        eng = Engine(operation_list, nqubits=1)
        eng.run()

        # A new qubit was created and both qubits have the expected output
        self.assertEqual(3, len(eng.data_store.measurements))
        self.assertEqual(
            0, eng.data_store.measurements["4"]["meas_0"]["measurement_result"]
        )
        self.assertEqual(
            1, eng.data_store.measurements["5"]["meas_1"]["measurement_result"]
        )


if __name__ == "__main__":
    unittest.main()
