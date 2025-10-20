"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

import unittest
from uuid import uuid4

import numpy as np

from loom.cliffordsim.engine import Engine
from loom.cliffordsim.pauli_frame import PauliFrame
from loom.cliffordsim.operations import (
    Hadamard,
    Phase,
    CNOT,
    CZ,
    X,
    Y,
    Z,
    Measurement,
    CreatePauliFrame,
    RecordPauliFrame,
    AddQubit,
    DeleteQubit,
)

# pylint: disable=import-error, wrong-import-order
from utilities import random_list_of_gates


class TestCliffordSimPauliFrame(unittest.TestCase):
    """
    We test the PauliFrame class and its associated functionality.
    """

    def test_basic_pf(self):
        "Test basic PF functionalities"
        pf0_id = str(uuid4())
        pf1_id = str(uuid4())
        pf0 = PauliFrame([0, 1], [1, 0], id=pf0_id)
        pf1 = PauliFrame.from_string("ZX", id=pf1_id)
        pf2 = PauliFrame.from_string("ZX")
        pf3 = PauliFrame([1, 1], [0, 0])

        # test len
        self.assertEqual(len(pf0), 2)

        # test equality
        self.assertTrue(pf0 == pf1 == pf2 != pf3)

        # test that ids are generated randomly
        self.assertTrue(pf0.id != pf1.id != pf2.id != pf3.id)

        # test that id is passed correctly
        self.assertEqual(pf0.id, pf0_id)
        self.assertEqual(pf1.id, pf1_id)

    def test_identity_pf_propagation(self):
        "Test identity PF behavior"
        nqubits = 5

        pff_identity = PauliFrame(
            np.zeros(nqubits), np.zeros(nqubits), direction="forward"
        )
        pfb_identity = PauliFrame(
            np.zeros(nqubits), np.zeros(nqubits), direction="backward"
        )
        print(pff_identity.id)
        print(pfb_identity.id)

        operation_list = random_list_of_gates(nqubits, 20)
        operation_list = (
            [
                CreatePauliFrame(pff_identity),
                RecordPauliFrame(pfb_identity),
            ]
            + operation_list
            + [
                RecordPauliFrame(pff_identity),
                CreatePauliFrame(pfb_identity),
            ]
        )

        cliffordsim_engine = Engine(operation_list, nqubits)

        cliffordsim_engine.run()
        # The recorded PF should all be identity
        pff_record = cliffordsim_engine.data_store.pf_records["forward"]
        for time_step in pff_record["time_step"]:
            for result in pff_record[str(time_step)].values():
                self.assertEqual(result["initial_pauli_frame"], pff_identity)
                self.assertEqual(result["recorded_pauli_frame"], pff_identity)
        pfb_record = cliffordsim_engine.data_store.pf_records["backward"]
        for time_step in pfb_record["time_step"]:
            for result in pfb_record[str(time_step)].values():
                self.assertEqual(result["initial_pauli_frame"], pfb_identity)
                self.assertEqual(result["recorded_pauli_frame"], pfb_identity)

    def test_pf_basic_operations(self):  # pylint: disable=too-many-statements
        "Test basic PF behavior"
        nqubits = 3

        i_pf = PauliFrame.from_string("XIZ")
        cpf_op = CreatePauliFrame(i_pf)
        rpf_op = RecordPauliFrame(i_pf)

        operation_list = []

        # apply phase gates everywhere
        for i in range(nqubits):
            operation_list.append(Phase(i))
        cat_engine = Engine([cpf_op] + operation_list + [rpf_op], nqubits)
        cat_engine.run()
        pff_records = cat_engine.data_store.pf_records["forward"]
        record_step = str(pff_records["time_step"][0])
        o_pf = pff_records[record_step][i_pf.id]["recorded_pauli_frame"]

        self.assertEqual(i_pf, pff_records[record_step][i_pf.id]["initial_pauli_frame"])
        self.assertEqual(o_pf, PauliFrame.from_string("YIZ"))

        # apply hadamards everywhere
        for i in range(nqubits):
            operation_list.append(Hadamard(i))
        cat_engine = Engine([cpf_op] + operation_list + [rpf_op], nqubits)
        cat_engine.run()
        pff_records = cat_engine.data_store.pf_records["forward"]
        record_step = str(pff_records["time_step"][0])
        o_pf = pff_records[record_step][i_pf.id]["recorded_pauli_frame"]

        self.assertEqual(i_pf, pff_records[record_step][i_pf.id]["initial_pauli_frame"])
        self.assertEqual(o_pf, PauliFrame.from_string("YIX"))

        # apply cnot
        operation_list.append(CNOT(0, 1))
        cat_engine = Engine([cpf_op] + operation_list + [rpf_op], nqubits)
        cat_engine.run()
        pff_records = cat_engine.data_store.pf_records["forward"]
        record_step = str(pff_records["time_step"][0])
        o_pf = pff_records[record_step][i_pf.id]["recorded_pauli_frame"]

        self.assertEqual(i_pf, pff_records[record_step][i_pf.id]["initial_pauli_frame"])
        self.assertEqual(o_pf, PauliFrame.from_string("YXX"))

        # apply CZ
        operation_list.append(CZ(0, 2))
        cat_engine = Engine([cpf_op] + operation_list + [rpf_op], nqubits)
        cat_engine.run()
        pff_records = cat_engine.data_store.pf_records["forward"]
        record_step = str(pff_records["time_step"][0])
        o_pf = pff_records[record_step][i_pf.id]["recorded_pauli_frame"]

        self.assertEqual(i_pf, pff_records[record_step][i_pf.id]["initial_pauli_frame"])
        self.assertEqual(o_pf, PauliFrame.from_string("XXY"))

        # apply cnot
        operation_list.append(CNOT(1, 2))
        cat_engine = Engine([cpf_op] + operation_list + [rpf_op], nqubits)
        cat_engine.run()
        pff_records = cat_engine.data_store.pf_records["forward"]
        record_step = str(pff_records["time_step"][0])
        o_pf = pff_records[record_step][i_pf.id]["recorded_pauli_frame"]

        self.assertEqual(i_pf, pff_records[record_step][i_pf.id]["initial_pauli_frame"])
        self.assertEqual(o_pf, PauliFrame.from_string("XYZ"))

        # apply paulis -> nothing changes
        operation_list.append(X(0))
        operation_list.append(Z(1))
        operation_list.append(Y(2))
        cat_engine = Engine([cpf_op] + operation_list + [rpf_op], nqubits)
        cat_engine.run()
        pff_records = cat_engine.data_store.pf_records["forward"]
        record_step = str(pff_records["time_step"][0])
        o_pf = pff_records[record_step][i_pf.id]["recorded_pauli_frame"]

        self.assertEqual(i_pf, pff_records[record_step][i_pf.id]["initial_pauli_frame"])
        self.assertEqual(o_pf, PauliFrame.from_string("XYZ"))

        # apply measurement
        operation_list.append(Measurement(1))
        cat_engine = Engine([cpf_op] + operation_list + [rpf_op], nqubits)
        cat_engine.run()
        pff_records = cat_engine.data_store.pf_records["forward"]
        record_step = str(pff_records["time_step"][0])
        o_pf = pff_records[record_step][i_pf.id]["recorded_pauli_frame"]

        self.assertEqual(i_pf, pff_records[record_step][i_pf.id]["initial_pauli_frame"])
        self.assertEqual(o_pf, PauliFrame.from_string("XYZ"))

    def test_random_circuit(self):
        "Test random PF behavior against cliffordsim"
        nqubits = 5
        ngates = 100

        for _ in range(3):
            op_list = random_list_of_gates(nqubits, ngates)

            # propagate ZIIII using PF propagation
            i_pf = PauliFrame.from_string("Z" + "I" * (nqubits - 1))
            cpf_op = CreatePauliFrame(i_pf)
            rpf_op = RecordPauliFrame(i_pf)
            cat_eng = Engine([cpf_op] + op_list + [rpf_op], nqubits)

            # run cliffordsim clifford simulator
            cat_eng.run()
            pff_records = cat_eng.data_store.pf_records["forward"]
            record_step = str(pff_records["time_step"][0])
            o_pf = pff_records[record_step][i_pf.id]["recorded_pauli_frame"]

            # the output should match the first row of the stabilizers
            # from the tableau
            z0_out = cat_eng.tableau_w_scratch.z_stabilizers[0, :]
            x0_out = cat_eng.tableau_w_scratch.x_stabilizers[0, :]
            pf_from_clifsim = PauliFrame(x0_out, z0_out)

            self.assertEqual(o_pf, pf_from_clifsim)

    def test_pf_output(self):
        """
        A function to test the Pauliframe string output
        """
        i_pf = PauliFrame.from_string("XIZ")
        str_pf = repr(i_pf)
        self.assertEqual("PauliFrame: ", str_pf[0:12])

    def test_random_circuit_back_propagation(self):
        "Test random PF back propagation against forward propagation"
        nqubits = 5
        ngates = 100

        # start with a ZZZZZ pauli frame
        i_pf = PauliFrame.from_string("Z" * nqubits)

        for _ in range(3):
            op_list = random_list_of_gates(nqubits, ngates)
            op_list = [CreatePauliFrame(i_pf)] + op_list + [RecordPauliFrame(i_pf)]
            cat_eng = Engine(op_list, nqubits)

            # propagate i_pf using PF propagation
            cat_eng.run()
            pff_records = cat_eng.data_store.pf_records["forward"]
            record_step = str(pff_records["time_step"][0])
            o_pf = pff_records[record_step][i_pf.id]["recorded_pauli_frame"]

            # backpropagate the output PF
            o_pf.id = str(uuid4())  # create a new id for the output PF
            o_pf.direction = "backward"  # make sure it propagates backward
            op_list = [RecordPauliFrame(o_pf)] + op_list + [CreatePauliFrame(o_pf)]
            cat_eng = Engine(op_list, nqubits)
            cat_eng.run()  # Don't forget to re-run the engine
            pfb_records = cat_eng.data_store.pf_records["backward"]
            record_step = str(pfb_records["time_step"][0])
            oo_pf = pfb_records[record_step][o_pf.id]["recorded_pauli_frame"]

            # assert that the output of the backpropagation is the same as the input
            self.assertEqual(oo_pf, i_pf)

            # to mix things up, the next input is the output of the previous
            i_pf = o_pf
            i_pf.id = str(uuid4())  # create a new id for the new input PF
            i_pf.direction = "forward"  # make sure it propagates forward

    def test_invalid_input_propagate_pauli_frame(self):
        "Test invalid input for propagate_pauli_frame"
        nqubits = 5

        # test invalid input to PauliFrame
        with self.assertRaises(ValueError):
            pf = PauliFrame.from_string(
                pauli_string="This is not a PauliFrame",
                id=str(uuid4()),
                direction="forward",
            )

        # test invalid direction
        with self.assertRaises(ValueError) as cm:
            pf = PauliFrame.from_string("Z" * nqubits, direction="Wrong direction")
        self.assertEqual(
            cm.exception.args[0],
            "Invalid direction 'Wrong direction'. Must be 'forward' or 'backward'.",
        )

        # test invalid input to CreatePauliFrame and RecordPauliFrame
        with self.assertRaises(TypeError) as cm:
            op = CreatePauliFrame(
                pauli_frame="This is not a PauliFrame",
            )
        self.assertEqual(
            cm.exception.args[0],
            "Invalid PauliFrame 'This is not a PauliFrame'. Must be of type 'PauliFrame'.",
        )

        # test RecordPauliFrame on non-existing PauliFrame
        with self.assertRaises(ValueError) as cm:
            op = RecordPauliFrame(
                PauliFrame.from_string("Z" * nqubits),
            )
            _ = Engine([op], nqubits)
        self.assertEqual(
            cm.exception.args[0],
            "RecordPauliFrame operations must be preceded by a CreatePauliFrame operation.",
        )

        # test non-unique ID for two CreatePauliFrame operations
        with self.assertRaises(ValueError) as cm:
            pf = PauliFrame.from_string("Z" * nqubits)
            _ = Engine([CreatePauliFrame(pf), CreatePauliFrame(pf)], nqubits)
        self.assertEqual(
            cm.exception.args[0],
            "CreatePauliFrame operations must have a unique PauliFrame id.",
        )

        # test wrong size PauliFrame
        with self.assertRaises(ValueError) as cm:
            pf = PauliFrame.from_string("Z" * (nqubits - 1))
            _ = Engine([CreatePauliFrame(pf)], nqubits)
        self.assertEqual(
            cm.exception.args[0],
            f"Wrong size for the PauliFrame "
            f"{pf.id}. It has size {len(pf.x)}. It must have "
            f"the same length as the number of qubits in the system "
            f"({nqubits}). Make sure that you take into "
            f"account resize operations.",
        )

        # test wrong size PauliFrame after AddQubit operations
        with self.assertRaises(ValueError) as cm:
            pf = PauliFrame.from_string("Z" * nqubits)
            cops = [AddQubit(0), CreatePauliFrame(pf)]
            _ = Engine(cops, nqubits)
        self.assertEqual(
            cm.exception.args[0],
            f"Wrong size for the PauliFrame "
            f"{pf.id}. It has size {len(pf.x)}. It must have "
            f"the same length as the number of qubits in the system "
            f"({nqubits+1}). Make sure that you take into "
            f"account resize operations.",
        )
        # test wrong size PauliFrame after DeleteQubit operations
        with self.assertRaises(ValueError) as cm:
            pf = PauliFrame.from_string("Z" * nqubits)
            cops = [DeleteQubit(0), CreatePauliFrame(pf)]
            _ = Engine(cops, nqubits)
        self.assertEqual(
            cm.exception.args[0],
            f"Wrong size for the PauliFrame "
            f"{pf.id}. It has size {len(pf.x)}. It must have "
            f"the same length as the number of qubits in the system "
            f"({nqubits-1}). Make sure that you take into "
            f"account resize operations.",
        )

    def test_pf_after_qubit_resize(self):
        """
        Test PF propagation after qubit resize operations.
        """
        nqubits = 3

        i_pf = PauliFrame.from_string("XIZ")
        cpf_op = CreatePauliFrame(i_pf)
        rpf_op = RecordPauliFrame(i_pf)

        operation_list = []

        # add a qubit at position 1
        operation_list.append(AddQubit(1))
        # now the PF should be XIIZ
        # expected_pf_after_add = PauliFrame.from_string("XIIZ")

        # apply hadamards everywhere
        for i in range(nqubits + 1):
            operation_list.append(Hadamard(i))
        # now the PF should be ZIIX
        # expected_pf_after_hadamard = PauliFrame.from_string("ZIIX")

        # delete qubit at position 2
        operation_list.append(DeleteQubit(3))
        # now the PF should be ZII
        expected_pf_after_delete = PauliFrame.from_string("ZII")

        cat_engine = Engine([cpf_op] + operation_list + [rpf_op], nqubits)
        cat_engine.run()
        pff_records = cat_engine.data_store.pf_records["forward"]
        record_step = str(pff_records["time_step"][0])
        o_pf = pff_records[record_step][i_pf.id]["recorded_pauli_frame"]

        self.assertEqual(i_pf, pff_records[record_step][i_pf.id]["initial_pauli_frame"])
        self.assertEqual(o_pf, expected_pf_after_delete)


if __name__ == "__main__":
    unittest.main()
