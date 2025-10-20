"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

import unittest

from loom.eka import (
    Circuit,
    Channel,
    Lattice,
    Eka,
    Block,
    Stabilizer,
    PauliOperator,
)
from loom.eka.operations.code_operation import (
    ResetAllDataQubits,
    MeasureBlockSyndromes,
    MeasureLogicalZ,
)
from loom.interpreter import interpret_eka, Syndrome, Detector
from loom.executor.eka_circuit_to_qasm_converter import convert_circuit_to_qasm


class TestEkaCircuitToQasmConverter(unittest.TestCase):
    """
    Test the EkaCircuitToQasmConverter class.
    """

    def setUp(self):
        # setup the eka circuit and InterpretationStep
        distance = 3
        lattice = Lattice.linear((distance + distance + 1,))

        base_position = (1,)
        # pylint: disable=duplicate-code
        initial_rep_block = Block(
            unique_label="q1",
            stabilizers=tuple(
                Stabilizer(
                    pauli="ZZ",
                    data_qubits=(
                        (base_position[0] + i, 0),
                        (base_position[0] + i + 1, 0),
                    ),
                    ancilla_qubits=((base_position[0] + i, 1),),
                )
                for i in range(distance - 1)
            ),
            logical_x_operators=(
                PauliOperator(
                    "XXX", tuple((base_position[0] + i, 0) for i in range(distance))
                ),
            ),
            logical_z_operators=(PauliOperator("Z", ((base_position[0], 0),)),),
        )
        meas_block_and_meas_log = [
            ResetAllDataQubits(initial_rep_block.unique_label, state="0"),
            MeasureBlockSyndromes(initial_rep_block.unique_label, n_cycles=2),
            MeasureLogicalZ(initial_rep_block.unique_label),
        ]

        eka = Eka(
            lattice, blocks=[initial_rep_block], operations=meas_block_and_meas_log
        )
        self.final_step = interpret_eka(eka)
        self.final_circuit = self.final_step.final_circuit

    def test_circuit_to_qasm(self):
        """
        Check Circuit to QASM string conversion
        """
        repetition_qasm_true = (
            "OPENQASM 3.0;\n"
            'include "stdgates.inc";\n'
            "qubit[5] data_qreg;\n"
            "qubit[0] anc_qreg;\n"
            "bit[3] data_creg0;\n"
            "bit[2] anc_creg0;\n"
            "bit[2] anc_creg1;\n"
            "reset data_qreg[0];\n"
            "reset data_qreg[2];\n"
            "reset data_qreg[4];\n"
            "barrier data_qreg[0], data_qreg[1], data_qreg[2], data_qreg[3], data_qreg[4], ;\n"
            "reset data_qreg[1];\n"
            "reset data_qreg[3];\n"
            "barrier data_qreg[0], data_qreg[1], data_qreg[2], data_qreg[3], data_qreg[4], ;\n"
            "h data_qreg[1];\n"
            "h data_qreg[3];\n"
            "barrier data_qreg[0], data_qreg[1], data_qreg[2], data_qreg[3], data_qreg[4], ;\n"
            "cz data_qreg[1], data_qreg[0];\n"
            "cz data_qreg[3], data_qreg[2];\n"
            "barrier data_qreg[0], data_qreg[1], data_qreg[2], data_qreg[3], data_qreg[4], ;\n"
            "cz data_qreg[1], data_qreg[2];\n"
            "cz data_qreg[3], data_qreg[4];\n"
            "barrier data_qreg[0], data_qreg[1], data_qreg[2], data_qreg[3], data_qreg[4], ;\n"
            "h data_qreg[1];\n"
            "h data_qreg[3];\n"
            "barrier data_qreg[0], data_qreg[1], data_qreg[2], data_qreg[3], data_qreg[4], ;\n"
            "anc_creg0[1] = measure data_qreg[1];\n"
            "anc_creg0[0] = measure data_qreg[3];\n"
            "barrier data_qreg[0], data_qreg[1], data_qreg[2], data_qreg[3], data_qreg[4], ;\n"
            "reset data_qreg[1];\n"
            "reset data_qreg[3];\n"
            "barrier data_qreg[0], data_qreg[1], data_qreg[2], data_qreg[3], data_qreg[4], ;\n"
            "h data_qreg[1];\n"
            "h data_qreg[3];\n"
            "barrier data_qreg[0], data_qreg[1], data_qreg[2], data_qreg[3], data_qreg[4], ;\n"
            "cz data_qreg[1], data_qreg[0];\n"
            "cz data_qreg[3], data_qreg[2];\n"
            "barrier data_qreg[0], data_qreg[1], data_qreg[2], data_qreg[3], data_qreg[4], ;\n"
            "cz data_qreg[1], data_qreg[2];\n"
            "cz data_qreg[3], data_qreg[4];\n"
            "barrier data_qreg[0], data_qreg[1], data_qreg[2], data_qreg[3], data_qreg[4], ;\n"
            "h data_qreg[1];\n"
            "h data_qreg[3];\n"
            "barrier data_qreg[0], data_qreg[1], data_qreg[2], data_qreg[3], data_qreg[4], ;\n"
            "anc_creg1[1] = measure data_qreg[1];\n"
            "anc_creg1[0] = measure data_qreg[3];\n"
            "barrier data_qreg[0], data_qreg[1], data_qreg[2], data_qreg[3], data_qreg[4], ;\n"
            "data_creg0[2] = measure data_qreg[0];\n"
            "data_creg0[1] = measure data_qreg[2];\n"
            "data_creg0[0] = measure data_qreg[4];\n"
            "barrier data_qreg[0], data_qreg[1], data_qreg[2], data_qreg[3], data_qreg[4], ;\n"
        )

        converted_dict = convert_circuit_to_qasm(self.final_circuit)
        repetition_qasm_converted = converted_dict["qasm_circuit"]
        syndrome_mapping = converted_dict["eka_to_qasm_syndromes"]
        detector_mapping = converted_dict["eka_to_qasm_detectors"]
        self.assertEqual(repr(repetition_qasm_true), repr(repetition_qasm_converted))
        self.assertEqual(len(syndrome_mapping), 0)
        self.assertEqual(len(detector_mapping), 0)

    def test_not_implemented_error(self):
        """
        Check that the function raises an error when the input circuit
        operation is not implemented.
        """
        channel = Channel(type="quantum", label="(1,0)")
        hw_gate = Circuit("Hello world", channels=[channel])
        circ = Circuit("my_circuit", circuit=[hw_gate])
        with self.assertRaises(NotImplementedError) as cm:
            _ = convert_circuit_to_qasm(circ)
        self.assertEqual(
            str(cm.exception),
            "Operation hello world not supported",
        )

    def test_syndrome_and_detector_mapping(self):
        """Test that the conversion process constructed the correct
        syndrome and detector mappings to respective measurements
        in qasm circuit
        """
        true_syndrome_mapping = {
            Syndrome(
                stabilizer="df585dfd-9408-4aeb-8cdb-23ffcbb1c49b",
                measurements=(),
                block="q1",
                round=-1,
                corrections=(),
                uuid="4e012c0c-7a34-4e66-8601-8993904578b9",
            ): [],
            Syndrome(
                stabilizer="83beeac5-123c-42f6-9e62-ab1ff884eaa0",
                measurements=(),
                block="q1",
                round=-1,
                corrections=(),
                uuid="f77084d4-190e-4b34-90d1-0d9b6987e848",
            ): [],
            Syndrome(
                stabilizer="df585dfd-9408-4aeb-8cdb-23ffcbb1c49b",
                measurements=(("c_(1, 1)", 0),),
                block="q1",
                round=0,
                corrections=(),
                uuid="11d57ba4-d08c-4232-81f6-c64378e39f61",
            ): [("anc_creg0", 0)],
            Syndrome(
                stabilizer="83beeac5-123c-42f6-9e62-ab1ff884eaa0",
                measurements=(("c_(2, 1)", 0),),
                block="q1",
                round=0,
                corrections=(),
                uuid="e7fa326e-c463-46e1-83e2-4b248b348e3c",
            ): [("anc_creg0", 1)],
            Syndrome(
                stabilizer="df585dfd-9408-4aeb-8cdb-23ffcbb1c49b",
                measurements=(("c_(1, 1)", 1),),
                block="q1",
                round=1,
                corrections=(),
                uuid="616977f3-206b-4508-9f35-6c7b9e7756b7",
            ): [("anc_creg1", 0)],
            Syndrome(
                stabilizer="83beeac5-123c-42f6-9e62-ab1ff884eaa0",
                measurements=(("c_(2, 1)", 1),),
                block="q1",
                round=1,
                corrections=(),
                uuid="63f58b68-d95a-4fb4-a98a-4dcbed9a4075",
            ): [("anc_creg1", 1)],
            Syndrome(
                stabilizer="df585dfd-9408-4aeb-8cdb-23ffcbb1c49b",
                measurements=(("c_(1, 0)", 0), ("c_(2, 0)", 0)),
                block="q1",
                round=2,
                corrections=(),
                uuid="7f1f21cf-4c68-4a23-92ce-b9510986860a",
            ): [("data_creg0", 0), ("data_creg0", 1)],
            Syndrome(
                stabilizer="83beeac5-123c-42f6-9e62-ab1ff884eaa0",
                measurements=(("c_(2, 0)", 0), ("c_(3, 0)", 0)),
                block="q1",
                round=2,
                corrections=(),
                uuid="e87f83ee-2a14-4c45-bcd3-12c222035669",
            ): [("data_creg0", 1), ("data_creg0", 2)],
        }

        syndromes_list = list(true_syndrome_mapping.keys())

        true_detector_mapping = {
            Detector(syndromes=[syndromes_list[0], syndromes_list[2]]): [
                ("anc_creg0", 0)
            ],
            Detector(syndromes=[syndromes_list[2], syndromes_list[4]]): [
                ("anc_creg0", 0),
                ("anc_creg1", 0),
            ],
            Detector(syndromes=[syndromes_list[1], syndromes_list[3]]): [
                ("anc_creg0", 1)
            ],
            Detector(syndromes=[syndromes_list[3], syndromes_list[5]]): [
                ("anc_creg0", 1),
                ("anc_creg1", 1),
            ],
            Detector(syndromes=[syndromes_list[4], syndromes_list[6]]): [
                ("anc_creg1", 0),
                ("data_creg0", 0),
                ("data_creg0", 1),
            ],
            Detector(syndromes=[syndromes_list[5], syndromes_list[7]]): [
                ("anc_creg1", 1),
                ("data_creg0", 1),
                ("data_creg0", 2),
            ],
        }

        converted_dict = convert_circuit_to_qasm(
            self.final_circuit,
            self.final_step.syndromes,
            self.final_step.detectors,
        )
        generated_syndrome_mapping = converted_dict["eka_to_qasm_syndromes"]
        generated_detector_mapping = converted_dict["eka_to_qasm_detectors"]

        # compare measurements from eka to measurements in qasm for syndromes
        true_meas_eka_to_meas_qasm_syndromes = {
            eka_syndrome.measurements: qasm_locs
            for eka_syndrome, qasm_locs in true_syndrome_mapping.items()
        }
        gen_meas_eka_to_meas_qasm_syndromes = {
            eka_syndrome.measurements: qasm_locs
            for eka_syndrome, qasm_locs in generated_syndrome_mapping.items()
        }

        # compare measurements from eka to measurements in qasm for detectors
        true_meas_eka_to_meas_qasm_detectors = {
            frozenset(syn.measurements for syn in eka_detector.syndromes): qasm_locs
            for eka_detector, qasm_locs in true_detector_mapping.items()
        }
        gen_meas_eka_to_meas_qasm_detectors = {
            frozenset(syn.measurements for syn in eka_detector.syndromes): qasm_locs
            for eka_detector, qasm_locs in generated_detector_mapping.items()
        }

        self.assertEqual(
            true_meas_eka_to_meas_qasm_syndromes, gen_meas_eka_to_meas_qasm_syndromes
        )
        self.assertEqual(
            true_meas_eka_to_meas_qasm_detectors, gen_meas_eka_to_meas_qasm_detectors
        )


if __name__ == "__main__":
    unittest.main()
