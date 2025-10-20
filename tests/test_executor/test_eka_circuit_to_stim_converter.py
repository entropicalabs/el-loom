"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

import random
import unittest
import stim

from loom.executor import (
    EkaCircuitToStimConverter,
    ErrorType,
    CircuitErrorModel,
    ApplicationMode,
    noise_annotated_stim_circuit,
    AsymmetricDepolarizeCEM,
    HomogeneousTimeDependentCEM,
    HomogeneousTimeIndependentCEM,
)
from loom.interpreter import InterpretationStep, interpret_eka
from loom.eka import (
    Eka,
    Lattice,
    Stabilizer,
    PauliOperator,
    Block,
    Circuit,
    Channel,
    ChannelType,
    SyndromeCircuit,
)
from loom.eka.operations.code_operation import MeasureBlockSyndromes

# pylint: disable=duplicate-code
gate_set = {
    "x",
    "y",
    "z",
    "h",
    "identity",
    "hadamard",
    "i",
    "phase",
    "phaseinv",
    "cnot",
    "cx",
    "cz",
    "cy",
    "swap",
    "reset",
    "reset_0",
    "reset_1",
    "reset_+",
    "reset_-",
    "reset_+i",
    "reset_-i",
    "measure_z",
    "measure_x",
    "measure_y",
    "measurement",
}


# pylint: disable=too-many-instance-attributes, duplicate-code, no-member, too-many-lines, eval-used
class TestEkaCircuitToStimConverter(unittest.TestCase):
    """
    Test the conversion functionality of the class,
    and verify that conversion happens with consistency
    """

    def setUp(self):
        # first simple stim circuit to test functionality
        self.base_step = InterpretationStep()

        self.gate_durations = {gate: random.uniform(0, 0.01) for gate in gate_set}

        # Repetition Code block
        self.linear_lattice = Lattice.linear((10,))
        distance = 3
        self.rep_code = Block(
            unique_label="q1",
            stabilizers=tuple(
                Stabilizer(
                    pauli="XX",
                    data_qubits=(
                        (i, 0),
                        (i + 1, 0),
                    ),
                    ancilla_qubits=((i, 1),),
                )
                for i in range(distance - 1)
            ),
            logical_x_operators=(PauliOperator("X", ((0, 0),)),),
            logical_z_operators=(
                PauliOperator("ZZZ", tuple((i, 0) for i in range(distance))),
            ),
        )

        self.block_qubits_ordered_rep = sorted([(0, 1), (0, 0), (1, 1), (2, 0), (1, 0)])
        self.stim_qubits_rep = [
            "QUBIT_COORDS(-0.5, 0.5) 0",
            "QUBIT_COORDS(0, 0) 1",
            "QUBIT_COORDS(0.5, 0.5) 2",
            "QUBIT_COORDS(1, 0) 3",
            "QUBIT_COORDS(1.5, 0.5) 4",
        ]
        self.meas_block_op_rep = MeasureBlockSyndromes(
            self.rep_code.unique_label, n_cycles=3
        )
        self.eka_rep = Eka(
            self.linear_lattice,
            blocks=[self.rep_code],
            operations=[self.meas_block_op_rep],
        )
        self.interpreted_eka_rep = interpret_eka(self.eka_rep)

        # Mock MeasureStabilizerSyndrome Operation for Rotated Surface Code
        self.square_2d_lattice = Lattice.square_2d((10, 10))
        channels = {
            "a": [
                Channel(type=ChannelType.QUANTUM, label="a0"),
            ],
            "q": [Channel(type=ChannelType.QUANTUM, label=f"d{i}") for i in range(4)],
            "c": [Channel(type=ChannelType.CLASSICAL, label="c0")],
        }
        xxxx_circuit = SyndromeCircuit(
            name="xxxx",
            pauli="XXXX",
            circuit=Circuit(
                name="xxxx",
                circuit=(
                    (Circuit("Reset_0", channels=channels["a"]),),
                    (Circuit("H", channels=channels["a"]),),
                    (Circuit("CX", channels=[channels["a"][0], channels["q"][0]]),),
                    (Circuit("CX", channels=[channels["a"][0], channels["q"][1]]),),
                    (Circuit("CX", channels=[channels["a"][0], channels["q"][2]]),),
                    (Circuit("CX", channels=[channels["a"][0], channels["q"][3]]),),
                    (Circuit("H", channels=channels["a"]),),
                    (
                        Circuit(
                            "Measurement", channels=[channels["a"][0], channels["c"][0]]
                        ),
                    ),
                ),
                channels=channels["q"] + channels["a"] + channels["c"],
            ),
        )
        zzzz_circuit = SyndromeCircuit(
            name="zzzz",
            pauli="ZZZZ",
            circuit=Circuit(
                name="zzzz",
                circuit=(
                    (
                        Circuit(
                            "Reset_0",
                            channels=channels["a"],
                        ),
                    ),
                    (Circuit("H", channels=channels["a"]),),
                    (Circuit("CZ", channels=[channels["a"][0], channels["q"][0]]),),
                    (Circuit("CZ", channels=[channels["a"][0], channels["q"][1]]),),
                    (Circuit("CZ", channels=[channels["a"][0], channels["q"][2]]),),
                    (Circuit("CZ", channels=[channels["a"][0], channels["q"][3]]),),
                    (Circuit("H", channels=channels["a"]),),
                    (
                        Circuit(
                            "Measurement",
                            channels=[channels["a"][0], channels["c"][0]],
                        ),
                    ),
                ),
                channels=channels["q"] + channels["a"] + channels["c"],
            ),
        )
        left_xx_circuit = SyndromeCircuit(
            pauli="XX",
            name="left_xx",
            circuit=Circuit(
                name="left_xx",
                circuit=(
                    (
                        Circuit(
                            "Reset_0",
                            channels=channels["a"],
                        ),
                    ),
                    (Circuit("H", channels=channels["a"]),),
                    (Circuit("CX", channels=[channels["a"][0], channels["q"][0]]),),
                    (Circuit("CX", channels=[channels["a"][0], channels["q"][1]]),),
                    (),
                    (),
                    (Circuit("H", channels=channels["a"]),),
                    (
                        Circuit(
                            "Measurement",
                            channels=[channels["a"][0], channels["c"][0]],
                        ),
                    ),
                ),
                channels=channels["q"][:2] + channels["a"] + channels["c"],
            ),
        )
        right_xx_circuit = SyndromeCircuit(
            pauli="XX",
            name="right_xx",
            circuit=Circuit(
                name="right_xx",
                circuit=(
                    (
                        Circuit(
                            "Reset_0",
                            channels=channels["a"],
                        ),
                    ),
                    (Circuit("H", channels=channels["a"]),),
                    (),
                    (),
                    (Circuit("CX", channels=[channels["a"][0], channels["q"][0]]),),
                    (Circuit("CX", channels=[channels["a"][0], channels["q"][1]]),),
                    (Circuit("H", channels=channels["a"]),),
                    (
                        Circuit(
                            "Measurement",
                            channels=[channels["a"][0], channels["c"][0]],
                        ),
                    ),
                ),
                channels=channels["q"][:2] + channels["a"] + channels["c"],
            ),
        )
        top_zz_circuit = SyndromeCircuit(
            pauli="ZZ",
            name="top_zz",
            circuit=Circuit(
                name="top_zz",
                circuit=(
                    (
                        Circuit(
                            "Reset_0",
                            channels=channels["a"],
                        ),
                    ),
                    (Circuit("H", channels=channels["a"]),),
                    (),
                    (),
                    (Circuit("CZ", channels=[channels["a"][0], channels["q"][0]]),),
                    (Circuit("CZ", channels=[channels["a"][0], channels["q"][1]]),),
                    (Circuit("H", channels=channels["a"]),),
                    (
                        Circuit(
                            "Measurement",
                            channels=[channels["a"][0], channels["c"][0]],
                        ),
                    ),
                ),
                channels=channels["q"][:2] + channels["a"] + channels["c"],
            ),
        )
        bottom_zz_circuit = SyndromeCircuit(
            name="bottom_zz",
            pauli="ZZ",
            circuit=Circuit(
                name="bottom_zz",
                circuit=(
                    (
                        Circuit(
                            "Reset_0",
                            channels=channels["a"],
                        ),
                    ),
                    (Circuit("H", channels=channels["a"]),),
                    (Circuit("CZ", channels=[channels["a"][0], channels["q"][0]]),),
                    (Circuit("CZ", channels=[channels["a"][0], channels["q"][1]]),),
                    (),
                    (),
                    (Circuit("H", channels=channels["a"]),),
                    (
                        Circuit(
                            "Measurement",
                            channels=[channels["a"][0], channels["c"][0]],
                        ),
                    ),
                ),
                channels=channels["q"][:2] + channels["a"] + channels["c"],
            ),
        )
        # pylint: disable=duplicate-code
        rsc_stabilizers = (
            Stabilizer(
                "ZZZZ",
                ((1, 0, 0), (0, 0, 0), (1, 1, 0), (0, 1, 0)),
                ancilla_qubits=((1, 1, 1),),
            ),
            Stabilizer(
                "ZZZZ",
                ((2, 1, 0), (1, 1, 0), (2, 2, 0), (1, 2, 0)),
                ancilla_qubits=((2, 2, 1),),
            ),
            Stabilizer(
                "XXXX",
                ((1, 1, 0), (1, 2, 0), (0, 1, 0), (0, 2, 0)),
                ancilla_qubits=((1, 2, 1),),
            ),
            Stabilizer(
                "XXXX",
                ((2, 0, 0), (2, 1, 0), (1, 0, 0), (1, 1, 0)),
                ancilla_qubits=((2, 1, 1),),
            ),
            Stabilizer(
                "XX",
                ((0, 0, 0), (0, 1, 0)),
                ancilla_qubits=((0, 1, 1),),
            ),
            Stabilizer(
                "XX",
                ((2, 1, 0), (2, 2, 0)),
                ancilla_qubits=((3, 2, 1),),
            ),
            Stabilizer(
                "ZZ",
                ((2, 0, 0), (1, 0, 0)),
                ancilla_qubits=((2, 0, 1),),
            ),
            Stabilizer(
                "ZZ",
                ((1, 2, 0), (0, 2, 0)),
                ancilla_qubits=((1, 3, 1),),
            ),
        )
        self.rot_surf_code_1 = Block(
            unique_label="q1",
            stabilizers=rsc_stabilizers,
            logical_x_operators=(
                PauliOperator("XXX", ((0, 0, 0), (1, 0, 0), (2, 0, 0))),
            ),
            logical_z_operators=(
                PauliOperator("ZZZ", ((0, 0, 0), (0, 1, 0), (0, 2, 0))),
            ),
            syndrome_circuits=[
                left_xx_circuit,
                right_xx_circuit,
                top_zz_circuit,
                bottom_zz_circuit,
                zzzz_circuit,
                xxxx_circuit,
            ],
            stabilizer_to_circuit={
                rsc_stabilizers[0].uuid: zzzz_circuit.uuid,
                rsc_stabilizers[1].uuid: zzzz_circuit.uuid,
                rsc_stabilizers[2].uuid: xxxx_circuit.uuid,
                rsc_stabilizers[3].uuid: xxxx_circuit.uuid,
                rsc_stabilizers[4].uuid: left_xx_circuit.uuid,
                rsc_stabilizers[5].uuid: right_xx_circuit.uuid,
                rsc_stabilizers[6].uuid: top_zz_circuit.uuid,
                rsc_stabilizers[7].uuid: bottom_zz_circuit.uuid,
            },
        )
        # self.rot_surf_code_1 = RotatedSurfaceCode.create(
        #     dx=3, dz=3, lattice=self.square_2d_lattice, unique_label="q1"
        # )
        self.meas_block_op = MeasureBlockSyndromes(
            self.rot_surf_code_1.unique_label, n_cycles=3
        )
        self.eka_rsc = Eka(
            self.square_2d_lattice,
            blocks=[self.rot_surf_code_1],
            operations=[self.meas_block_op],
        )

        self.block_qubits_ordered_rsc = [
            (0, 1, 1),
            (0, 0, 0),
            (0, 1, 0),
            (0, 2, 0),
            (1, 1, 1),
            (1, 2, 1),
            (1, 3, 1),
            (1, 0, 0),
            (1, 1, 0),
            (1, 2, 0),
            (2, 0, 1),
            (2, 1, 1),
            (2, 2, 1),
            (2, 0, 0),
            (2, 1, 0),
            (2, 2, 0),
            (3, 2, 1),
        ]
        self.stim_qubits_rsc = [
            "QUBIT_COORDS(0, 1) 0",
            "QUBIT_COORDS(0.5, 0.5) 1",
            "QUBIT_COORDS(0.5, 1.5) 2",
            "QUBIT_COORDS(0.5, 2.5) 3",
            "QUBIT_COORDS(1, 1) 4",
            "QUBIT_COORDS(1, 2) 5",
            "QUBIT_COORDS(1, 3) 6",
            "QUBIT_COORDS(1.5, 0.5) 7",
            "QUBIT_COORDS(1.5, 1.5) 8",
            "QUBIT_COORDS(1.5, 2.5) 9",
            "QUBIT_COORDS(2, 0) 10",
            "QUBIT_COORDS(2, 1) 11",
            "QUBIT_COORDS(2, 2) 12",
            "QUBIT_COORDS(2.5, 0.5) 13",
            "QUBIT_COORDS(2.5, 1.5) 14",
            "QUBIT_COORDS(2.5, 2.5) 15",
            "QUBIT_COORDS(3, 2) 16",
        ]
        self.interpreted_eka_rsc = interpret_eka(self.eka_rsc)
        self.converter = EkaCircuitToStimConverter()

    def test_converter_mappings(self):
        """
        Tests whether the correct mapping is generated between:
        1) The block qubits and the stim qubits for the conversion
        2) The Channel and stim qubit instructions
        """

        # Test full workflow for Rotated Surface Code
        eka_stim_dict = self.converter.eka_coords_to_stim_qubit_instruction_mapper(
            self.interpreted_eka_rsc.final_circuit
        )

        # stim qubit instructions as type stim.CircuitInstruction object
        stim_qubit_instructions = [
            stim.Circuit(qubit)[0] for qubit in self.stim_qubits_rsc
        ]

        expected_block_stim_dict = dict(
            zip(self.block_qubits_ordered_rsc, stim_qubit_instructions, strict=True)
        )

        # verify converter.blocks_to_stim_qubits_mapper
        self.assertEqual(expected_block_stim_dict, eka_stim_dict)

        crd_qubit_channels = [
            channel
            for channel in self.interpreted_eka_rsc.final_circuit.channels
            if channel.is_quantum()
        ]
        crd_channels = sorted(
            [
                channel
                for channel in crd_qubit_channels
                if eval(channel.label) in self.block_qubits_ordered_rsc
            ],
            key=lambda x: (
                eval(x.label)[0] + 0.5 * (1 - eval(x.label)[2]),
                eval(x.label)[1] + 0.5 * (1 - eval(x.label)[2]),
            ),
        )
        expected_channel_stim_dict = dict(
            zip(crd_channels, stim_qubit_instructions, strict=True)
        )
        channel_stim_dict = self.converter.eka_channel_to_stim_qubit_instruction_mapper(
            self.interpreted_eka_rsc.final_circuit, eka_stim_dict
        )
        self.assertEqual(expected_channel_stim_dict, channel_stim_dict)

        # Test correct handling of indices for one dimensional repetition code

        eka_stim_dict = self.converter.eka_coords_to_stim_qubit_instruction_mapper(
            self.interpreted_eka_rep.final_circuit
        )
        stim_qubit_instructions = [
            stim.Circuit(qubit)[0] for qubit in self.stim_qubits_rep
        ]
        expected_block_stim_dict = dict(
            zip(self.block_qubits_ordered_rep, stim_qubit_instructions, strict=True)
        )

        self.assertEqual(expected_block_stim_dict, eka_stim_dict)

    def test_generate_stim_instruction(self):
        """Test the function to generate stim instructions from a name, qubit index
        and gate_args
        """
        expected_cx_gate_instruction = stim.CircuitInstruction("CX", [0, 1], [])
        generated_cx_gate_instruction = (
            self.converter.generate_stim_circuit_instruction(
                name="CX", targets=[0, 1], gate_args=[]
            )
        )

        expected_meas_instruction = stim.CircuitInstruction("M", [0, 1, 2, 3, 4, 5], [])
        generated_meas_instruction = self.converter.generate_stim_circuit_instruction(
            name="M", targets=[0, 1, 2, 3, 4, 5], gate_args=[]
        )

        expected_qubit_instruction = stim.CircuitInstruction(
            "QUBIT_COORDS", [4], [0.5, 2.5]
        )
        generated_qubit_instruction = self.converter.generate_stim_circuit_instruction(
            name="QUBIT_COORDS", targets=[4], gate_args=[0.5, 2.5]
        )
        self.assertEqual(expected_cx_gate_instruction, generated_cx_gate_instruction)
        self.assertEqual(expected_meas_instruction, generated_meas_instruction)
        self.assertEqual(expected_qubit_instruction, generated_qubit_instruction)

    def test_measureblock_circuit(self):
        """
        Test circuit conversion generates the correct syndrome measurement
        circuit for all surface code stabilizers
        """
        expected_stim_circuit = stim.Circuit(
            """
            QUBIT_COORDS(0, 1) 0
            QUBIT_COORDS(0.5, 0.5) 1
            QUBIT_COORDS(0.5, 1.5) 2
            QUBIT_COORDS(0.5, 2.5) 3
            QUBIT_COORDS(1, 1) 4
            QUBIT_COORDS(1, 2) 5
            QUBIT_COORDS(1, 3) 6
            QUBIT_COORDS(1.5, 0.5) 7
            QUBIT_COORDS(1.5, 1.5) 8
            QUBIT_COORDS(1.5, 2.5) 9
            QUBIT_COORDS(2, 0) 10
            QUBIT_COORDS(2, 1) 11
            QUBIT_COORDS(2, 2) 12
            QUBIT_COORDS(2.5, 0.5) 13
            QUBIT_COORDS(2.5, 1.5) 14
            QUBIT_COORDS(2.5, 2.5) 15
            QUBIT_COORDS(3, 2) 16
            R 4 12 5 11 0 16 10 6
            H 4 12 5 11 0 16 10 6
            CZ 4 7 12 14 6 9
            CX 5 8 11 13 0 1
            CZ 4 1 12 8 6 3
            CX 5 9 11 14 0 2
            CZ 4 8 12 15 10 13
            CX 5 2 11 7 16 14
            CZ 4 2 12 9 10 7
            CX 5 3 11 8 16 15
            H 4 12 5 11 0 16 10 6
            M 4 12 5 11 0 16 10 6
            R 4 12 5 11 0 16 10 6
            H 4 12 5 11 0 16 10 6
            CZ 4 7 12 14 6 9
            CX 5 8 11 13 0 1
            CZ 4 1 12 8 6 3
            CX 5 9 11 14 0 2
            CZ 4 8 12 15 10 13
            CX 5 2 11 7 16 14
            CZ 4 2 12 9 10 7
            CX 5 3 11 8 16 15
            H 4 12 5 11 0 16 10 6
            M 4 12 5 11 0 16 10 6
            R 4 12 5 11 0 16 10 6
            H 4 12 5 11 0 16 10 6
            CZ 4 7 12 14 6 9
            CX 5 8 11 13 0 1
            CZ 4 1 12 8 6 3
            CX 5 9 11 14 0 2
            CZ 4 8 12 15 10 13
            CX 5 2 11 7 16 14
            CZ 4 2 12 9 10 7
            CX 5 3 11 8 16 15
            H 4 12 5 11 0 16 10 6
            M 4 12 5 11 0 16 10 6
            DETECTOR(1, 1, 0) rec[-24] rec[-16]
            DETECTOR(2, 2, 0) rec[-23] rec[-15]
            DETECTOR(1, 2, 0) rec[-22] rec[-14]
            DETECTOR(2, 1, 0) rec[-21] rec[-13]
            DETECTOR(0, 1, 0) rec[-20] rec[-12]
            DETECTOR(3, 2, 0) rec[-19] rec[-11]
            DETECTOR(2, 0, 0) rec[-18] rec[-10]
            DETECTOR(1, 3, 0) rec[-17] rec[-9]
            DETECTOR(1, 1, 0) rec[-16] rec[-8]
            DETECTOR(2, 2, 0) rec[-15] rec[-7]
            DETECTOR(1, 2, 0) rec[-14] rec[-6]
            DETECTOR(2, 1, 0) rec[-13] rec[-5]
            DETECTOR(0, 1, 0) rec[-12] rec[-4]
            DETECTOR(3, 2, 0) rec[-11] rec[-3]
            DETECTOR(2, 0, 0) rec[-10] rec[-2]
            DETECTOR(1, 3, 0) rec[-9] rec[-1]
        """
        )
        converted_stim_circuit = self.converter.convert(self.interpreted_eka_rsc)
        self.assertEqual(converted_stim_circuit, expected_stim_circuit)

    def test_stim_circuit_with_noise(self):
        """Add noise to stim circuit and test whether the noise is applied correctly"""
        p = 0.01  # noise strength
        stim_circuit_noiseless = stim.Circuit(
            """
            QUBIT_COORDS(0, 1) 0
            QUBIT_COORDS(0.5, 0.5) 1
            QUBIT_COORDS(0.5, 1.5) 2
            QUBIT_COORDS(0.5, 2.5) 3
            QUBIT_COORDS(1, 1) 4
            QUBIT_COORDS(1, 2) 5
            QUBIT_COORDS(1, 3) 6
            QUBIT_COORDS(1.5, 0.5) 7
            QUBIT_COORDS(1.5, 1.5) 8
            QUBIT_COORDS(1.5, 2.5) 9
            QUBIT_COORDS(2, 0) 10
            QUBIT_COORDS(2, 1) 11
            QUBIT_COORDS(2, 2) 12
            QUBIT_COORDS(2.5, 0.5) 13
            QUBIT_COORDS(2.5, 1.5) 14
            QUBIT_COORDS(2.5, 2.5) 15
            QUBIT_COORDS(3, 2) 16
            R 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
            H 4 12 5 11 0 16 10 6
            CZ 4 7 12 14
            CX 5 8 11 13 0 1
            CZ 6 9 4 8 12 15
            CX 5 2 11 7 16 14
            CZ 10 13 4 1 12 8
            CX 5 9 11 14 0 2
            CZ 6 3 4 2 12 9
            CX 5 3 11 8 16 15
            CZ 10 7
            H 4 12 5 11 0 16 10 6
            M 4 12 5 11 0 16 10 6
            R 4 12 5 11 0 16 10 6
            H 4 12 5 11 0 16 10 6
            CZ 4 7 12 14
            CX 5 8 11 13 0 1
            CZ 6 9 4 8 12 15
            CX 5 2 11 7 16 14
            CZ 10 13 4 1 12 8
            CX 5 9 11 14 0 2
            CZ 6 3 4 2 12 9
            CX 5 3 11 8 16 15
            CZ 10 7
            H 4 12 5 11 0 16 10 6
            M 4 12 5 11 0 16 10 6
            """
        )

        generated_stim_circuit_with_measure_flip_noise = noise_annotated_stim_circuit(
            stim_circuit_noiseless,
            before_measure_flip_probability=p,
        )
        expected_stim_circuit_with_measure_flip_noise = stim.Circuit(
            f"""
            QUBIT_COORDS(0, 1) 0
            QUBIT_COORDS(0.5, 0.5) 1
            QUBIT_COORDS(0.5, 1.5) 2
            QUBIT_COORDS(0.5, 2.5) 3
            QUBIT_COORDS(1, 1) 4
            QUBIT_COORDS(1, 2) 5
            QUBIT_COORDS(1, 3) 6
            QUBIT_COORDS(1.5, 0.5) 7
            QUBIT_COORDS(1.5, 1.5) 8
            QUBIT_COORDS(1.5, 2.5) 9
            QUBIT_COORDS(2, 0) 10
            QUBIT_COORDS(2, 1) 11
            QUBIT_COORDS(2, 2) 12
            QUBIT_COORDS(2.5, 0.5) 13
            QUBIT_COORDS(2.5, 1.5) 14
            QUBIT_COORDS(2.5, 2.5) 15
            QUBIT_COORDS(3, 2) 16
            R 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
            H 4 12 5 11 0 16 10 6
            CZ 4 7 12 14
            CX 5 8 11 13 0 1
            CZ 6 9 4 8 12 15
            CX 5 2 11 7 16 14
            CZ 10 13 4 1 12 8
            CX 5 9 11 14 0 2
            CZ 6 3 4 2 12 9
            CX 5 3 11 8 16 15
            CZ 10 7
            H 4 12 5 11 0 16 10 6
            X_ERROR({p}) 4 12 5 11 0 16 10 6
            M 4 12 5 11 0 16 10 6
            R 4 12 5 11 0 16 10 6
            H 4 12 5 11 0 16 10 6
            CZ 4 7 12 14
            CX 5 8 11 13 0 1
            CZ 6 9 4 8 12 15
            CX 5 2 11 7 16 14
            CZ 10 13 4 1 12 8
            CX 5 9 11 14 0 2
            CZ 6 3 4 2 12 9
            CX 5 3 11 8 16 15
            CZ 10 7
            H 4 12 5 11 0 16 10 6
            X_ERROR({p}) 4 12 5 11 0 16 10 6
            M 4 12 5 11 0 16 10 6
            """
        )

        generated_stim_circuit_with_after_clifford_noise = noise_annotated_stim_circuit(
            stim_circuit_noiseless,
            after_clifford_depolarization=p,
        )
        expected_stim_circuit_with_after_clifford_noise = stim.Circuit(
            f"""
            QUBIT_COORDS(0, 1) 0
            QUBIT_COORDS(0.5, 0.5) 1
            QUBIT_COORDS(0.5, 1.5) 2
            QUBIT_COORDS(0.5, 2.5) 3
            QUBIT_COORDS(1, 1) 4
            QUBIT_COORDS(1, 2) 5
            QUBIT_COORDS(1, 3) 6
            QUBIT_COORDS(1.5, 0.5) 7
            QUBIT_COORDS(1.5, 1.5) 8
            QUBIT_COORDS(1.5, 2.5) 9
            QUBIT_COORDS(2, 0) 10
            QUBIT_COORDS(2, 1) 11
            QUBIT_COORDS(2, 2) 12
            QUBIT_COORDS(2.5, 0.5) 13
            QUBIT_COORDS(2.5, 1.5) 14
            QUBIT_COORDS(2.5, 2.5) 15
            QUBIT_COORDS(3, 2) 16
            R 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
            H 4 12 5 11 0 16 10 6
            DEPOLARIZE1({p}) 4 12 5 11 0 16 10 6
            CZ 4 7 12 14
            DEPOLARIZE2({p}) 4 7 12 14
            CX 5 8 11 13 0 1
            DEPOLARIZE2({p}) 5 8 11 13 0 1
            CZ 6 9 4 8 12 15
            DEPOLARIZE2({p}) 6 9 4 8 12 15
            CX 5 2 11 7 16 14
            DEPOLARIZE2({p}) 5 2 11 7 16 14
            CZ 10 13 4 1 12 8
            DEPOLARIZE2({p}) 10 13 4 1 12 8
            CX 5 9 11 14 0 2
            DEPOLARIZE2({p}) 5 9 11 14 0 2
            CZ 6 3 4 2 12 9
            DEPOLARIZE2({p}) 6 3 4 2 12 9
            CX 5 3 11 8 16 15
            DEPOLARIZE2({p}) 5 3 11 8 16 15
            CZ 10 7
            DEPOLARIZE2({p}) 10 7
            H 4 12 5 11 0 16 10 6
            DEPOLARIZE1({p}) 4 12 5 11 0 16 10 6
            M 4 12 5 11 0 16 10 6
            R 4 12 5 11 0 16 10 6
            H 4 12 5 11 0 16 10 6
            DEPOLARIZE1({p}) 4 12 5 11 0 16 10 6
            CZ 4 7 12 14
            DEPOLARIZE2({p}) 4 7 12 14
            CX 5 8 11 13 0 1
            DEPOLARIZE2({p}) 5 8 11 13 0 1
            CZ 6 9 4 8 12 15
            DEPOLARIZE2({p}) 6 9 4 8 12 15
            CX 5 2 11 7 16 14
            DEPOLARIZE2({p}) 5 2 11 7 16 14
            CZ 10 13 4 1 12 8
            DEPOLARIZE2({p}) 10 13 4 1 12 8
            CX 5 9 11 14 0 2
            DEPOLARIZE2({p}) 5 9 11 14 0 2
            CZ 6 3 4 2 12 9
            DEPOLARIZE2({p}) 6 3 4 2 12 9
            CX 5 3 11 8 16 15
            DEPOLARIZE2({p}) 5 3 11 8 16 15
            CZ 10 7
            DEPOLARIZE2({p}) 10 7
            H 4 12 5 11 0 16 10 6
            DEPOLARIZE1({p}) 4 12 5 11 0 16 10 6
            M 4 12 5 11 0 16 10 6
            """
        )

        generated_stim_cicuit_with_reset_flip_noise = noise_annotated_stim_circuit(
            stim_circuit_noiseless,
            after_reset_flip_probability=p,
        )

        expected_stim_circuit_with_reset_flip_noise = stim.Circuit(
            f"""
            QUBIT_COORDS(0, 1) 0
            QUBIT_COORDS(0.5, 0.5) 1
            QUBIT_COORDS(0.5, 1.5) 2
            QUBIT_COORDS(0.5, 2.5) 3
            QUBIT_COORDS(1, 1) 4
            QUBIT_COORDS(1, 2) 5
            QUBIT_COORDS(1, 3) 6
            QUBIT_COORDS(1.5, 0.5) 7
            QUBIT_COORDS(1.5, 1.5) 8
            QUBIT_COORDS(1.5, 2.5) 9
            QUBIT_COORDS(2, 0) 10
            QUBIT_COORDS(2, 1) 11
            QUBIT_COORDS(2, 2) 12
            QUBIT_COORDS(2.5, 0.5) 13
            QUBIT_COORDS(2.5, 1.5) 14
            QUBIT_COORDS(2.5, 2.5) 15
            QUBIT_COORDS(3, 2) 16
            R 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
            X_ERROR({p}) 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
            H 4 12 5 11 0 16 10 6
            CZ 4 7 12 14
            CX 5 8 11 13 0 1
            CZ 6 9 4 8 12 15
            CX 5 2 11 7 16 14
            CZ 10 13 4 1 12 8
            CX 5 9 11 14 0 2
            CZ 6 3 4 2 12 9
            CX 5 3 11 8 16 15
            CZ 10 7
            H 4 12 5 11 0 16 10 6
            M 4 12 5 11 0 16 10 6
            R 4 12 5 11 0 16 10 6
            X_ERROR({p}) 4 12 5 11 0 16 10 6
            H 4 12 5 11 0 16 10 6
            CZ 4 7 12 14
            CX 5 8 11 13 0 1
            CZ 6 9 4 8 12 15
            CX 5 2 11 7 16 14
            CZ 10 13 4 1 12 8
            CX 5 9 11 14 0 2
            CZ 6 3 4 2 12 9
            CX 5 3 11 8 16 15
            CZ 10 7
            H 4 12 5 11 0 16 10 6
            M 4 12 5 11 0 16 10 6
            """
        )

        # test for correct measure flip noise annotataion
        self.assertEqual(
            generated_stim_circuit_with_measure_flip_noise,
            expected_stim_circuit_with_measure_flip_noise,
        )

        # test for correct after clifford noise annotataion
        self.assertEqual(
            generated_stim_circuit_with_after_clifford_noise,
            expected_stim_circuit_with_after_clifford_noise,
        )

        # test for correct reset flip noise annotataion
        self.assertEqual(
            generated_stim_cicuit_with_reset_flip_noise,
            expected_stim_circuit_with_reset_flip_noise,
        )

    def assert_without_detector_order(
        self, expected: stim.Circuit, observed: stim.Circuit
    ) -> str:
        """Assert that two stim.Circuit objects are equal, ignoring DETECTOR lines order."""
        expected_str = str(expected)
        observed_str = str(observed)

        self.assertEqual(len(expected_str.splitlines()), len(observed_str.splitlines()))

        def strip_detector(circuit_str: str) -> tuple[str, set[str]]:
            """Remove DETECTOR lines from the observed string and return the rest."""
            res = []
            detectors_set = set()
            for line in circuit_str.splitlines():
                if line.startswith("DETECTOR"):
                    detectors_set.add(line)
                else:
                    res.append(line)
            return "\n".join(res), detectors_set

        self.assertEqual(
            strip_detector(expected_str),
            strip_detector(observed_str),
        )

    def test_conversion_with_depolarize1_time_independent_error(self):
        """test the conversion of a circuit with error models"""

        # Define error models for depolarization after 1 qubits Clifford gates
        class After1CliffordDepolarization(CircuitErrorModel):
            """A simple error model that applies depolarizing noise after single qubit
            Clifford gates."""

            probability: float

            is_time_dependent: bool = False
            error_type: ErrorType = ErrorType.DEPOLARIZING1
            application_mode: ApplicationMode = ApplicationMode.AFTER_GATE

            def model_post_init(self, __context):
                super().model_post_init(__context)
                single_qubit_gates = ["x", "h", "z", "y", "hadamard", "identity", "i"]
                object.__setattr__(
                    self,
                    "gate_error_probabilities",
                    self.validate_gate_error_probabilities(
                        {
                            gate: lambda t: [self.probability]
                            for gate in single_qubit_gates
                        }
                    ),
                )

        expected_output_str = stim.Circuit(
            """
                QUBIT_COORDS(0, 1) 0
                QUBIT_COORDS(0.5, 0.5) 1
                QUBIT_COORDS(0.5, 1.5) 2
                QUBIT_COORDS(0.5, 2.5) 3
                QUBIT_COORDS(1, 1) 4
                QUBIT_COORDS(1, 2) 5
                QUBIT_COORDS(1, 3) 6
                QUBIT_COORDS(1.5, 0.5) 7
                QUBIT_COORDS(1.5, 1.5) 8
                QUBIT_COORDS(1.5, 2.5) 9
                QUBIT_COORDS(2, 0) 10
                QUBIT_COORDS(2, 1) 11
                QUBIT_COORDS(2, 2) 12
                QUBIT_COORDS(2.5, 0.5) 13
                QUBIT_COORDS(2.5, 1.5) 14
                QUBIT_COORDS(2.5, 2.5) 15
                QUBIT_COORDS(3, 2) 16
                R 4 12 5 11 0 16 10 6
                H 4 12 5 11 0 16 10 6
                DEPOLARIZE1(0.01) 4 12 5 11 0 16 10 6
                CZ 4 7 12 14 6 9
                CX 5 8 11 13 0 1
                CZ 4 1 12 8 6 3
                CX 5 9 11 14 0 2
                CZ 4 8 12 15 10 13
                CX 5 2 11 7 16 14
                CZ 4 2 12 9 10 7
                CX 5 3 11 8 16 15
                H 4 12 5 11 0 16 10 6
                DEPOLARIZE1(0.01) 4 12 5 11 0 16 10 6
                M 4 12 5 11 0 16 10 6
                R 4 12 5 11 0 16 10 6
                H 4 12 5 11 0 16 10 6
                DEPOLARIZE1(0.01) 4 12 5 11 0 16 10 6
                CZ 4 7 12 14 6 9
                CX 5 8 11 13 0 1
                CZ 4 1 12 8 6 3
                CX 5 9 11 14 0 2
                CZ 4 8 12 15 10 13
                CX 5 2 11 7 16 14
                CZ 4 2 12 9 10 7
                CX 5 3 11 8 16 15
                H 4 12 5 11 0 16 10 6
                DEPOLARIZE1(0.01) 4 12 5 11 0 16 10 6
                M 4 12 5 11 0 16 10 6
                R 4 12 5 11 0 16 10 6
                H 4 12 5 11 0 16 10 6
                DEPOLARIZE1(0.01) 4 12 5 11 0 16 10 6
                CZ 4 7 12 14 6 9
                CX 5 8 11 13 0 1
                CZ 4 1 12 8 6 3
                CX 5 9 11 14 0 2
                CZ 4 8 12 15 10 13
                CX 5 2 11 7 16 14
                CZ 4 2 12 9 10 7
                CX 5 3 11 8 16 15
                H 4 12 5 11 0 16 10 6
                DEPOLARIZE1(0.01) 4 12 5 11 0 16 10 6
                M 4 12 5 11 0 16 10 6
                DETECTOR(1, 1, 0) rec[-24] rec[-16]
                DETECTOR(2, 2, 0) rec[-23] rec[-15]
                DETECTOR(1, 2, 0) rec[-22] rec[-14]
                DETECTOR(2, 1, 0) rec[-21] rec[-13]
                DETECTOR(0, 1, 0) rec[-20] rec[-12]
                DETECTOR(3, 2, 0) rec[-19] rec[-11]
                DETECTOR(2, 0, 0) rec[-18] rec[-10]
                DETECTOR(1, 3, 0) rec[-17] rec[-9]
                DETECTOR(1, 1, 0) rec[-16] rec[-8]
                DETECTOR(2, 2, 0) rec[-15] rec[-7]
                DETECTOR(1, 2, 0) rec[-14] rec[-6]
                DETECTOR(2, 1, 0) rec[-13] rec[-5]
                DETECTOR(0, 1, 0) rec[-12] rec[-4]
                DETECTOR(3, 2, 0) rec[-11] rec[-3]
                DETECTOR(2, 0, 0) rec[-10] rec[-2]
                DETECTOR(1, 3, 0) rec[-9] rec[-1]
            """
        )

        error_model = [
            After1CliffordDepolarization(
                circuit=self.interpreted_eka_rsc.final_circuit, probability=0.01
            )
        ]
        converted_stim_circuit = self.converter.convert(
            self.interpreted_eka_rsc,
            error_models=error_model,
        )

        self.assert_without_detector_order(expected_output_str, converted_stim_circuit)

    def test_conversion_with_depolarize2_time_dependent_error(self):
        """test the conversion of a circuit with error models"""

        class After2CliffordDepolarization(CircuitErrorModel):
            """Define a time-dependent error model for depolarization after
            2 qubits Clifford gates"""

            probability_scale_over_time: float

            is_time_dependent: bool = True
            error_type: ErrorType = ErrorType.DEPOLARIZING2
            application_mode: ApplicationMode = ApplicationMode.AFTER_GATE

            def model_post_init(self, __context):
                super().model_post_init(__context)
                two_qubit_gates = ["cz", "cx", "cy", "swap"]
                object.__setattr__(
                    self,
                    "gate_error_probabilities",
                    self.validate_gate_error_probabilities(
                        {
                            gate: lambda t: [self.probability_scale_over_time * t]
                            for gate in two_qubit_gates
                        }
                    ),
                )

        expected_output_str = stim.Circuit(
            """
                QUBIT_COORDS(0, 1) 0
                QUBIT_COORDS(0.5, 0.5) 1
                QUBIT_COORDS(0.5, 1.5) 2
                QUBIT_COORDS(0.5, 2.5) 3
                QUBIT_COORDS(1, 1) 4
                QUBIT_COORDS(1, 2) 5
                QUBIT_COORDS(1, 3) 6
                QUBIT_COORDS(1.5, 0.5) 7
                QUBIT_COORDS(1.5, 1.5) 8
                QUBIT_COORDS(1.5, 2.5) 9
                QUBIT_COORDS(2, 0) 10
                QUBIT_COORDS(2, 1) 11
                QUBIT_COORDS(2, 2) 12
                QUBIT_COORDS(2.5, 0.5) 13
                QUBIT_COORDS(2.5, 1.5) 14
                QUBIT_COORDS(2.5, 2.5) 15
                QUBIT_COORDS(3, 2) 16
                R 4 12 5 11 0 16 10 6
                H 4 12 5 11 0 16 10 6
                CZ 4 7 12 14 6 9
                CX 5 8 11 13 0 1
                DEPOLARIZE2(0.03) 4 7 12 14 5 8 11 13 0 1 6 9
                CZ 4 1 12 8 6 3
                CX 5 9 11 14 0 2
                DEPOLARIZE2(0.04) 4 1 12 8 5 9 11 14 0 2 6 3
                CZ 4 8 12 15 10 13
                CX 5 2 11 7 16 14
                DEPOLARIZE2(0.05) 4 8 12 15 5 2 11 7 16 14 10 13
                CZ 4 2 12 9 10 7
                CX 5 3 11 8 16 15
                DEPOLARIZE2(0.06) 4 2 12 9 5 3 11 8 16 15 10 7
                H 4 12 5 11 0 16 10 6
                M 4 12 5 11 0 16 10 6
                R 4 12 5 11 0 16 10 6
                H 4 12 5 11 0 16 10 6
                CZ 4 7 12 14 6 9
                CX 5 8 11 13 0 1
                DEPOLARIZE2(0.11) 4 7 12 14 5 8 11 13 0 1 6 9
                CZ 4 1 12 8 6 3
                CX 5 9 11 14 0 2
                DEPOLARIZE2(0.12) 4 1 12 8 5 9 11 14 0 2 6 3
                CZ 4 8 12 15 10 13
                CX 5 2 11 7 16 14
                DEPOLARIZE2(0.13) 4 8 12 15 5 2 11 7 16 14 10 13
                CZ 4 2 12 9 10 7
                CX 5 3 11 8 16 15
                DEPOLARIZE2(0.14) 4 2 12 9 5 3 11 8 16 15 10 7
                H 4 12 5 11 0 16 10 6
                M 4 12 5 11 0 16 10 6
                R 4 12 5 11 0 16 10 6
                H 4 12 5 11 0 16 10 6
                CZ 4 7 12 14 6 9
                CX 5 8 11 13 0 1
                DEPOLARIZE2(0.19) 4 7 12 14 5 8 11 13 0 1 6 9
                CZ 4 1 12 8 6 3
                CX 5 9 11 14 0 2
                DEPOLARIZE2(0.2) 4 1 12 8 5 9 11 14 0 2 6 3
                CZ 4 8 12 15 10 13
                CX 5 2 11 7 16 14
                DEPOLARIZE2(0.21) 4 8 12 15 5 2 11 7 16 14 10 13
                CZ 4 2 12 9 10 7
                CX 5 3 11 8 16 15
                DEPOLARIZE2(0.22) 4 2 12 9 5 3 11 8 16 15 10 7
                H 4 12 5 11 0 16 10 6
                M 4 12 5 11 0 16 10 6
                DETECTOR(1, 1, 0) rec[-24] rec[-16]
                DETECTOR(2, 2, 0) rec[-23] rec[-15]
                DETECTOR(1, 2, 0) rec[-22] rec[-14]
                DETECTOR(2, 1, 0) rec[-21] rec[-13]
                DETECTOR(0, 1, 0) rec[-20] rec[-12]
                DETECTOR(3, 2, 0) rec[-19] rec[-11]
                DETECTOR(2, 0, 0) rec[-18] rec[-10]
                DETECTOR(1, 3, 0) rec[-17] rec[-9]
                DETECTOR(1, 1, 0) rec[-16] rec[-8]
                DETECTOR(2, 2, 0) rec[-15] rec[-7]
                DETECTOR(1, 2, 0) rec[-14] rec[-6]
                DETECTOR(2, 1, 0) rec[-13] rec[-5]
                DETECTOR(0, 1, 0) rec[-12] rec[-4]
                DETECTOR(3, 2, 0) rec[-11] rec[-3]
                DETECTOR(2, 0, 0) rec[-10] rec[-2]
                DETECTOR(1, 3, 0) rec[-9] rec[-1]
            """
        )

        circ = self.interpreted_eka_rsc.final_circuit
        g_set = {g.name for layer in Circuit.unroll(circ) for g in layer}
        gate_duration = {gate: 1 for gate in g_set}

        error_model = [
            After2CliffordDepolarization(
                circuit=circ,
                probability_scale_over_time=0.01,
                gate_durations=gate_duration,
            )
        ]
        converted_stim_circuit = self.converter.convert(
            self.interpreted_eka_rsc,
            error_models=error_model,
        )

        self.assert_without_detector_order(expected_output_str, converted_stim_circuit)

    def test_conversion_with_multiple_error_models(self):
        """Test the correction creation of a Stim Circuit with multiple error models"""

        # Test creation for Rotated Surface Code
        circ_rsc = self.interpreted_eka_rsc.final_circuit
        g_set = {g.name for layer in Circuit.unroll(circ_rsc) for g in layer}
        gate_duration = {gate: 1 for gate in g_set}

        m1 = HomogeneousTimeDependentCEM(
            circuit=circ_rsc,
            error_type=ErrorType.PAULI_X,
            gate_durations=gate_duration,
            application_mode=ApplicationMode.END_OF_TICK,
            error_probability=lambda t, _=None: [0.01 * t],
        )

        m2 = HomogeneousTimeIndependentCEM(
            circuit=circ_rsc,
            error_type=ErrorType.PAULI_Y,
            gate_durations=gate_duration,
            application_mode=ApplicationMode.AFTER_GATE,
            error_probability=0.02,
            target_gates=["cx", "cz", "cy", "swap"],
        )

        m3 = AsymmetricDepolarizeCEM(
            circuit=circ_rsc,
            gate_durations=gate_duration,
            t1=1,
            t2=1.5,
        )

        expected_output_str = stim.Circuit(
            """
            QUBIT_COORDS(0, 1) 0
            QUBIT_COORDS(0.5, 0.5) 1
            QUBIT_COORDS(0.5, 1.5) 2
            QUBIT_COORDS(0.5, 2.5) 3
            QUBIT_COORDS(1, 1) 4
            QUBIT_COORDS(1, 2) 5
            QUBIT_COORDS(1, 3) 6
            QUBIT_COORDS(1.5, 0.5) 7
            QUBIT_COORDS(1.5, 1.5) 8
            QUBIT_COORDS(1.5, 2.5) 9
            QUBIT_COORDS(2, 0) 10
            QUBIT_COORDS(2, 1) 11
            QUBIT_COORDS(2, 2) 12
            QUBIT_COORDS(2.5, 0.5) 13
            QUBIT_COORDS(2.5, 1.5) 14
            QUBIT_COORDS(2.5, 2.5) 15
            QUBIT_COORDS(3, 2) 16
            R 4 12 5 11 0 16 10 6
            X_ERROR(0.01) 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
            PAULI_CHANNEL_1(0.15803, 0.15803, 0.0852613) 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
            H 4 12 5 11 0 16 10 6
            X_ERROR(0.02) 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
            PAULI_CHANNEL_1(0.15803, 0.15803, 0.0852613) 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
            CZ 4 7 12 14 6 9
            CX 5 8 11 13 0 1
            Y_ERROR(0.02) 4 7 12 14 5 8 11 13 0 1 6 9
            X_ERROR(0.03) 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
            PAULI_CHANNEL_1(0.15803, 0.15803, 0.0852613) 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
            CZ 4 1 12 8 6 3
            CX 5 9 11 14 0 2
            Y_ERROR(0.02) 4 1 12 8 5 9 11 14 0 2 6 3
            X_ERROR(0.04) 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
            PAULI_CHANNEL_1(0.15803, 0.15803, 0.0852613) 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
            CZ 4 8 12 15 10 13
            CX 5 2 11 7 16 14
            Y_ERROR(0.02) 4 8 12 15 5 2 11 7 16 14 10 13
            X_ERROR(0.05) 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
            PAULI_CHANNEL_1(0.15803, 0.15803, 0.0852613) 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
            CZ 4 2 12 9 10 7
            CX 5 3 11 8 16 15
            Y_ERROR(0.02) 4 2 12 9 5 3 11 8 16 15 10 7
            X_ERROR(0.06) 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
            PAULI_CHANNEL_1(0.15803, 0.15803, 0.0852613) 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
            H 4 12 5 11 0 16 10 6
            X_ERROR(0.07) 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
            PAULI_CHANNEL_1(0.15803, 0.15803, 0.0852613) 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
            M 4 12 5 11 0 16 10 6
            X_ERROR(0.08) 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
            PAULI_CHANNEL_1(0.15803, 0.15803, 0.0852613) 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
            R 4 12 5 11 0 16 10 6
            X_ERROR(0.09) 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
            PAULI_CHANNEL_1(0.15803, 0.15803, 0.0852613) 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
            H 4 12 5 11 0 16 10 6
            X_ERROR(0.1) 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
            PAULI_CHANNEL_1(0.15803, 0.15803, 0.0852613) 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
            CZ 4 7 12 14 6 9
            CX 5 8 11 13 0 1
            Y_ERROR(0.02) 4 7 12 14 5 8 11 13 0 1 6 9
            X_ERROR(0.11) 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
            PAULI_CHANNEL_1(0.15803, 0.15803, 0.0852613) 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
            CZ 4 1 12 8 6 3
            CX 5 9 11 14 0 2
            Y_ERROR(0.02) 4 1 12 8 5 9 11 14 0 2 6 3
            X_ERROR(0.12) 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
            PAULI_CHANNEL_1(0.15803, 0.15803, 0.0852613) 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
            CZ 4 8 12 15 10 13
            CX 5 2 11 7 16 14
            Y_ERROR(0.02) 4 8 12 15 5 2 11 7 16 14 10 13
            X_ERROR(0.13) 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
            PAULI_CHANNEL_1(0.15803, 0.15803, 0.0852613) 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
            CZ 4 2 12 9 10 7
            CX 5 3 11 8 16 15
            Y_ERROR(0.02) 4 2 12 9 5 3 11 8 16 15 10 7
            X_ERROR(0.14) 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
            PAULI_CHANNEL_1(0.15803, 0.15803, 0.0852613) 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
            H 4 12 5 11 0 16 10 6
            X_ERROR(0.15) 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
            PAULI_CHANNEL_1(0.15803, 0.15803, 0.0852613) 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
            M 4 12 5 11 0 16 10 6
            X_ERROR(0.16) 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
            PAULI_CHANNEL_1(0.15803, 0.15803, 0.0852613) 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
            R 4 12 5 11 0 16 10 6
            X_ERROR(0.17) 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
            PAULI_CHANNEL_1(0.15803, 0.15803, 0.0852613) 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
            H 4 12 5 11 0 16 10 6
            X_ERROR(0.18) 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
            PAULI_CHANNEL_1(0.15803, 0.15803, 0.0852613) 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
            CZ 4 7 12 14 6 9
            CX 5 8 11 13 0 1
            Y_ERROR(0.02) 4 7 12 14 5 8 11 13 0 1 6 9
            X_ERROR(0.19) 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
            PAULI_CHANNEL_1(0.15803, 0.15803, 0.0852613) 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
            CZ 4 1 12 8 6 3
            CX 5 9 11 14 0 2
            Y_ERROR(0.02) 4 1 12 8 5 9 11 14 0 2 6 3
            X_ERROR(0.2) 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
            PAULI_CHANNEL_1(0.15803, 0.15803, 0.0852613) 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
            CZ 4 8 12 15 10 13
            CX 5 2 11 7 16 14
            Y_ERROR(0.02) 4 8 12 15 5 2 11 7 16 14 10 13
            X_ERROR(0.21) 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
            PAULI_CHANNEL_1(0.15803, 0.15803, 0.0852613) 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
            CZ 4 2 12 9 10 7
            CX 5 3 11 8 16 15
            Y_ERROR(0.02) 4 2 12 9 5 3 11 8 16 15 10 7
            X_ERROR(0.22) 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
            PAULI_CHANNEL_1(0.15803, 0.15803, 0.0852613) 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
            H 4 12 5 11 0 16 10 6
            X_ERROR(0.23) 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
            PAULI_CHANNEL_1(0.15803, 0.15803, 0.0852613) 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
            M 4 12 5 11 0 16 10 6
            X_ERROR(0.24) 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
            PAULI_CHANNEL_1(0.15803, 0.15803, 0.0852613) 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
            DETECTOR(1, 1, 0) rec[-24] rec[-16]
            DETECTOR(2, 2, 0) rec[-23] rec[-15]
            DETECTOR(1, 2, 0) rec[-22] rec[-14]
            DETECTOR(2, 1, 0) rec[-21] rec[-13]
            DETECTOR(0, 1, 0) rec[-20] rec[-12]
            DETECTOR(3, 2, 0) rec[-19] rec[-11]
            DETECTOR(2, 0, 0) rec[-18] rec[-10]
            DETECTOR(1, 3, 0) rec[-17] rec[-9]
            DETECTOR(1, 1, 0) rec[-16] rec[-8]
            DETECTOR(2, 2, 0) rec[-15] rec[-7]
            DETECTOR(1, 2, 0) rec[-14] rec[-6]
            DETECTOR(2, 1, 0) rec[-13] rec[-5]
            DETECTOR(0, 1, 0) rec[-12] rec[-4]
            DETECTOR(3, 2, 0) rec[-11] rec[-3]
            DETECTOR(2, 0, 0) rec[-10] rec[-2]
            DETECTOR(1, 3, 0) rec[-9] rec[-1]
            """
        )
        error_model = [m1, m2, m3]
        converted_stim_circuit = self.converter.convert(
            self.interpreted_eka_rsc,
            error_models=error_model,
        )
        self.assert_without_detector_order(expected_output_str, converted_stim_circuit)


if __name__ == "__main__":
    unittest.main()
