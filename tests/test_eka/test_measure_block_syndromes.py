"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

from copy import deepcopy
import unittest
import logging
import networkx as nx

from loom.eka import (
    Circuit,
    Channel,
    ChannelType,
    SyndromeCircuit,
    Eka,
    Lattice,
    Stabilizer,
    Block,
    PauliOperator,
)
from loom.eka.operations import MeasureBlockSyndromes
from loom.interpreter import InterpretationStep, Syndrome, Detector
from loom.interpreter.applicator import CodeApplicator
from loom.eka.tanner_graphs import ClassicalTannerGraph


# pylint: disable=invalid-name, too-many-instance-attributes, too-many-locals
class TestMeasureBlockSyndromes(unittest.TestCase):
    """
    Test for the MeasureBlockSyndromes applicator and logging.
    """

    def setUp(self):
        self.base_step = InterpretationStep()

        self.square_2d_lattice = Lattice.square_2d((10, 20))
        self.eka_no_blocks = Eka(self.square_2d_lattice)
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
        self.distance_rep = 3
        self.nodes_bitflip_rep = [
            ((i,), {"label": "data"}) for i in range(self.distance_rep)
        ] + [
            ((i,), {"label": "Z"})
            for i in range(self.distance_rep, 2 * self.distance_rep - 1)
        ]
        self.edges_bitflip_rep = [
            ((i + j,), (self.distance_rep + i,))
            for i in range(self.distance_rep - 1)
            for j in range(2)
        ]

        self.T_bitflip_rep = nx.Graph()
        self.T_bitflip_rep.add_nodes_from(self.nodes_bitflip_rep)
        self.T_bitflip_rep.add_edges_from(self.edges_bitflip_rep)

        self.rep_code = ClassicalTannerGraph(self.T_bitflip_rep)

        # Set to ERROR to avoid cluttering the test output
        logging.getLogger().setLevel(logging.ERROR)
        self.formatter = lambda lvl, msg: (
            f"{lvl}:loom.interpreter.applicator.measure_block_syndromes:{msg}"
        )

    def test_logging_measureblocksyndromes(self):
        """
        Test that the logging in measureblocksyndromes works as expected.

        Note: There is no way to test logging formatting, so we only check that the
        logging is called
        """
        lattice = Lattice.linear()
        repc = Block(
            stabilizers=(
                Stabilizer(
                    pauli="ZZ",
                    data_qubits=((0, 0, 0), (1, 0, 0)),
                    ancilla_qubits=((0, 0, 1),),
                ),
                Stabilizer(
                    pauli="ZZ",
                    data_qubits=((1, 0, 0), (2, 0, 0)),
                    ancilla_qubits=((1, 0, 1),),
                ),
            ),
            logical_x_operators=[
                PauliOperator(
                    pauli="XXX", data_qubits=((0, 0, 0), (1, 0, 0), (2, 0, 0))
                )
            ],
            logical_z_operators=[PauliOperator(pauli="Z", data_qubits=((0, 0, 0),))],
            unique_label="q1",
        )
        input_eka = Eka(
            lattice,
            blocks=[repc],
            operations=[MeasureBlockSyndromes(repc.unique_label)],
        )
        with self.assertLogs(
            "loom.interpreter.applicator.measure_block_syndromes",
            level="DEBUG",
        ) as cm:
            _ = CodeApplicator(input_eka).apply(
                InterpretationStep(block_history=((repc,),)),
                MeasureBlockSyndromes(repc.unique_label),
                same_timeslice=True,
                debug_mode=False,
            )

        # Build the expected log message as a single string (no newlines)
        err_msg = (
            "measure q1 syndromes 1 time(s)\n"
            "0: reset_0 - (0, 0, 1)\n"
            "reset_0 - (1, 0, 1)\n"
            "1: h - (0, 0, 1)\n"
            "h - (1, 0, 1)\n"
            "2: cz - (0, 0, 1) (0, 0, 0)\n"
            "cz - (1, 0, 1) (1, 0, 0)\n"
            "3: cz - (0, 0, 1) (1, 0, 0)\n"
            "cz - (1, 0, 1) (2, 0, 0)\n"
            "4: h - (0, 0, 1)\n"
            "h - (1, 0, 1)\n"
            "5: measurement - (0, 0, 1) c_(0, 0, 1)_0\n"
            "measurement - (1, 0, 1) c_(1, 0, 1)_0\n"
        )
        self.assertEqual(self.formatter("DEBUG", err_msg), cm.output[0])

    def test_applicator_measureblocksyndromes(self):
        """
        Test that the applicator creates the correct circuit and syndromes for a
        MeasureStabilizerSyndromes operation.
        The Circuit can be found in the InterpretationStep object.
        """
        lattice = Lattice.square_2d((10, 20))
        rsc_block = self.rot_surf_code_1

        # In order 110, 010, 120, 210, 020, 000, 100, 200, 220
        data_channels = [
            Channel("quantum", label=f"{data_qubit}")
            for data_qubit in rsc_block.data_qubits
        ]
        # In order 121, 211, 131, 201, 321, 221, 111, 011
        ancilla_channels = [
            Channel("quantum", label=f"{ancilla_qubit}")
            for ancilla_qubit in rsc_block.ancilla_qubits
        ]
        classical_channels = [
            Channel("classical", label=f"c_{q}_0")
            for stab in rsc_block.stabilizers
            for q in stab.ancilla_qubits
        ]  # There are 8 measurements in a single round
        qubit_map = {
            "(0,0,0)": data_channels[0],
            "(0,1,0)": data_channels[1],
            "(0,2,0)": data_channels[2],
            "(1,0,0)": data_channels[3],
            "(1,1,0)": data_channels[4],
            "(1,2,0)": data_channels[5],
            "(2,0,0)": data_channels[6],
            "(2,1,0)": data_channels[7],
            "(2,2,0)": data_channels[8],
            "(1,2,1)": ancilla_channels[0],
            "(2,1,1)": ancilla_channels[1],
            "(1,3,1)": ancilla_channels[2],
            "(2,0,1)": ancilla_channels[3],
            "(3,2,1)": ancilla_channels[4],
            "(2,2,1)": ancilla_channels[5],
            "(1,1,1)": ancilla_channels[6],
            "(0,1,1)": ancilla_channels[7],
        }

        expected_rsc_circuit = Circuit(
            "RSC_circuit",
            circuit=[
                [
                    Circuit("Reset_0", channels=[qubit_map["(1,1,1)"]]),
                    Circuit("Reset_0", channels=[qubit_map["(2,2,1)"]]),
                    Circuit("Reset_0", channels=[qubit_map["(1,2,1)"]]),
                    Circuit("Reset_0", channels=[qubit_map["(2,1,1)"]]),
                    Circuit("Reset_0", channels=[qubit_map["(0,1,1)"]]),
                    Circuit("Reset_0", channels=[qubit_map["(3,2,1)"]]),
                    Circuit("Reset_0", channels=[qubit_map["(2,0,1)"]]),
                    Circuit("Reset_0", channels=[qubit_map["(1,3,1)"]]),
                ],
                [
                    Circuit("H", channels=[qubit_map["(1,1,1)"]]),
                    Circuit("H", channels=[qubit_map["(2,2,1)"]]),
                    Circuit("H", channels=[qubit_map["(1,2,1)"]]),
                    Circuit("H", channels=[qubit_map["(2,1,1)"]]),
                    Circuit("H", channels=[qubit_map["(0,1,1)"]]),
                    Circuit("H", channels=[qubit_map["(3,2,1)"]]),
                    Circuit("H", channels=[qubit_map["(2,0,1)"]]),
                    Circuit("H", channels=[qubit_map["(1,3,1)"]]),
                ],
                [
                    Circuit(
                        "CZ", channels=[qubit_map["(1,1,1)"], qubit_map["(1,0,0)"]]
                    ),
                    Circuit(
                        "CZ", channels=[qubit_map["(2,2,1)"], qubit_map["(2,1,0)"]]
                    ),
                    Circuit(
                        "CX", channels=[qubit_map["(1,2,1)"], qubit_map["(1,1,0)"]]
                    ),
                    Circuit(
                        "CX", channels=[qubit_map["(2,1,1)"], qubit_map["(2,0,0)"]]
                    ),
                    Circuit(
                        "CX", channels=[qubit_map["(0,1,1)"], qubit_map["(0,0,0)"]]
                    ),
                    Circuit(
                        "CZ", channels=[qubit_map["(1,3,1)"], qubit_map["(1,2,0)"]]
                    ),
                ],
                [
                    Circuit(
                        "CZ", channels=[qubit_map["(1,1,1)"], qubit_map["(0,0,0)"]]
                    ),
                    Circuit(
                        "CZ", channels=[qubit_map["(2,2,1)"], qubit_map["(1,1,0)"]]
                    ),
                    Circuit(
                        "CX", channels=[qubit_map["(1,2,1)"], qubit_map["(1,2,0)"]]
                    ),
                    Circuit(
                        "CX", channels=[qubit_map["(2,1,1)"], qubit_map["(2,1,0)"]]
                    ),
                    Circuit(
                        "CX", channels=[qubit_map["(0,1,1)"], qubit_map["(0,1,0)"]]
                    ),
                    Circuit(
                        "CZ", channels=[qubit_map["(1,3,1)"], qubit_map["(0,2,0)"]]
                    ),
                ],
                [
                    Circuit(
                        "CZ", channels=[qubit_map["(1,1,1)"], qubit_map["(1,1,0)"]]
                    ),
                    Circuit(
                        "CZ", channels=[qubit_map["(2,2,1)"], qubit_map["(2,2,0)"]]
                    ),
                    Circuit(
                        "CX", channels=[qubit_map["(1,2,1)"], qubit_map["(0,1,0)"]]
                    ),
                    Circuit(
                        "CX", channels=[qubit_map["(2,1,1)"], qubit_map["(1,0,0)"]]
                    ),
                    Circuit(
                        "CX", channels=[qubit_map["(3,2,1)"], qubit_map["(2,1,0)"]]
                    ),
                    Circuit(
                        "CZ", channels=[qubit_map["(2,0,1)"], qubit_map["(2,0,0)"]]
                    ),
                ],
                [
                    Circuit(
                        "CZ", channels=[qubit_map["(1,1,1)"], qubit_map["(0,1,0)"]]
                    ),
                    Circuit(
                        "CZ", channels=[qubit_map["(2,2,1)"], qubit_map["(1,2,0)"]]
                    ),
                    Circuit(
                        "CX", channels=[qubit_map["(1,2,1)"], qubit_map["(0,2,0)"]]
                    ),
                    Circuit(
                        "CX", channels=[qubit_map["(2,1,1)"], qubit_map["(1,1,0)"]]
                    ),
                    Circuit(
                        "CX", channels=[qubit_map["(3,2,1)"], qubit_map["(2,2,0)"]]
                    ),
                    Circuit(
                        "CZ", channels=[qubit_map["(2,0,1)"], qubit_map["(1,0,0)"]]
                    ),
                ],
                [
                    Circuit("H", channels=[qubit_map["(1,1,1)"]]),
                    Circuit("H", channels=[qubit_map["(2,2,1)"]]),
                    Circuit("H", channels=[qubit_map["(1,2,1)"]]),
                    Circuit("H", channels=[qubit_map["(2,1,1)"]]),
                    Circuit("H", channels=[qubit_map["(0,1,1)"]]),
                    Circuit("H", channels=[qubit_map["(3,2,1)"]]),
                    Circuit("H", channels=[qubit_map["(2,0,1)"]]),
                    Circuit("H", channels=[qubit_map["(1,3,1)"]]),
                ],
                [
                    Circuit(
                        "Measurement",
                        channels=[qubit_map["(1,1,1)"], classical_channels[0]],
                    ),
                    Circuit(
                        "Measurement",
                        channels=[qubit_map["(2,2,1)"], classical_channels[1]],
                    ),
                    Circuit(
                        "Measurement",
                        channels=[qubit_map["(1,2,1)"], classical_channels[2]],
                    ),
                    Circuit(
                        "Measurement",
                        channels=[qubit_map["(2,1,1)"], classical_channels[3]],
                    ),
                    Circuit(
                        "Measurement",
                        channels=[qubit_map["(0,1,1)"], classical_channels[4]],
                    ),
                    Circuit(
                        "Measurement",
                        channels=[qubit_map["(3,2,1)"], classical_channels[5]],
                    ),
                    Circuit(
                        "Measurement",
                        channels=[qubit_map["(2,0,1)"], classical_channels[6]],
                    ),
                    Circuit(
                        "Measurement",
                        channels=[qubit_map["(1,3,1)"], classical_channels[7]],
                    ),
                ],
            ],
            channels=data_channels + ancilla_channels + classical_channels,
        )
        meas_block_op = MeasureBlockSyndromes(rsc_block.unique_label)
        input_eka = Eka(lattice, blocks=[rsc_block], operations=[meas_block_op])
        base_step = InterpretationStep(
            block_history=((rsc_block,),),
        )
        output_step = CodeApplicator(input_eka).apply(
            deepcopy(base_step), meas_block_op, same_timeslice=False, debug_mode=True
        )

        # First retrieve the output circuit
        meas_synd_circuit = output_step.intermediate_circuit_sequence[0][0]
        # The Circuit has the right number of timesteps.
        self.assertEqual(len(meas_synd_circuit.circuit), 8)
        self.assertEqual(meas_synd_circuit.name, "measure q1 syndromes 1 time(s)")
        # Test that circuits are equivalent
        self.assertEqual(expected_rsc_circuit, meas_synd_circuit)
        # The order of gates within a timestep does not matter but the gate and the
        # channels should be the same.
        self.assertEqual(
            [
                {
                    (
                        gate.name,
                        tuple((chan.label, chan.type) for chan in gate.channels),
                    )
                    for gate in timestep
                }
                for timestep in meas_synd_circuit.circuit
            ],
            [
                {
                    (
                        gate.name,
                        tuple((chan.label, chan.type) for chan in gate.channels),
                    )
                    for gate in timestep
                }
                for timestep in expected_rsc_circuit.circuit
            ],
        )
        self.assertEqual(output_step.block_qec_rounds[rsc_block.uuid], 1)
        # Test that the syndromes are created correctly
        expected_syndromes = tuple(
            Syndrome(
                stabilizer=stab.uuid,
                measurements=[(f"c_{q}", 0) for q in stab.ancilla_qubits],
                block=rsc_block.uuid,
                round=0,
                corrections=(),
            )
            for stab in rsc_block.stabilizers
        )
        self.assertEqual(output_step.syndromes, expected_syndromes)

        # Measure for 2 cycles
        classical_channels2 = [
            Channel("classical", label=f"c_{q}_1")
            for stab in rsc_block.stabilizers
            for q in stab.ancilla_qubits
        ]
        meas_block_op2 = MeasureBlockSyndromes(rsc_block.unique_label, 2)
        input_eka2 = Eka(lattice, blocks=[rsc_block], operations=[meas_block_op2])
        expected_rsc_circuit2 = expected_rsc_circuit.clone(
            channels=data_channels + ancilla_channels + classical_channels2
        )
        expected_total_synd_circ = Circuit(
            "two rounds",
            circuit=expected_rsc_circuit.circuit + expected_rsc_circuit2.circuit,
        )

        output_step_2 = CodeApplicator(input_eka2).apply(
            deepcopy(base_step), meas_block_op2, same_timeslice=False, debug_mode=True
        )
        # First retrieve the output circuit
        meas_synd_circuit_2 = output_step_2.intermediate_circuit_sequence[0][0]
        self.assertEqual(len(meas_synd_circuit_2.circuit), 16)
        self.assertEqual(meas_synd_circuit_2.name, "measure q1 syndromes 2 time(s)")
        # Test that circuits are equivalent
        self.assertEqual(meas_synd_circuit_2, expected_total_synd_circ)
        # The Circuit in the InterpretationStep object returned by the Applicator should
        # be the same as the Syndrome Circuit.
        self.assertEqual(
            [
                {
                    (
                        gate.name,
                        tuple((chan.label, chan.type) for chan in gate.channels),
                    )
                    for gate in timestep
                }
                for timestep in meas_synd_circuit_2.circuit
            ],
            [
                {
                    (
                        gate.name,
                        tuple((chan.label, chan.type) for chan in gate.channels),
                    )
                    for gate in timestep
                }
                for timestep in expected_total_synd_circ.circuit
            ],
        )
        self.assertEqual(output_step_2.block_qec_rounds[rsc_block.uuid], 2)
        # Test that the syndromes are created correctly
        expected_syndromes2 = tuple(
            Syndrome(
                stabilizer=stab.uuid,
                measurements=[(f"c_{q}", 0) for q in stab.ancilla_qubits],
                block=rsc_block.uuid,
                round=0,
                corrections=(),
            )
            for stab in rsc_block.stabilizers
        ) + tuple(
            Syndrome(
                stabilizer=stab.uuid,
                measurements=[(f"c_{q}", 1) for q in stab.ancilla_qubits],
                block=rsc_block.uuid,
                round=1,
                corrections=(),
            )
            for stab in rsc_block.stabilizers
        )
        self.assertEqual(output_step_2.syndromes, expected_syndromes2)

    def test_measureblocksyndromes_detectors(self):
        """Test that the detectors generated for the operation MeasureBlockSyndrome
        for n_cylces > 1 are correct
        """
        n_cycles = 4
        lattice = Lattice.square_2d((10, 20))
        rsc_block = self.rot_surf_code_1
        meas_block_op = MeasureBlockSyndromes(rsc_block.unique_label, n_cycles=n_cycles)
        input_eka = Eka(lattice, blocks=[rsc_block], operations=[meas_block_op])
        base_step = InterpretationStep(
            block_history=((rsc_block,),),
        )

        output_step = CodeApplicator(input_eka).apply(
            base_step, meas_block_op, same_timeslice=False, debug_mode=True
        )

        generated_detectors = output_step.detectors

        expected_detectors = []
        for syndrome in output_step.syndromes:
            prev_syndrome = output_step.get_prev_syndrome(
                syndrome.stabilizer, rsc_block.uuid, current_round=syndrome.round
            )
            if prev_syndrome != []:
                expected_detectors.append(
                    Detector(
                        (
                            prev_syndrome[0],
                            syndrome,
                        )
                    )
                )
        self.assertSetEqual(set(expected_detectors), set(generated_detectors))


if __name__ == "__main__":
    unittest.main()
