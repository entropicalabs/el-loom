"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

import math
import random
import unittest
from pydantic import Field

from loom.eka import (
    Eka,
    Lattice,
    Channel,
    Circuit,
    Block,
    Stabilizer,
    PauliOperator,
    ChannelType,
    SyndromeCircuit,
)
from loom.executor import (
    ApplicationMode,
    AsymmetricDepolarizeCEM,
    CircuitErrorModel,
    ErrorProbProtocol,
    ErrorType,
    HomogeneousTimeDependentCEM,
    HomogeneousTimeIndependentCEM,
)
from loom.eka.operations import MeasureBlockSyndromes
from loom.interpreter import interpret_eka

# pylint: disable=too-many-instance-attributes, duplicate-code
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


# pylint: disable=missing-class-docstring
class TestCircuitErrorModel(unittest.TestCase):
    """Test the CircuitErrorModel class."""

    def setUp(self):
        """
        Set up a circuit for a rsc with a measurement operation.
        Set up a default random gate duration dictionary.
        """
        self.square_2d_lattice = Lattice.square_2d((4, 4))
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
        self.meas_block_op = MeasureBlockSyndromes(
            self.rot_surf_code_1.unique_label, n_cycles=3
        )
        self.eka_sf = Eka(
            self.square_2d_lattice,
            blocks=[self.rot_surf_code_1],
            operations=[self.meas_block_op],
        )
        self.meas_circ = interpret_eka(self.eka_sf).final_circuit
        self.gate_durations = {gate: random.uniform(0, 0.01) for gate in gate_set}

    def test_defining_simple_error_model(self):
        """Test defining a simple error model with a fixed error probability for each
        gate."""

        class SimpleErrorModel(CircuitErrorModel):
            is_time_dependent: bool = False
            error_type: ErrorType = ErrorType.PAULI_X
            gate_error_probabilities: dict[str, ErrorProbProtocol] = (
                CircuitErrorModel.validate_gate_error_probabilities(
                    {
                        "x": lambda _: [0.01],
                        "y": lambda _: [0.02],
                        "z": lambda _: [0.03],
                        "reset_0": lambda _: [0.04],
                    }
                )
            )
            application_mode: ApplicationMode = ApplicationMode.BEFORE_GATE

        try:
            instance = SimpleErrorModel(circuit=self.meas_circ)
        except Exception as e:  # pylint: disable=broad-exception-caught
            self.fail(f"Instantiation failed with exception: {e}")

        # Optionally check the instance type
        self.assertIsInstance(instance, SimpleErrorModel)
        self.assertEqual(instance.error_type, ErrorType.PAULI_X)
        # The first gate is a reset_0 operation
        self.assertEqual(
            instance.get_gate_error_probability(
                self.meas_circ.circuit[0][0].circuit[0][0]
            ),
            [0.04],
        )

    def test_defining_error_model_with_tick_application(self):
        """Test defining an error model with error probability applied at tick."""

        class TickApplicationErrorModel(CircuitErrorModel):
            is_time_dependent: bool = Field(default=True, init=False)
            error_type: ErrorType = Field(default=ErrorType.PAULI_Z, init=False)
            gate_error_probabilities: dict[str, ErrorProbProtocol] = (
                CircuitErrorModel.validate_gate_error_probabilities(
                    {
                        "h": lambda _: [0.05],
                    }
                )
            )
            gate_durations: dict[str, float] = self.gate_durations
            global_time_error_probability: ErrorProbProtocol = lambda t, _=None: [
                0.2 * t
            ]
            application_mode: ApplicationMode = ApplicationMode.END_OF_TICK

        instance = TickApplicationErrorModel(circuit=self.meas_circ)
        self.assertIsInstance(instance, TickApplicationErrorModel)
        self.assertEqual(instance.error_type, ErrorType.PAULI_Z)
        # Check the error probability for the h gate
        # The first gate is a reset_0 operation
        self.assertAlmostEqual(
            instance.get_tick_error_probability(1),
            [(self.gate_durations["reset_0"] + self.gate_durations["h"]) * 0.2],
            places=10,
        )

    def test_defining_error_model_with_time_dependency(self):
        """Test defining a time-dependent error model with error probability as a
        function of time for each gate."""

        class TimeDependentErrorModel(CircuitErrorModel):
            is_time_dependent: bool = True
            error_type: ErrorType = ErrorType.PAULI_Y
            gate_error_probabilities: dict[str, ErrorProbProtocol] = (
                CircuitErrorModel.validate_gate_error_probabilities(
                    {"h": lambda t: [0.04 * t]}
                )
            )

            application_mode: ApplicationMode = ApplicationMode.AFTER_GATE

        instance = TimeDependentErrorModel(
            circuit=self.meas_circ, gate_durations=self.gate_durations
        )

        self.assertIsInstance(instance, TimeDependentErrorModel)
        self.assertEqual(instance.error_type, ErrorType.PAULI_Y)
        # Check the error probability for the h gate at step 1
        self.assertEqual(
            instance.get_gate_error_probability(
                self.meas_circ.circuit[0][0].circuit[1][2]
            ),
            # step 0 is reset_0 op, then h gate starts and since the error is applied
            # after the gate, we use the time of the gate plus the reset_0 duration.
            [(self.gate_durations["reset_0"] + self.gate_durations["h"]) * 0.04],
        )
        # Check undefined error probability return empty instruction
        self.assertEqual(
            instance.get_gate_error_probability(
                self.meas_circ.circuit[0][0].circuit[0][0]
            ),
            None,
        )

        # Check that what would be a typical converter workflow runs without errors
        unrolled = Circuit.unroll(self.meas_circ)
        error = []
        for layer in unrolled:
            for gate in layer:
                if instance.get_gate_error_probability(gate):
                    error.append(
                        (instance.error_type, instance.get_gate_error_probability(gate))
                    )

    def test_error_handling(self):
        """Test error handling for invalid gate names in the error model."""

        # Should raise ValidationError if gate_error_probabilities is not a callable
        with self.assertRaises(TypeError):
            _ = CircuitErrorModel(
                circuit=self.meas_circ,
                gate_error_probabilities={
                    "x": "not_a_callable",
                },
                is_time_dependent=False,
                error_type=ErrorType.PAULI_X,
                application_mode=ApplicationMode.BEFORE_GATE,
            )

        # It is barely impossible to validate this properly.
        # Should raise ValidationError if gate_error_probabilities lambda does not
        # return a list or float
        # with self.assertRaises(TypeError):
        #     instance = CircuitErrorModel(
        #         circuit=self.meas_circ,
        #         is_time_dependent=False,
        #         gate_error_probabilities={
        #             "x": lambda _: 0.01,
        #         },
        #         error_type=ErrorType.PAULI_X,
        #         application_mode=ApplicationMode.BEFORE_GATE,
        #     )

        with self.assertRaises(ValueError):
            # Should raise ValueError if time dependent and gate_duration not defined
            _ = CircuitErrorModel(
                circuit=self.meas_circ,
                is_time_dependent=True,
                error_type=ErrorType.PAULI_X,
                application_mode=ApplicationMode.BEFORE_GATE,
            )

    def test_homogeneous_cem(self):
        """Test defining a homogeneous circuit error model with a fixed error
        probability for each gate."""
        unrolled = Circuit.unroll(self.meas_circ)
        target_gates = {op.name for layer in unrolled for op in layer if op is not None}

        model_time_dep = HomogeneousTimeDependentCEM(
            circuit=self.meas_circ,
            error_probability=lambda t, _=None: [1 / t if t != 0 else 0],
            gate_durations=self.gate_durations,
            error_type=ErrorType.PAULI_X,
            application_mode=ApplicationMode.AFTER_GATE,
            target_gates=target_gates,
        )

        self.assertEqual(
            model_time_dep.get_gate_error_probability(
                self.meas_circ.circuit[0][0].circuit[0][0]
            ),
            [1 / (self.gate_durations["reset_0"])],
        )
        model_time_indep = HomogeneousTimeIndependentCEM(
            circuit=self.meas_circ,
            error_probability=0.1,
            error_type=ErrorType.PAULI_X,
            application_mode=ApplicationMode.BEFORE_GATE,
            target_gates=target_gates,
        )

        self.assertEqual(
            model_time_indep.get_gate_error_probability(
                self.meas_circ.circuit[0][0].circuit[0][0]
            ),
            [0.1],
        )

    def test_asymmetric_depolarize(self):
        """Test defining an asymmetric depolarizing error model."""
        unrolled = Circuit.unroll(self.meas_circ)
        g_set = {op.name for layer in unrolled for op in layer if op is not None}
        t1 = 0.01
        t2 = 0.02
        gate_duration = 0.01
        step = 2
        model = AsymmetricDepolarizeCEM(
            circuit=self.meas_circ,
            t1=t1,
            t2=t2,
            gate_durations={g: gate_duration for g in g_set},
        )

        t = gate_duration
        exp_t_t1 = math.exp(-t / t1)
        px = py = (1 - exp_t_t1) / 4
        exp_t_t2 = math.exp(-t / t2)
        pz = (1 - exp_t_t2) / 2 - px
        expected = [px, py, pz]

        self.assertEqual(
            model.get_tick_error_probability(step),
            expected,
        )

    def test_idle_model(self):
        """Test idle error model with a fixed error probability."""
        unrolled = Circuit.unroll(self.meas_circ)
        g_set = {op.name for layer in unrolled for op in layer if op is not None}
        gate_durations = {g: 0.01 for g in g_set if g != "cz"}
        gate_durations["cz"] = 0.02
        step = 2
        model = CircuitErrorModel(
            circuit=self.meas_circ,
            error_type=ErrorType.PAULI_X,
            application_mode=ApplicationMode.IDLE_END_OF_TICK,
            is_time_dependent=True,
            gate_durations=gate_durations,
            global_time_error_probability=lambda t, tt: [2 * tt],
        )
        expected = [0.02]
        self.assertEqual(
            model.get_idle_tick_error_probability(
                tick_index=step,
                channel_id=unrolled[3][2].channels[0].id,
            ),
            expected,
        )

    def test_asymmetric_depolarize_zero_t1_t2(self):
        """Test asymmetric depolarizing error model with t1 and t2 set to 0."""
        unrolled = Circuit.unroll(self.meas_circ)
        g_set = {op.name for layer in unrolled for op in layer if op is not None}
        t1 = 0
        t2 = 0
        gate_duration = 0.01
        step = 2
        model = AsymmetricDepolarizeCEM(
            circuit=self.meas_circ,
            t1=t1,
            t2=t2,
            gate_durations={g: gate_duration for g in g_set},
        )

        expected = [0.25, 0.25, 0.25]
        self.assertEqual(
            model.get_tick_error_probability(step),
            expected,
        )

    def test_error_model_on_simple_circuit(self):
        """Test a wide range of error models on a simple circuit with well-defined properties."""

        channels = [Channel() for _ in range(4)]
        # Define a simple circuit and input parameters for the error model
        simple_circuit_test = {
            "channels": channels,
            "circuit": Circuit(
                name="simple_circuit",
                circuit=[
                    [
                        Circuit(
                            name="x", channels=channels[0]
                        ),  # .circuit[0][0] is x on q0
                        Circuit(
                            name="y", channels=channels[1]
                        ),  # .circuit[0][1] is y on q1
                        Circuit(
                            name="cz", channels=[channels[2], channels[3]]
                        ),  # .circuit[0][2] is cz on q2, q3
                    ],
                    [
                        Circuit(
                            name="z", channels=channels[0]
                        ),  # .circuit[1][0] is z on q0
                        # idle on 1, 2, 3
                    ],
                    [
                        Circuit(
                            name="measurement", channels=channels[0]
                        ),  # .circuit[2][0] is measure on q0
                        Circuit(
                            name="measurement", channels=channels[1]
                        ),  # .circuit[2][1] is measure on q1
                        Circuit(
                            name="measurement", channels=channels[2]
                        ),  # .circuit[2][2] is measure on q2
                        # idle on 3
                    ],
                ],
            ),
            "gate_durations": {"x": 2, "y": 2, "z": 2, "cz": 4, "measurement": 1},
            "tick_durations": [4, 2, 1],  # durations for each tick
            "idle_times": {
                "q0": [2, 0, 0],
                "q1": [2, 2, 0],
                "q2": [0, 2, 0],
                "q3": [0, 2, 1],
            },
            "gate_error_probabilities_time_independent": {
                "x": lambda _: [0.01],
                "y": lambda _: [0.02],
                "z": lambda _: [0.03],
                "cz": lambda _: [0.04],
                "measurement": lambda _: [0.05],
            },
            "gate_error_probabilities_time_dependent": {
                "x": lambda t: [0.01 * t],
                "y": lambda t: [0.02 * t],
                "z": lambda t: [0.03 * t],
                "cz": lambda t: [0.04 * t],
                "measurement": lambda t: [0.05 * t],
            },
            "global_time_error_probability_time_independant": lambda _, __: [0.1],
            "global_time_error_probability_time_dependent": lambda t, _: [0.1 * t],
            "global_time_error_probability_with_tick_time_param": lambda _, tt: [
                0.2 * tt
            ],
            "global_time_error_probability_with_both_params": lambda t, tt: [
                0.2 * t + 0.01 * tt
            ],
        }

        with self.subTest("Time Independent X Error Model"):
            model = CircuitErrorModel(
                circuit=simple_circuit_test["circuit"],
                is_time_dependent=False,
                gate_error_probabilities=simple_circuit_test[
                    "gate_error_probabilities_time_independent"
                ],
                error_type=ErrorType.PAULI_X,
                application_mode=ApplicationMode.BEFORE_GATE,
            )
            self.assertEqual(
                model.get_gate_error_probability(model.circuit.circuit[0][0]),
                simple_circuit_test["gate_error_probabilities_time_independent"]["x"](
                    0
                ),
            )
            self.assertEqual(
                model.get_gate_error_probability(model.circuit.circuit[0][2]),
                simple_circuit_test["gate_error_probabilities_time_independent"]["cz"](
                    None
                ),
            )
            self.assertEqual(
                model.get_gate_error_probability(model.circuit.circuit[1][0]),
                simple_circuit_test["gate_error_probabilities_time_independent"]["z"](
                    None
                ),
            )
        with self.subTest("Time Dependent Y Error Model, Before Gate"):
            model = CircuitErrorModel(
                circuit=simple_circuit_test["circuit"],
                is_time_dependent=True,
                gate_error_probabilities=simple_circuit_test[
                    "gate_error_probabilities_time_dependent"
                ],
                gate_durations=simple_circuit_test["gate_durations"],
                global_time_error_probability=simple_circuit_test[
                    "global_time_error_probability_time_dependent"
                ],
                error_type=ErrorType.PAULI_Y,
                application_mode=ApplicationMode.BEFORE_GATE,
            )
            # P = 0 so it should return None
            self.assertIsNone(
                model.get_gate_error_probability(model.circuit.circuit[0][0]),
            )
            # P = 0 so it should return None
            self.assertIsNone(
                model.get_gate_error_probability(model.circuit.circuit[0][1]),
            )
            self.assertEqual(
                model.get_gate_error_probability(model.circuit.circuit[1][0]),
                simple_circuit_test["gate_error_probabilities_time_dependent"]["z"](4),
            )
        with self.subTest("Time Dependent Z Error Model, After Gate"):
            model = CircuitErrorModel(
                circuit=simple_circuit_test["circuit"],
                is_time_dependent=True,
                gate_error_probabilities=simple_circuit_test[
                    "gate_error_probabilities_time_dependent"
                ],
                gate_durations=simple_circuit_test["gate_durations"],
                error_type=ErrorType.PAULI_Z,
                application_mode=ApplicationMode.AFTER_GATE,
            )
            self.assertEqual(
                model.get_gate_error_probability(model.circuit.circuit[0][0]),
                simple_circuit_test["gate_error_probabilities_time_dependent"]["x"](2),
            )
            self.assertEqual(
                model.get_gate_error_probability(model.circuit.circuit[0][2]),
                simple_circuit_test["gate_error_probabilities_time_dependent"]["cz"](4),
            )
            self.assertEqual(
                model.get_gate_error_probability(model.circuit.circuit[1][0]),
                simple_circuit_test["gate_error_probabilities_time_dependent"]["z"](6),
            )
            self.assertEqual(
                model.get_gate_error_probability(model.circuit.circuit[2][0]),
                simple_circuit_test["gate_error_probabilities_time_dependent"][
                    "measurement"
                ](7),
            )
        with self.subTest(
            "X Error Model, End of Tick, Time Independent, no tick time param"
        ):
            model = CircuitErrorModel(
                circuit=simple_circuit_test["circuit"],
                global_time_error_probability=simple_circuit_test[
                    "global_time_error_probability_time_independant"
                ],
                is_time_dependent=False,
                gate_durations=simple_circuit_test["gate_durations"],
                error_type=ErrorType.PAULI_X,
                application_mode=ApplicationMode.END_OF_TICK,
            )
            self.assertEqual(
                model.get_tick_error_probability(),
                simple_circuit_test["global_time_error_probability_time_independant"](
                    None, None
                ),
            )
            self.assertEqual(
                model.get_tick_error_probability(),
                simple_circuit_test["global_time_error_probability_time_independant"](
                    None, None
                ),
            )
        with self.subTest(
            "X Error Model, End of Tick, Time Dependent, with tick time param"
        ):
            model = CircuitErrorModel(
                circuit=simple_circuit_test["circuit"],
                is_time_dependent=True,
                global_time_error_probability=simple_circuit_test[
                    "global_time_error_probability_with_both_params"
                ],
                gate_durations=simple_circuit_test["gate_durations"],
                error_type=ErrorType.PAULI_Z,
                application_mode=ApplicationMode.END_OF_TICK,
            )
            self.assertEqual(
                model.get_tick_error_probability(0),
                simple_circuit_test["global_time_error_probability_with_both_params"](
                    4, 4
                ),
            )
            self.assertEqual(
                model.get_tick_error_probability(1),
                simple_circuit_test["global_time_error_probability_with_both_params"](
                    6, 2
                ),
            )
        with self.subTest(
            "X Error Model, Idle End of Tick, Time Dependent, with tick time param"
        ):
            model = CircuitErrorModel(
                circuit=simple_circuit_test["circuit"],
                is_time_dependent=True,
                global_time_error_probability=simple_circuit_test[
                    "global_time_error_probability_with_both_params"
                ],
                gate_durations=simple_circuit_test["gate_durations"],
                error_type=ErrorType.PAULI_X,
                application_mode=ApplicationMode.IDLE_END_OF_TICK,
            )

            self.assertEqual(
                model.get_idle_tick_error_probability(
                    0, simple_circuit_test["channels"][0].id
                ),
                simple_circuit_test["global_time_error_probability_with_both_params"](
                    4, 2
                ),
            )
            self.assertEqual(
                model.get_idle_tick_error_probability(
                    1, simple_circuit_test["channels"][1].id
                ),
                simple_circuit_test["global_time_error_probability_with_both_params"](
                    6, 2
                ),
            )
            self.assertEqual(
                model.get_idle_tick_error_probability(
                    1, simple_circuit_test["channels"][0].id
                ),
                simple_circuit_test["global_time_error_probability_with_both_params"](
                    6, 0
                ),
            )


if __name__ == "__main__":
    unittest.main()
