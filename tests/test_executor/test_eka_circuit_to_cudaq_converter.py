"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

from collections import Counter
import re
import unittest
from unittest.mock import MagicMock
from dataclasses import dataclass
from typing import Optional, Any
from scipy.stats import chisquare

# pylint: disable=import-error, wrong-import-position
import pytest

pytest.importorskip("cudaq")
import cudaq
from cudaq import Kernel

from loom.eka import Channel, ChannelType, Circuit
from loom.executor import EkaToCudaqConverter, Converter


@dataclass
class ConverterCircuitTestData:
    """Test data for converter circuits. This is used to test the converter's by
    comparing simulation results to expected results.

    Attributes:
        test_name (str): Name of the test.
        input (Circuit): The input circuit to be tested.
        expected_distribution (Counter):
            The expected distribution of measurement results.
            expected_raw_output (Optional[Any]): Optional raw output data to assert some
        more exact matching, If defined, assertion on the raw output will be done.
        expected_content (Optional[list[str]]):
            Optional list of expected content in the kernel.
            If defined, the content of the kernel will be asserted to match this list.
        num_shots (Optional[int]): Number of shots for the simulation, default is 1000.
    """

    test_name: str
    input: Circuit
    expected_distribution: Counter
    expected_raw_output: Optional[Any] = (
        None  # Allow some raw optional data for exact converter's output matching
    )
    expected_content: Optional[list[str]] = None
    num_shots: Optional[int] = 1000

    # This can be used to compare two sample distributions using the chi-squared test.
    # Since the cudaq.sample is too heavy to run in the CI, this is not used currently.
    @staticmethod
    def assert_chisquare_equal(sample1, sample2):
        """Compare two sample distributions using the chi-squared test.

        This function asserts that `sample1` and `sample2` come from the same
        underlying categorical distribution. It uses the Pearson's chi-squared
        test to evaluate the null hypothesis:
            H0: The two samples are drawn from the same distribution.

        The test uses count histograms (not probabilities) and computes a test
        statistic:
            χ² = Σ_i ((O_i - E_i)^2 / E_i)
        where O_i and E_i are the observed and expected counts, respectively.

        If the null hypothesis (both sample come from the same distribution) is true and
        the sample size is large, then:
            χ² ~ χ²_{k - 1}
        and the resulting p-value will tend toward a uniform distribution on [0,1].
        Thus, this test will:
        - Almost always pass when the distributions are the same (for large n),
        - Almost always fail when the distributions are different (for large n).
        """
        # Count occurrences
        c1 = Counter(sample1)
        c2 = Counter(sample2)
        # Union of outcomes to ensure we have common keys
        outcomes = sorted(set(c1.keys()) | set(c2.keys()))
        if len(outcomes) == 1 and len(c1) == 1 and len(c2) == 1:
            # If both samples have only one outcome, they are trivially equal
            return
        # Raw frequencies
        # Avoid zero expected frequencies (required by chi-squared test)
        f1 = [c1.get(k, 0) for k in outcomes]
        f2 = [c2.get(k, 0) for k in outcomes]
        # Normalize to probabilities (or relative frequencies)
        total1 = sum(f1)
        total2 = sum(f2)
        significance_level = 0.05

        # Handle zero-total edge case
        if total1 == 0 or total2 == 0:
            raise ValueError("One of the samples is empty or all-zero.")

        if total1 < total2:
            p1 = [
                v * total2 / total1 + 1e-15 for v in f1
            ]  # Add small value to avoid division by zero
            p2 = [v + 1e-15 for v in f2]
        else:
            p1 = [v + 1e-15 for v in f1]
            p2 = [v * total1 / total2 + 1e-15 for v in f2]

        # Chi-squared test
        _, p_value = chisquare(f_obs=p1, f_exp=p2)
        # Assert similarity
        assert p_value >= significance_level


# pylint: disable=redefined-builtin
def get_n_qchannel(n: int, type: ChannelType = ChannelType.QUANTUM) -> tuple[Channel]:
    """Create a tuple of quantum channels with labels "q0", "q1", ..., "qn-1".

    Parameters
    ----------
    n : int
        number of quantum channels to create.

    Returns
    -------
    tuple[Channel]
        Channels with labels "q0", "q1", ..., "qn-1".

    Raises
    ------
    ValueError
        If n is less than or equal to 0.
    """
    if n <= 0:
        raise ValueError
    if type == ChannelType.CLASSICAL:
        return tuple(
            Channel(label=f"c{i}", type=ChannelType.CLASSICAL) for i in range(n)
        )
    return tuple(Channel(label=f"q{i}", type=ChannelType.QUANTUM) for i in range(n))


class TestEkaCircuitToCudaqConverter(unittest.TestCase):
    """Test the EkaToCudaqConverter."""

    @property
    def converter(self) -> ConverterCircuitTestData:
        """Return the converter instance to be tested."""
        return EkaToCudaqConverter()

    # pylint: disable=too-many-locals
    def setUp(self):
        """Set up the test case with a converter instance."""
        # List all test data
        # Define test data for converter circuits

        # Some of these data won't be used for the tests. The test will assert all the
        # ConverterCircuitTestData are converted to cudaq.Kernel without errors.
        # The test will NOT assert the expected distribution of measurement results.
        # The expected content of the kernel will be asserted if defined.

        cudaq.set_target("qpp-cpu")  # Set the target to CPU for testing

        # Simple circuit that applies H and mx
        channels = get_n_qchannel(1)
        meas_x = ConverterCircuitTestData(
            "meas_x_test",
            Circuit(
                name="meas_x_test",
                circuit=[
                    Circuit(
                        name="h", channels=[channels[0]]
                    ),  # Hadamard gate to change basis
                    Circuit(
                        name="measure_x", channels=[channels[0]]
                    ),  # Measure in X basis
                ],
            ),
            Counter({"0": 1}),
            num_shots=1,
            expected_content=["h", "mx"],
        )

        # Simple circuit that test I and mz
        channels = get_n_qchannel(1)
        identity_test = ConverterCircuitTestData(
            "identity_test",
            Circuit(
                name="identity_test",
                circuit=[
                    Circuit(name="i", channels=[channels[0]]),  # Identity gate
                    Circuit(
                        name="measure_z", channels=[channels[0]]
                    ),  # Measure in Z basis
                ],
            ),
            Counter({"0": 1}),
            num_shots=1,
            expected_content=["mz"],
            expected_raw_output=(
                "module attributes {quake.mangled_name_map = {"
                "__nvqpp__mlirgen____nvqppBuilderKernel_NN3IRTVZAC = "
                '"__nvqpp__mlirgen____nvqppBuilderKernel_NN3IRTVZAC_PyKernelEntryPointRewrite"'
                "}} {\n"
                "  func.func @__nvqpp__mlirgen____nvqppBuilderKernel_NN3IRTVZAC()"
                ' attributes {"cudaq-entrypoint", "cudaq-kernel"} {\n'
                "    %0 = quake.alloca !quake.ref\n"
                '    %measOut = quake.mz %0 name "q0" : (!quake.ref) -> !quake.measure\n'
                "    return\n"
                "  }\n"
                "}\n"
            ),
        )

        # Simple circuit that test my behavior
        channels = get_n_qchannel(1)
        meas_y = ConverterCircuitTestData(
            "meas_y_test",
            Circuit(
                name="meas_y_test",
                circuit=[
                    Circuit(
                        name="h", channels=[channels[0]]
                    ),  # Hadamard gate to change basis
                    Circuit(
                        name="phase", channels=[channels[0]]
                    ),  # Phase gate to change basis
                    Circuit(
                        name="measure_y", channels=[channels[0]]
                    ),  # Measure in Y basis
                ],
            ),
            Counter({"0": 1}),
            num_shots=1,
            expected_content=["h", "s", "my"],
        )

        # This is a simple X gate test circuit that applies X gates to two qubits and
        # measures them.
        # Fully deterministic circuit and test are runs without noise,
        # so we can use 1 shot
        channels = get_n_qchannel(2)
        x_gate = ConverterCircuitTestData(
            "x_gate",
            Circuit(
                name="x_gate",
                circuit=[
                    [
                        Circuit(name="x", channels=[channels[0]]),
                        Circuit(name="x", channels=[channels[1]]),
                    ],
                    [
                        Circuit(name="x", channels=[channels[1]]),
                    ],
                    [
                        Circuit(name="measure_z", channels=[channels[0]]),
                        Circuit(name="measure_z", channels=[channels[1]]),
                    ],
                ],
            ),
            Counter({"00": 0, "01": 0, "10": 1, "11": 0}),
            num_shots=1,
            expected_content=["x", "x", "x", "mz", "mz"],
        )

        # This is a Bell state circuit that creates a Bell state
        # (|00> + |11>)/sqrt(2) and measures the qubits.
        channels = get_n_qchannel(2)
        bell_state = ConverterCircuitTestData(
            "bell_state",
            Circuit(
                name="bell_state",
                circuit=[
                    [Circuit(name="h", channels=[channels[0]])],
                    [
                        Circuit(name="cnot", channels=[channels[0], channels[1]]),
                    ],
                    [
                        Circuit(name="measure_z", channels=[channels[0]]),
                        Circuit(name="measure_z", channels=[channels[1]]),
                    ],
                ],
            ),
            Counter({"00": 1, "01": 0, "10": 0, "11": 1}),
            expected_content=["h", "cx", "mz", "mz"],
        )

        # This is a single Pauli gate circuit that applies X, Y, and Z gates
        # to three qubits and measures them.
        # Fully deterministic circuit and test are runs without noise,
        # so we can use 1 shot
        channels = get_n_qchannel(3)
        single_pauli_gate = ConverterCircuitTestData(
            "single_pauli_gate",
            Circuit(
                name="single_pauli_gate",
                circuit=[
                    [
                        Circuit(name="x", channels=[channels[0]]),
                        Circuit(name="y", channels=[channels[1]]),
                        Circuit(name="z", channels=[channels[2]]),
                    ],
                    [
                        Circuit(name="measure_z", channels=[channels[0]]),
                        Circuit(name="measure_z", channels=[channels[1]]),
                        Circuit(name="measure_z", channels=[channels[2]]),
                    ],
                ],
            ),
            Counter(
                {
                    "000": 0,
                    "001": 0,
                    "010": 0,
                    "011": 0,
                    "100": 0,
                    "101": 0,
                    "110": 1,
                    "111": 0,
                }
            ),
            num_shots=1,
            expected_content=["x", "y", "z", "mz", "mz", "mz"],
        )

        # This is a reset circuit that resets all qubits to the different pauli
        # eigenstates and measures them (in z-basis).
        channels = get_n_qchannel(7)
        resets_circuit = ConverterCircuitTestData(
            "all_reset_test",
            Circuit(
                name="all_reset_test",
                circuit=[
                    [
                        # Just a H gate on each qubit that will be reset to z_basis
                        # eigenstates to have some meaningful measurement
                        Circuit(name="h", channels=[channels[0]]),
                        Circuit(name="h", channels=[channels[1]]),
                        Circuit(name="h", channels=[channels[2]]),
                    ],
                    [
                        Circuit(name="reset", channels=[channels[0]]),
                        Circuit(name="reset_0", channels=[channels[1]]),
                        Circuit(name="reset_1", channels=[channels[2]]),
                        Circuit(name="reset_+", channels=[channels[3]]),
                        Circuit(name="reset_-", channels=[channels[4]]),
                        Circuit(name="reset_+i", channels=[channels[5]]),
                        Circuit(name="reset_-i", channels=[channels[6]]),
                    ],
                    [
                        Circuit(
                            name="h", channels=[channels[3]]
                        ),  # From + state, to 0 state
                        Circuit(
                            name="h", channels=[channels[4]]
                        ),  # From - state, to 1 state
                    ],
                    [
                        Circuit(
                            name="measure_z", channels=[channels[0]]
                        ),  # Expected to be reset to 0
                        Circuit(
                            name="measure_z", channels=[channels[1]]
                        ),  # Expected to be reset to 0
                        Circuit(
                            name="measure_z", channels=[channels[2]]
                        ),  # Expected to be reset to 1
                        Circuit(
                            name="measure_z", channels=[channels[3]]
                        ),  # Expected to be 0
                        Circuit(
                            name="measure_z", channels=[channels[4]]
                        ),  # Expected to be 1
                        Circuit(
                            name="measure_z", channels=[channels[5]]
                        ),  # Expected to be in +i state, so 50-50
                        Circuit(
                            name="measure_z", channels=[channels[6]]
                        ),  # Expected to be in -i state, so 50-50
                    ],
                ],
            ),
            Counter(
                {
                    "0010100": 1,
                    "0010101": 1,
                    "0010110": 1,
                    "0010111": 1,
                }
            ),
            num_shots=1000,
        )

        # This is a simple circuit to test phase gates (S, adj(S)) on two qubits.
        channels = get_n_qchannel(2)
        phase_gates = ConverterCircuitTestData(
            "phase_gates",
            Circuit(
                name="phase_gates",
                circuit=[
                    [
                        Circuit(name="h", channels=[channels[0]]),
                        Circuit(name="h", channels=[channels[1]]),
                    ],
                    [
                        Circuit(name="phase", channels=[channels[0]]),
                        Circuit(name="phaseinv", channels=[channels[1]]),
                    ],
                    [
                        Circuit(name="measure_z", channels=[channels[0]]),
                        Circuit(name="measure_z", channels=[channels[1]]),
                    ],
                ],
            ),
            Counter({"01": 1, "10": 1, "11": 1, "00": 1}),
        )
        # This is a simple circuit to test two qubit gates.
        channels = get_n_qchannel(6)
        two_qubit_circuit = ConverterCircuitTestData(
            "two_qubit_circuit",
            Circuit(
                name="two_qubit_circuit",
                circuit=[
                    [
                        Circuit(name="x", channels=[channels[0]]),
                        Circuit(name="x", channels=[channels[1]]),
                        Circuit(name="x", channels=[channels[2]]),
                        Circuit(name="phase", channels=[channels[3]]),
                        Circuit(name="x", channels=[channels[4]]),
                        Circuit(name="y", channels=[channels[5]]),
                    ],
                    [
                        Circuit(name="h", channels=[channels[3]]),
                        Circuit(name="i", channels=[channels[4]]),
                        Circuit(name="phase", channels=[channels[5]]),
                    ],
                    [
                        Circuit(name="cx", channels=[channels[0], channels[1]]),
                        Circuit(name="cy", channels=[channels[2], channels[3]]),
                        Circuit(name="cz", channels=[channels[4], channels[5]]),
                    ],
                    [
                        Circuit(name="measure_z", channels=[channels[0]]),
                        Circuit(name="measure_z", channels=[channels[1]]),
                        Circuit(name="measure_z", channels=[channels[2]]),
                        Circuit(name="measure_z", channels=[channels[3]]),
                        Circuit(name="measure_z", channels=[channels[4]]),
                        Circuit(name="measure_z", channels=[channels[5]]),
                    ],
                ],
            ),
            Counter({"101011": 1, "101111": 1}),
            expected_content=[
                "x",
                "x",
                "x",
                "s",
                "x",
                "y",
                "h",
                "s",
                "cx",
                "cy",
                "cz",
                "mz",
                "mz",
                "mz",
                "mz",
                "mz",
                "mz",
            ],
        )

        # Simple classical control circuit with two qubits.
        q_chan1 = Channel(label="q1")
        q_chan2 = Channel(label="q2")
        c_chan1 = Channel(label="c1", type=ChannelType.CLASSICAL)
        simple_classical_control_1 = ConverterCircuitTestData(
            "simple_classical_control_1",
            Circuit(
                name="simple_classical_control_1",
                circuit=[
                    Circuit(name="reset_+", channels=[q_chan1]),
                    Circuit(name="measure_z", channels=[q_chan1, c_chan1]),
                    Circuit(
                        name="classically_controlled_x", channels=[c_chan1, q_chan2]
                    ),
                    Circuit(name="measure_z", channels=[q_chan1]),
                ],
            ),
            Counter({"00": 1, "11": 1}),
            num_shots=1000,
            expected_content=["reset", "h", "mz", "x", "mz"],
        )

        # Test classicaly controlled for each pauli. The tuple contains the
        # (pauli to apply, measurement basis, init_state) for each Pauli.
        # The circuit will reset the qubits to the different pauli eigenstates,
        # Then apply 3 classically controlled gates, where the condition
        # should be 0, 1, 50-50.
        # Then measure the qubits choosing a relevant basis.
        # We expect c3 = 0, c4 = 1, c5 = 50-50,
        # where c3, c4, c5 are the classical channels
        # The prefix allows to add c to the pauli in order to test the
        # classical controlled cx, cy and cz gates.
        def generate_class_ctrl_pauli_test_data(
            p: str, prefix: str = ""
        ) -> ConverterCircuitTestData:
            map_m_and_r = {
                "x": ("z", "0"),
                "y": ("x", "+"),
                "z": ("y", "+i"),
            }

            m, r = map_m_and_r[p]
            q_chan = get_n_qchannel(6)
            if prefix == "c":
                q_ctrl_chan = Channel(label="qctrl", type=ChannelType.QUANTUM)
            c_chan = get_n_qchannel(6, ChannelType.CLASSICAL)

            reset_step = (
                Circuit(
                    name=f"reset_{r}",
                    channels=[q_chan[3]],
                ),
                Circuit(
                    name=f"reset_{r}",
                    channels=[q_chan[4]],
                ),
                Circuit(
                    name=f"reset_{r}",
                    channels=[q_chan[5]],
                ),
            )

            init_control_step = (
                Circuit(
                    name="i",
                    channels=[q_chan[0]],
                ),
                Circuit(
                    name="x",
                    channels=[q_chan[1]],
                ),
                Circuit(
                    name="h",
                    channels=[q_chan[2]],
                ),
                *(
                    (
                        Circuit(
                            name="x",
                            # pylint: disable=possibly-used-before-assignment
                            channels=[q_ctrl_chan],
                        ),
                    )
                    if prefix == "c"
                    else ()
                ),
            )

            measure_condition = (
                Circuit(name="measure_z", channels=[c_chan[0], q_chan[0]]),
                Circuit(name="measure_z", channels=[c_chan[1], q_chan[1]]),
                Circuit(name="measure_z", channels=[c_chan[2], q_chan[2]]),
            )
            apply_classical_controlled_step = ()
            if prefix == "c":
                apply_classical_controlled_step = (
                    (
                        Circuit(
                            name=f"classically_controlled_{prefix}{p}",
                            channels=[c_chan[0], q_ctrl_chan, q_chan[3]],
                        ),
                    ),
                    (
                        Circuit(
                            name=f"classically_controlled_{prefix}{p}",
                            channels=[c_chan[1], q_ctrl_chan, q_chan[4]],
                        ),
                    ),
                    (
                        Circuit(
                            name=f"classically_controlled_{prefix}{p}",
                            channels=[c_chan[2], q_ctrl_chan, q_chan[5]],
                        ),
                    ),
                )
            else:
                apply_classical_controlled_step = (
                    (
                        Circuit(
                            name=f"classically_controlled_{p}",
                            channels=[c_chan[0], q_chan[3]],
                        ),
                    ),
                    (
                        Circuit(
                            name=f"classically_controlled_{p}",
                            channels=[c_chan[1], q_chan[4]],
                        ),
                    ),
                    (
                        Circuit(
                            name=f"classically_controlled_{p}",
                            channels=[c_chan[2], q_chan[5]],
                        ),
                    ),
                )

            measure_qubits = (
                Circuit(name=f"measure_{m}", channels=[c_chan[3], q_chan[3]]),
                Circuit(name=f"measure_{m}", channels=[c_chan[4], q_chan[4]]),
                Circuit(name=f"measure_{m}", channels=[c_chan[5], q_chan[5]]),
            )

            if prefix == "c":
                measure_qubits += (Circuit(name="measure_z", channels=[q_ctrl_chan]),)

            if prefix == "c":
                expected_distribution = Counter({"0100101": 1, "0110111": 1})
            else:
                expected_distribution = Counter({"010010": 1, "011011": 1})
            test_data = ConverterCircuitTestData(
                f"classical_ctrl_pauli_{prefix}{p}",
                Circuit(
                    name=f"classical_ctrl_pauli_{prefix}{p}",
                    circuit=(reset_step, init_control_step, measure_condition)
                    + apply_classical_controlled_step
                    + (measure_qubits,),
                ),
                expected_distribution,
                num_shots=1000,
            )
            return test_data

        # List of all test data
        self.input_test_data = (
            [
                identity_test,
                meas_y,
                meas_x,
                x_gate,
                bell_state,
                single_pauli_gate,
                resets_circuit,
                phase_gates,
                two_qubit_circuit,
                simple_classical_control_1,
            ]
            + list(generate_class_ctrl_pauli_test_data(p) for p in "xyz")
            + list(generate_class_ctrl_pauli_test_data(p, prefix="c") for p in "xyz")
        )

    # Optionally override assert_raw_output if needed
    def assert_raw_output(self, observed: cudaq.Kernel, expected: cudaq.Kernel):
        """compare the string representation of the observed and expected kernels.
        Since Kernel have Uid, we need to ignore it for the comparison, using some regex.
        """

        def remove_uid(kernel_str: str) -> str:
            return re.sub(
                r"(__nvqpp__mlirgen____nvqppBuilderKernel_)[A-Z0-9]+",
                r"\1<UID>",
                kernel_str,
            )

        self.assertEqual(remove_uid(str(expected)), remove_uid(str(observed)))

    def assert_kernel_content(
        self, expected: cudaq.Kernel, expected_content: list[str]
    ):
        """Assert that the kernel string contains all the expected content.
        This function filters the string representation of the kernel to extract
        quake instructions and checks if they match the expected content. This doesn't
        include classical control flow instructions, it will include all the instructions
        in the branching regardless of the condition. e.g. if the kernel has
        `if (c0) do x()`, The `x` instruction will be included in the content.

        Parameters
        ----------
        expected : cudaq.Kernel
            The expected kernel to check.
        expected_content : list[str]
            The expected content to find in the kernel.
        """
        kernel_str = str(expected)

        def extract_quake_instructions(kernel_str: str) -> list[str]:
            """
            Extracts a list of quake instructions from the given module string.
            Example: For 'quake.reset %1 : (!quake.ref) -> ()', instruction is 'reset'.
            Returns a list of instruction names in the order they appear.
            """
            # Match 'quake.<instruction>'
            # where <instruction> is letters, numbers, or underscores
            pattern = r"quake\.([a-zA-Z0-9_]+)(?: \[(.*?)\])? %"
            instructions = []
            for match in re.finditer(pattern, kernel_str):
                instr, bracketed = match.group(1), match.group(2)
                if instr == "discriminate":
                    continue
                if bracketed is not None:
                    instructions.append(f"c{instr}")
                else:
                    instructions.append(instr)
            return instructions

        content = extract_quake_instructions(kernel_str)
        self.assertEqual(content, expected_content)

    def simulate(
        self, output: "cudaq.kernel", num_shots=1000
    ) -> dict[str, dict[int, int]]:
        """Run the simulation of a given kernel. Using cudaq.sample, this is a noiseless
        simulation.
        Parameters
        ----------
        output : cudaq.kernel
            kerenl to run
        num_shots : int, optional
            number of shots, by default 1000
        Returns
        -------
        dict[str, dict[int, int]]
            the simulation output, a dictionary where keys are measurement labels and
            values are dictionaries of counts for each register.
        """

        # Run the simulation using cudaq.sample
        simulation_output = cudaq.sample(output, shots_count=num_shots)
        return simulation_output

    def test_conversion_single_qubit_gates(self):
        """
        Test that a circuit with every single qubit gates supported by the converter is
        converted to a Kernel without errors.
        """
        one_qubit_gates_to_test = (
            Converter.REQUIRED_Q_OP_SINGLE_QBIT_GATE | Converter.REQUIRED_Q_OP_RESET
        )

        single_qubit_gates = Circuit(
            name="single_qubit_gates",
            circuit=[
                Circuit(name=gate, channels=[Channel(label="q0")])
                for gate in one_qubit_gates_to_test
            ],
        )
        single_qubit_gates_kernel, _, _ = self.converter.convert_circuit(
            single_qubit_gates
        )
        self.assertIsInstance(single_qubit_gates_kernel, Kernel)

    def test_conversion_two_qubit_gates(self):
        """
        Test that a circuit with every two qubits gates supported by the converter is
        converted to a Kernel without errors.
        """
        two_qubit_gates_to_test = Converter.REQUIRED_Q_OP_TWO_QBIT_GATE
        c1, c2 = Channel(label="q0"), Channel(label="q1")
        two_qubit_gates = Circuit(
            name="two_qubit_gates",
            circuit=[
                Circuit(name=gate, channels=[c1, c2])
                for gate in two_qubit_gates_to_test
            ],
        )
        two_qubit_gates_kernel, _, _ = self.converter.convert_circuit(two_qubit_gates)
        self.assertIsInstance(two_qubit_gates_kernel, Kernel)

    def test_conversion_measurement_to_classical_register(self):
        """
        Test that a circuit with every measurement supported by the converter is
        converted to a Kernel without errors.
        """
        measuremet_gates_to_test = Converter.REQUIRED_Q_OP_MEAS
        classical_register = Channel(label="c0", type=ChannelType.CLASSICAL)
        q_chan = Channel(label="q0")
        measurement_circuit = Circuit(
            name="measurement",
            circuit=[
                Circuit(name=gate, channels=[q_chan, classical_register])
                for gate in measuremet_gates_to_test
            ],
        )
        measurement_kernel, _, _ = self.converter.convert_circuit(measurement_circuit)
        self.assertIsInstance(measurement_kernel, Kernel)

    def test_conversion_classical_controlled_gates(self):
        """
        Test that a circuit with every classically controlled gate supported by the
        converter is converted to a Kernel without errors.
        """
        classical_register = Channel(label="c0", type=ChannelType.CLASSICAL)
        q_chan0 = Channel(label="q0")
        q_chan1 = Channel(label="q1")
        classical_controlled_circuit = Circuit(
            name="classical_controlled_gates",
            circuit=(Circuit(name="measure_x", channels=[classical_register, q_chan1]),)
            + tuple(
                Circuit(
                    name=f"classically_controlled_{gate}",
                    channels=[classical_register, q_chan1, q_chan0],
                )
                for gate in Converter.REQUIRED_Q_OP_TWO_QBIT_GATE
            )
            + tuple(
                Circuit(
                    name=f"classically_controlled_{gate}",
                    channels=[q_chan0, classical_register],
                )
                for gate in Converter.REQUIRED_Q_OP_SINGLE_QBIT_GATE
                | Converter.REQUIRED_Q_OP_MEAS
                | Converter.REQUIRED_Q_OP_RESET
            )
            + (Circuit(name="measure_z", channels=[q_chan0]),),
        )

        classical_controlled_kernel, _, _ = self.converter.convert_circuit(
            classical_controlled_circuit
        )
        self.assertIsInstance(classical_controlled_kernel, Kernel)

    def run_convert_circuit_test(self, test_data: ConverterCircuitTestData):
        """Run the conversion test for a given input circuit, it asserts that the
        conversion is successful and the output matches if raw output or content
        are defined in the test data.

        Parameters
        ----------
        test_data (ConverterCircuitTestData): The test data.
        """

        res, _, _ = self.converter.convert_circuit(test_data.input)
        self.assertIsInstance(res, Kernel)
        if test_data.expected_raw_output:
            self.assert_raw_output(str(res), test_data.expected_raw_output)
        if test_data.expected_content:
            self.assert_kernel_content(res, test_data.expected_content)

    def test_generic_cases(self):
        """Test all the generic case defined"""
        for case in self.input_test_data:
            with self.subTest(input=case.test_name):
                self.run_convert_circuit_test(case)

    # Test that a circuit provide the correct state vector.
    # def test_state_vector(self):
    #     """Test that a circuit provide the correct state vector."""
    #     channels = get_n_qchannel(3)
    #     circ = Circuit(
    #         name="state_vector_test",
    #         circuit=[
    #             [
    #                 Circuit(name="h", channels=channels[0]),
    #                 Circuit(name="h", channels=channels[1]),
    #             ],
    #             [Circuit(name="cx", channels=[channels[1], channels[2]])],
    #             [Circuit(name="cy", channels=[channels[0], channels[1]])],
    #             [Circuit(name="phase", channels=[channels[0]])],
    #         ],
    #         channels=channels,
    #     )
    #     kernel, qr, cr = self.converter.convert_circuit(circ)
    #     state = np.array(cudaq.get_state(kernel))

    #     expected_state = np.array([0.5, 0, 0, -0.5, 0, 0.5, 0.5, 0]).astype(
    #         np.complex128
    #     )
    #     self.assertTrue(np.allclose(state, expected_state, 1e-15))

    def test_return_register(self):
        """
        Test that the converter returns the correct quantum and classical registers.
        """
        q_chan = get_n_qchannel(2)
        c_chan = get_n_qchannel(2, ChannelType.CLASSICAL)
        circ = Circuit(
            name="return_register_test",
            circuit=[
                Circuit(
                    name="reset_0", channels=q_chan[1]
                ),  # Do something on the second qubit so that it is part of the circuit
                Circuit(name="x", channels=q_chan[0]),
                Circuit(name="measure_z", channels=[q_chan[0], c_chan[0]]),
            ],
        )
        kernel, qr, cr = self.converter.convert_circuit(circ)
        # this should x on the second qubit
        kernel.c_if(cr[c_chan[0].id], lambda: kernel.x(qr[q_chan[1].id]))
        cr[c_chan[1].id] = kernel.mz(qr[q_chan[1].id], c_chan[1].label)

        self.assert_kernel_content(
            kernel,
            [
                "reset",
                "x",
                "mz",
                "x",
                "mz",
            ],
        )

    def test_get_outcomes_parity(self):
        """Test the get_outcomes_parity method of the converter."""

        # Mock the cudaq.SampleResult of a circuit that has been run with 3 shots
        num_shots = 3
        mock_sample_result = MagicMock(spec=cudaq.SampleResult)
        mock_sample_result.register_names = ["__global__", "c0_0", "c0_1", "c0_2"]
        mock_data = {
            "__global__": ["1", "1", "1"],
            "c0_0": ["1", "1", "1"],
            "c0_1": ["0", "0", "0"],
            "c0_2": ["1", "1", "1"],
        }

        # Mock get_sequential_data to return mapped values
        def get_sequential_data_side_effect(l):
            return mock_data[l]

        mock_sample_result.get_sequential_data.side_effect = (
            get_sequential_data_side_effect
        )
        cbits = [
            ("c0", 0),
            ("c0", 1),
            ("c0", 2),
        ]  # Classical bit for measurement result
        parity = EkaToCudaqConverter.get_outcomes_parity(cbits[1:], mock_sample_result)
        self.assertEqual(parity, [1 for _ in range(num_shots)])
        parity = EkaToCudaqConverter.get_outcomes_parity(cbits, mock_sample_result)
        self.assertEqual(parity, [0 for _ in range(num_shots)])


if __name__ == "__main__":
    unittest.main()
