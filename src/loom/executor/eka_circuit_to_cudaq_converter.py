"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

import operator
from abc import ABC, abstractmethod
from functools import partial, reduce, cached_property
from typing import Any, Callable, Tuple

# pylint: disable=import-error, wrong-import-position, possibly-used-before-assignment
import importlib.util

if importlib.util.find_spec("cudaq"):
    import cudaq
    from cudaq import Kernel, SampleResult

from ..eka import ChannelType, Circuit
from ..interpreter import InterpretationStep, Cbit


# pylint: disable=redefined-builtin
class Converter(ABC):
    """
    !!! This class is meant to be replaced by some similar construction when refactoring
    Executor !!!

    Abstract base class for converting EKA circuits to a specific format.
    This class defines the required quantum operations and provides a method to
    validate that the converter supports all the required operations.
    Subclasses must implement the abstract methods to provide the actual conversion logic.

    Properties
    ----------
    q_op_single_qbit_gate : dict
        Mapping of single qubit gate operations.
    q_op_two_qbit_gate : dict
        Mapping of two qubit gate operations.
    q_op_reset : dict
        Mapping of reset operations.
    q_op_meas : dict
        Mapping of measurement operations.
    q_op_misc : dict
        Miscellaneous quantum operations that do not fit into the other categories.
    quantum_operations_map : dict
        Unified operations map from all categories, excluding classically controlled operations.
    classically_controlled_operations : dict
        Combined classically controlled operations from both single and two qubit gates.
    operations_map : dict
        A unified operations map from all categories, combining single qubit gates,
        two qubit gates, reset operations, measurement operations, and miscellaneous operations.

    Raises
    ------
    TypeError
        If the mapping is not a dict.
    ValueError
        If the mapping is missing any of the required keys.
    """

    REQUIRED_Q_OP_SINGLE_QBIT_GATE = {"i", "x", "y", "z", "h", "phase", "phaseinv"}

    @property
    @abstractmethod
    def q_op_single_qbit_gate(self) -> dict[str, Callable]:
        """Mapping of single qubit gate operations"""

    REQUIRED_Q_OP_TWO_QBIT_GATE = {"cnot", "cx", "cy", "cz", "swap"}

    @property
    @abstractmethod
    def q_op_two_qbit_gate(self) -> dict[str, Callable]:
        """Mapping of two qubit gate operations"""

    REQUIRED_Q_OP_RESET = {
        "reset",
        "reset_0",
        "reset_1",
        "reset_+",
        "reset_-",
        "reset_+i",
        "reset_-i",
    }

    @property
    @abstractmethod
    def q_op_reset(self) -> dict[str, tuple[Callable, ...]]:
        """Mapping of reset operations"""

    REQUIRED_Q_OP_MEAS = {"measurement", "measure_z", "measure_x", "measure_y"}

    @property
    @abstractmethod
    def q_op_meas(self) -> dict[str, Callable]:
        """Mapping of measurement operations"""

    REQUIRED_Q_OP_MISC = (
        set()
    )  # It is not clear what operations are required here, so I left it empty.

    @property
    @abstractmethod
    def q_op_misc(self) -> dict[str, Callable]:
        """Miscellaneous quantum operations that do not fit into the other categories."""

    @property
    def quantum_operations_map(self) -> dict[str, Callable]:
        """Unified operations map from all categories."""
        return (
            self.q_op_single_qbit_gate
            | self.q_op_two_qbit_gate
            | self.q_op_reset
            | self.q_op_meas
            | self.q_op_misc
        )

    REQUIRED_CLASSICALLY_CONTROLLED_OPERATIONS = set(
        "classically_controlled_" + op
        for op in REQUIRED_Q_OP_TWO_QBIT_GATE
        | REQUIRED_Q_OP_SINGLE_QBIT_GATE
        | REQUIRED_Q_OP_RESET
        | REQUIRED_Q_OP_MISC
    )

    @property
    @abstractmethod
    def classically_controlled_operations(self) -> dict[str, Any]:
        """Mapping of classically controlled two qubit gate operations"""

    @property
    def operations_map(self) -> dict[str, Any]:
        """Unified operations map from all categories, including classically controlled
        operations."""
        return self.quantum_operations_map | self.classically_controlled_operations

    def __init__(self):
        """
        Ensure that the converter support all the required quantum operations
        """
        self._validate_ops()

    def _validate_ops(self):
        """Validate that the converter supports all the required quantum operations."""

        def _check_keys(name: str, mapping: dict, required_keys: set):
            """Check that the mapping has the required keys."""
            if not isinstance(mapping, dict):
                raise TypeError(f"{name} must be a dict")
            # Add prefix to required keys if provided
            # (for classically controlled operations)
            missing = required_keys - mapping.keys()
            if missing:
                raise ValueError(f"{name} is missing required keys: {missing}")

        values_to_check = [
            (
                "q_op_single_qbit_gate",
                self.q_op_single_qbit_gate,
                self.REQUIRED_Q_OP_SINGLE_QBIT_GATE,
            ),
            (
                "q_op_two_qbit_gate",
                self.q_op_two_qbit_gate,
                self.REQUIRED_Q_OP_TWO_QBIT_GATE,
            ),
            ("q_op_reset", self.q_op_reset, self.REQUIRED_Q_OP_RESET),
            ("q_op_meas", self.q_op_meas, self.REQUIRED_Q_OP_MEAS),
            ("q_op_misc", self.q_op_misc, self.REQUIRED_Q_OP_MISC),
            (
                "classically_controlled_operations",
                self.classically_controlled_operations,
                self.REQUIRED_CLASSICALLY_CONTROLLED_OPERATIONS,
            ),
        ]
        for values in values_to_check:
            _check_keys(*values)

    @abstractmethod
    def convert(self, input: InterpretationStep) -> Any:
        """Convert and InterpretationStep."""

    @abstractmethod
    def convert_circuit(self, input: Circuit) -> Any:
        """Convert a Circuit."""

    @staticmethod
    def _validate_ops_args(op_name: str, q_target: list, c_target: list) -> None:
        """Validate the arguments for the operation."""
        if (
            op_name
            in Converter.REQUIRED_Q_OP_SINGLE_QBIT_GATE | Converter.REQUIRED_Q_OP_RESET
        ):
            if len(q_target) != 1:
                raise ValueError(
                    f"Operation {op_name} requires exactly one quantum register, "
                    f"but got {len(q_target)}."
                )
        elif op_name in Converter.REQUIRED_Q_OP_TWO_QBIT_GATE:
            if len(q_target) != 2:
                raise ValueError(
                    f"Operation {op_name} requires exactly two quantum registers, "
                    f"but got {len(q_target)}."
                )
        elif op_name in Converter.REQUIRED_CLASSICALLY_CONTROLLED_OPERATIONS:
            if len(c_target) == 0:
                raise ValueError(
                    f"Classically controlled operation {op_name} requires at least "
                    f"one classical register, but got {len(c_target)}."
                )
        elif op_name in Converter.REQUIRED_Q_OP_MEAS:
            if len(q_target) != 1:
                raise ValueError(
                    f"Measurement operation {op_name} requires exactly one quantum "
                    f"register, but got {len(q_target)}."
                )
            if len(c_target) > 1:
                raise ValueError(
                    f"Measurement operation {op_name} can have at most one classical "
                    f"register, but got {len(c_target)}."
                )


class EkaToCudaqConverter(Converter):
    """Converter for EKA circuits to cudaq kernels."""

    # KernelCallable is a type alias for a callable that takes a Kernel, quantum target,
    # classical target, and an operation, and returns a tuple of operations to be
    # applied to the kernel.
    # The target are given as a list of tuples, where each tuple contains the channel
    # ID, channel label, and the allocated register for that channel.
    KernelCallable = Callable[
        [
            Kernel,
            list[tuple[str, str, cudaq.QuakeValue]],
            list[tuple[str, str, cudaq.QuakeValue]],
        ],
        Any,
    ]

    # pylint: disable=unused-argument
    @cached_property
    def q_op_single_qbit_gate(self) -> dict[str, KernelCallable]:
        def _op(kernel: Kernel, q_target, c_target, op):
            return op(kernel, q_target[0][2])  # target the first quantum register

        # List or dict of gate names and their corresponding ops
        gates = {
            "i": lambda ker, tar: (ker, tar),
            "h": Kernel.h,
            "x": Kernel.x,
            "y": Kernel.y,
            "z": Kernel.z,
            "phase": Kernel.s,
            "phaseinv": lambda kernel, target: (
                Kernel.z(kernel, target),
                Kernel.s(kernel, target),
            ),
        }

        # Build and return the dict with partials
        return {name: partial(_op, op=op) for name, op in gates.items()}

    # pylint: disable=unused-argument
    @cached_property
    def q_op_two_qbit_gate(self) -> dict[str, KernelCallable]:
        def _op(
            kernel: Kernel,
            q_target: list,
            c_target: list,
            op,
        ):
            return op(
                kernel, q_target[0][2], q_target[1][2]
            )  # target the two first quantum registers

        gates = {
            "cnot": Kernel.cx,
            "cy": Kernel.cy,
            "cz": Kernel.cz,
            "cx": Kernel.cx,
            "swap": Kernel.swap,
        }
        return {name: partial(_op, op=op) for name, op in gates.items()}

    # pylint: disable=unused-argument
    @cached_property
    def q_op_reset(self) -> dict[str, KernelCallable]:
        def _op(kernel: Kernel, q_target: list, c_target: list, op):
            return op(kernel, q_target[0][2])  # target the first quantum register

        gates = {
            "reset": Kernel.reset,
            "reset_0": Kernel.reset,
            "reset_1": lambda ker, targ: (ker.reset(targ), ker.x(targ)),
            "reset_+": lambda ker, targ: (ker.reset(targ), ker.h(targ)),
            "reset_-": lambda ker, targ: (ker.reset(targ), ker.x(targ), ker.h(targ)),
            "reset_+i": lambda ker, targ: (ker.reset(targ), ker.h(targ), ker.s(targ)),
            "reset_-i": lambda ker, targ: (
                ker.reset(targ),
                ker.x(targ),
                ker.h(targ),
                ker.s(targ),
            ),
        }
        return {name: partial(_op, op=op) for name, op in gates.items()}

    @cached_property
    def q_op_meas(self) -> dict[str, KernelCallable]:

        def _op(kernel, q_target: list, c_target: list, op):
            if len(c_target) > 0:
                # target the first quantum register and label according to the first
                # classical channel's label
                return op(kernel, q_target[0][2], regName=c_target[0][1])

            return op(kernel, q_target[0][2], regName=q_target[0][1])

        gates = {
            "measurement": Kernel.mz,
            "measure_z": Kernel.mz,
            "measure_x": Kernel.mx,
            "measure_y": Kernel.my,
        }

        return {name: partial(_op, op=op) for name, op in gates.items()}

    @cached_property
    def q_op_misc(self) -> dict[str, KernelCallable]:
        return {}

    @cached_property
    def classically_controlled_operations(self) -> dict[str, KernelCallable]:
        """Mapping of classically controlled two qubit gate operations,
        which includes single qubit gates, two qubit gates, reset operations,
        measurement operations, and miscellaneous operations.
        The first given classical channel is the control channel,
        and the rest are forwarded as the target channels.
        """

        def _op(
            kernel: Kernel,
            q_target: list,
            c_target: list,
            op,
        ):
            """Classically controlled operation."""
            # The first element of c_target is the classical register that
            # controls the operation.
            remaining_c_target = list[c_target[1:]] if len(c_target) > 1 else []
            return Kernel.c_if(
                kernel,
                c_target[0][2],  # control register
                lambda: op(kernel, q_target, remaining_c_target),
            )

        return {
            f"classically_controlled_{name}": partial(
                _op, op=self.quantum_operations_map[name]
            )
            for name in Converter.REQUIRED_Q_OP_SINGLE_QBIT_GATE
            | Converter.REQUIRED_Q_OP_RESET
            | Converter.REQUIRED_Q_OP_MEAS
            | Converter.REQUIRED_Q_OP_TWO_QBIT_GATE
            | Converter.REQUIRED_Q_OP_MISC
        }

    def convert_circuit(
        self,
        input: Circuit,
    ) -> tuple[
        cudaq.kernel, dict[str, cudaq.QuakeValue], dict[str, cudaq.QuakeValue | None]
    ]:
        """Convert a Circuit to a cudaq kernel.
        Parameters
        ----------
        input : Circuit
            The input circuit to convert.
        Returns
        -------
        cudaq.kernel
            The converted cudaq kernel.
        dict[str, cudaq.QuakeValue]
            A dictionary mapping quantum channel IDs to their allocated registers.
        dict[str, cudaq.QuakeValue | None]
            A dictionary mapping classical channel IDs to their allocated registers.
            If a classical channel is not allocated, its value will be None.

        Raises
        ------
        TypeError
            If the input is not a Circuit.
        ValueError
            If the input circuit is empty or does not contain any quantum channels.
        """

        # Create a context kernel for the converter.
        if not isinstance(input, Circuit):
            raise TypeError("Input must be a Circuit")
        if not input.channels:
            # Return an empty kernel if there are no channels.
            return cudaq.make_kernel()
        if not input.circuit and input.name not in self.operations_map:
            return cudaq.make_kernel()
        unroll_cricuit = Circuit.unroll(input)
        # Retrieve quantum channels from the input circuit.
        # Warning, order of channels in the list is random.
        q_channels = [
            chan for chan in input.channels if chan.type == ChannelType.QUANTUM
        ]

        c_channels = [
            chan for chan in input.channels if chan.type == ChannelType.CLASSICAL
        ]

        # Sort the channels by their label. Eka do not enforce any order, this is mostly
        # for convenience, if the user used labels with implicit order like q0, q1, q2,
        # etc. This will ensure the outcome to have some meaningful order.
        q_channels = sorted(q_channels, key=lambda ch: ch.label)
        c_channels = sorted(c_channels, key=lambda ch: ch.label)

        # Create a kernel for the circuit.
        kernel = cudaq.make_kernel()

        # Allocate quantum registers for each quantum channel.
        q_registers = {c.id: kernel.qalloc() for c in q_channels}

        # Allocate classical registers for each classical channel.
        c_registers = {c.id: None for c in c_channels}

        for tick in unroll_cricuit:
            for item in tick:
                quantum_target_reg = [
                    (c.id, c.label, q_registers[c.id])
                    for c in item.channels
                    if c.type != ChannelType.CLASSICAL
                ]

                classical_target_reg = [
                    (c.id, c.label, c_registers[c.id])
                    for c in item.channels
                    if c.type == ChannelType.CLASSICAL
                ]

                if item.name not in self.operations_map:
                    raise KeyError(
                        f"Operation {item.name} not found in operations map."
                    )

                op = self.operations_map[item.name]

                self._validate_ops_args(
                    item.name, quantum_target_reg, classical_target_reg
                )

                # Measurement store the results
                if item.name in self.q_op_meas:
                    if len(classical_target_reg) > 0:
                        # If there is a classical target, we need to store the
                        # measurement outcome.
                        # In a dict, using channel ID as the key.
                        c_registers[classical_target_reg[0][0]] = op(
                            kernel, quantum_target_reg, classical_target_reg
                        )
                    else:
                        op(kernel, quantum_target_reg, classical_target_reg)
                else:
                    op(kernel, quantum_target_reg, classical_target_reg)

        return kernel, q_registers, c_registers

    def convert(
        self, input: InterpretationStep
    ) -> Tuple[
        cudaq.kernel, dict[str, cudaq.QuakeValue], dict[str, cudaq.QuakeValue | None]
    ]:
        """Convert an InterpretationStep to a cudaq kernel.
        For now it just calls `convert_circuit` on the final circuit of the
        InterpretationStep.
        """
        return self.convert_circuit(input.final_circuit)

    @staticmethod
    def get_outcomes_parity(
        cbits: list[Cbit], simulation_output: SampleResult  # type: ignore
    ) -> list[int]:
        """Get the parity of the outcomes of multiple measurements from the simulation
        output.
        The parity is the xor of all outcomes.

        Parameters
        ----------
        cbits : List[Cbit]
            The list of cbits to get the outcomes for.
        simulation_output : cudaq.SampleResult
            The simulation output containing the measurement results.
        shot_idx : int, optional
            The index of the shot to get the outcome for. If None, a list with outcomes
            for all shots is returned.

        Returns
        -------
        list[int]
            The parity of the outcomes for each shot of the simulation output.
        """

        def get_outcomes(cbit: Cbit, simulation_output: SampleResult) -> list[int]:  # type: ignore
            """Get the outcome of a measurement from the simulation output.
            If the measurement is not present in the output, return None."""

            if not isinstance(cbit, tuple):
                if isinstance(cbit, int) and cbit in (0, 1):
                    return cbit
                raise TypeError("cbit must be a tuple[str, int]")

            if len(cbit) != 2:
                raise TypeError("cbit must be a tuple[str, int] or a Literal[0, 1]")

            if not isinstance(cbit[0], str) or not isinstance(cbit[1], int):
                raise TypeError("cbit must be a tuple[str, int]")

            label = f"{cbit[0]}_{cbit[1]}"
            if label not in simulation_output.register_names:
                raise KeyError(f"Measurement {label} not found in simulation output.")

            return simulation_output.get_sequential_data(label)

        outcome_lists = [get_outcomes(cbit, simulation_output) for cbit in cbits]
        outcome_lists = [outcome for outcome in outcome_lists if outcome is not None]

        shotwise_outcomes = zip(*outcome_lists, strict=True)

        return [
            reduce(operator.xor, (int(x) for x in shot if x is not None), 0)
            for shot in shotwise_outcomes
        ]
