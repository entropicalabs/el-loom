"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

from collections import defaultdict
from enum import Enum
import math
from typing import Any, Callable, Optional, Protocol, runtime_checkable
from pydantic import BaseModel, Field, field_validator, model_validator, PrivateAttr


from loom.eka import Circuit


# ====== Type / Aliases =====


# Callable type for error probability, return a list of float given a gate name and an optional
# time.
# Allow returning a list of floats for error probabilities. Some noise instructions
# may require multiple parameters (e.g., pauli_channel).
# pylint: disable=too-few-public-methods
@runtime_checkable
class ErrorProbProtocol(Protocol):
    """Protocol for error probability functions."""

    def __call__(
        self, time_from_start: Optional[float], time_of_tick: Optional[float]
    ) -> list[float]:
        """Compute error probabilities for a given gate and time parameters.

        Parameters
        ----------
        time_from_start : Optional[float]
            Cumulative time from the start of the circuit.
        time_of_tick : Optional[float]
            Duration or idle time within the current tick.

        Returns
        -------
        list[float]
            List of error probabilities.
        """


GateErrorProbProtocol = Callable[[Optional[float]], list[float]]


class ApplicationMode(str, Enum):
    """
    Enum-like class to define the application mode of the error model.
    """

    BEFORE_GATE = "before_gate"
    AFTER_GATE = "after_gate"
    END_OF_TICK = "end_of_tick"
    IDLE_END_OF_TICK = "idle_end_of_tick"


# Error model shall define the types of errors that can occur in a quantum circuit,
# In order to properly map to noise instructions in the different backends.
class ErrorType(Enum):
    """
    Provides a set of error types that can be used to define the error model for a quantum
    circuit.
    Each error type has a label and a number of parameters that it expects.
    Also provides a method to validate the parameters for the error type by checking they
    are a proper probability distribution (sum doesn't exceed 1 if multiple parameters)
    and that they are in the range [0, 1].
    """

    PAULI_X = ("pauli_x", 1)
    PAULI_Y = ("pauli_y", 1)
    PAULI_Z = ("pauli_z", 1)
    PAULI_CHANNEL = ("pauli_channel", 3)
    BIT_FLIP = ("bit-flip", 1)
    PHASE_FLIP = ("phase-flip", 1)
    DEPOLARIZING1 = ("depolarizing1", 1)
    DEPOLARIZING2 = ("depolarizing2", 1)

    def __init__(self, label: str, param_count: int):
        self.label = label
        self.param_count = param_count

    def validate_params(self, params: list[float]) -> None:
        """Raise ValueError if params are not valid for this error type."""
        if len(params) != self.param_count:
            raise ValueError(
                f"{self.name} expects {self.param_count} parameter(s), got {len(params)}."
            )

        if not all(isinstance(p, (float, int)) for p in params):
            raise TypeError(f"All parameters for {self.name} must be numbers.")

        if not all(0 <= p <= 1 for p in params):
            raise ValueError(f"All parameters for {self.name} must be in [0, 1].")

        if self.param_count > 1:
            if sum(params) > 1.0:
                raise ValueError(f"{self.name} parameters sum must not exceed 1.0.")


# ====== CircuitErrorModel Class =====
class CircuitErrorModel(BaseModel):
    """
    Define a circuit error model that can be used to simulate errors in quantum circuits.
    This model can be time-dependent or not, having error applied in different modes,
    and can define error probabilities for each gate in the circuit or for each tick in
    the circuit.
    It is designed to be used with the Circuit class and its operations.

    This class is very general and allows for very flexible error modeling, however, it
    is a bit tedious to work with, so we recommend to define subclasses that
    define specific error models for your use case.

    Parameters
    ----------
    circuit : Circuit
        The quantum circuit to which the error model will be applied.
        This is frozen after initialization, so it cannot be changed.
    error_type : ErrorType
        The type of error that the model will apply to the circuit.
        This is frozen after initialization, so it cannot be changed.
    is_time_dependent : bool
        Whether the error model is time-dependent or not.
        If True, the model will use gate_durations to compute error probabilities.
        If False, the model will not use gate_durations.
        This is frozen after initialization, so it cannot be changed.
    application_mode : ApplicationMode
        The mode in which the error is applied to the circuit.
        It can be BEFORE_GATE, AFTER_GATE, or END_OF_TICK.
        This is frozen after initialization, so it cannot be changed.
    gate_durations : Optional[dict[str, float]]
        A dictionary mapping gate names to their execution times.
        This is only used if the model is time-dependent.
        If the model is not time-dependent, this can be None.
        If provided, it  must assign a duration to each gate type present in the circuit.
    gate_error_probabilities : Optional[dict[str, GateErrorProbProtocol]]
        A dictionary mapping gate names to a callable that returns the error probability
        for that gate.
        If the model is time-dependent, the callable can take an optional time parameter.
        If the model is not time-dependent, the callable should not take any parameters.
        If one gate isn't in the dictionary, it will default to a callable that
        returns 0.
    global_time_error_probability : Callable[[Optional[float]], list[float]]
        A callable that returns the error probability at a specific time in the circuit.
        It can take an optional time parameter, which represents some information in the
        current tick:
        - If the application mode is END_OF_TICK, it represents the duration of the tick.
        - If the application mode is IDLE_END_OF_TICK, it represents the idle times
        of the channel in the tick.

        The function must be well-defined at t = 0 (for both inputs).
    """

    # Pydantic configuration, this is necessary to allow arbitrary types in the model.
    # In particular, we need to allow Circuit and ErrorProbProtocol custom types.
    model_config = {"arbitrary_types_allowed": True, "frozen": True}

    # ====== Required Class Attributes =====

    # We assume a single instance of error model has a single error type and a single
    # application mode.
    # No initial value given, Pydantic will enforce the constructor to provide them.
    circuit: Circuit
    error_type: ErrorType  # Define the instruction type of the error model.
    is_time_dependent: bool
    application_mode: ApplicationMode

    # ====== Attribute ======

    # Dictionary of gate durations, mapping gate names to their execution times.
    # It must assign a duration to each gate type used in the circuit.
    # For time-independent models, this can be left undefined (None).
    gate_durations: Optional[dict[str, float]] = defaultdict(lambda: lambda _: [0.0])

    def update_gate_durations(self, gate_durations: dict[str, float]):
        """Update the gate durations for the error model.
        This will recompute the operation times and tick durations if the model is
        time-dependent."""
        new_gate_durations = self.validate_gate_duration(gate_durations)
        return self.model_copy(
            update={
                "gate_durations": new_gate_durations,
            }
        )

    # Dictionary of gate error probabilities, mapping gate names to a callable
    # that returns the error probabilities for that gate, with optional time attribute.
    gate_error_probabilities: Optional[dict[str, GateErrorProbProtocol]] = defaultdict(
        lambda: lambda _: [0.0]
    )

    # If the application mode is END_OF_TICK or IDLE_END_OF_TICK,
    # this callable defines the error probability
    # for a given time in the circuit.
    # It returns a list of floats according to the expected parameters of the
    # noise instruction.
    global_time_error_probability: ErrorProbProtocol = lambda _, __: [0.0]

    # If the model is time-dependent, this will be computed after initialization.
    # List the ticks duration, ordered according to occurance in the circuit.
    # !! Note: This contains the duration, not the time at which the tick occurs.
    # In order to compute the time at which the tick occurs,
    # you must sum the tick durations up to that point.
    _tick_durations: Optional[list[float]] = PrivateAttr(default=None)

    # If the model is time-dependent, this will be computed after initialization.
    # Dictionary mapping gate IDs to their execution times.
    _op_time: Optional[dict[str, float]] = PrivateAttr(default=None)

    # If the model depends on the idle time of channels during ticks,
    # this will be computed after initialization.
    # Dictionary mapping channel IDs to a list of idle times during each tick.
    _idle_times: Optional[dict[str, list[float]]] = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        """
        Post-initialization method to compute operation times and ticks duration
        if the model is time-dependent.

        This is called after the model is initialized and all validators have run.
        If the model is time-dependent, it computes the operation times and tick durations
        This bypass the frozen nature of the model to set the private attributes.
        """

        if self.is_time_dependent and self.gate_durations is not None:
            op_time, tick_durations, idle_durations = (
                self._compute__op_times_and__tick_durations()
            )
            object.__setattr__(self, "_idle_times", idle_durations)
            object.__setattr__(self, "_op_time", op_time)
            object.__setattr__(self, "_tick_durations", tick_durations)
        else:
            object.__setattr__(self, "_op_time", {})
            object.__setattr__(self, "_tick_durations", [])
            object.__setattr__(self, "_idle_times", {})

    # ====== Validators =====

    # if time-dependent, gate_durations must be provided
    @model_validator(mode="after")
    @classmethod
    def check_duration_defined_if_time_dependent(cls, model):
        """Ensure that gate_durations is defined if the model is time-dependent."""

        if not isinstance(model, CircuitErrorModel):
            raise TypeError("model must be an instance of CircuitErrorModel")
        if model.is_time_dependent and model.gate_durations is None:
            raise ValueError(
                "gate_durations must be provided for time-dependent error models."
            )
        return model

    @model_validator(mode="after")
    @classmethod
    def check_idle_application_is_time_dependent(cls, model):
        """Ensure that idle application mode is only used for time-dependent models."""
        if (
            model.application_mode == ApplicationMode.IDLE_END_OF_TICK
            and not model.is_time_dependent
        ):
            raise ValueError(
                "Idle application mode can only be used with time-dependent error models."
            )
        return model

    # if time-dependent, _op_time and tick_time must be computed
    # pylint: disable=protected-access
    @model_validator(mode="after")
    @classmethod
    def check__op_time_defined_if_time_dependent(cls, model):
        """Ensure that _op_time is defined if the model is time-dependent."""
        if model.is_time_dependent and model._op_time is None:
            raise ValueError(
                "_op_time must be computed for time-dependent error models."
            )
        if model.is_time_dependent and model._tick_durations is None:
            raise ValueError(
                "_tick_durations must be computed for time-dependent error models."
            )
        return model

    @model_validator(mode="after")
    @classmethod
    def validate_gate_duration(cls, model):
        """Ensure the gates duration are defined for all gate used in the circuit
        and that each gate has a valid duration.
        """
        if model.is_time_dependent:
            unrolled = Circuit.unroll(model.circuit)
            gate_set = {op.name for layer in unrolled for op in layer if op is not None}
            v = model.gate_durations

            if v is not None:
                if not isinstance(v, dict):
                    raise TypeError("gate_duration must be a dictionary or None")
                missing_gates = [gate for gate in gate_set if gate not in v]
                if missing_gates:
                    raise ValueError(
                        f"Missing gate durations for: {missing_gates}. Must provide durations",
                        f"for all gates in {gate_set}",
                    )
        return model

    @field_validator("gate_error_probabilities", mode="before")
    @classmethod
    def validate_gate_error_probabilities(
        cls, v: dict
    ) -> dict[str, GateErrorProbProtocol]:
        """Validate the gate error probabilities dictionary.
        Ensure it contains only gates in the gate_set
        Also ensure that each value is a callable.

        Parameters
        ----------
        v : dict[str, GateErrorProbProtocol] | None
            The dictionary of gate error probabilities, where keys are gate names
            and values are callables that return a list of floats representing the
            error probabilities for that gate.
            If None, it defaults to a callable that returns 0.0 for all gates.
        Returns
        -------
        dict[str, GateErrorProbProtocol]
            A validated dictionary of gate error probabilities, where each key is a gate name
            and each value is a callable that returns the
            error probabilities for that gate.
        Raises
        ------
        TypeError: If v is not a dictionary or callable.
        ValueError: If v contains invalid gate names or if a value is not callable.
        """

        # if undefined, return a default callable that returns 0.0
        # This can be allowed if end_of_tick application is used.
        if v is None:
            v = defaultdict(lambda: lambda _: [0.0])

        if not isinstance(v, dict):
            raise TypeError("gate_error_probabilities must be a dictionary")
        # Check it only contains valid gate names

        for key, func in v.items():
            # Check that it's callable
            if not callable(func):
                raise TypeError(f"Value for '{key}' must be callable.")
        invalid_values = [k for k, v in v.items() if not callable(v)]
        if invalid_values:
            raise TypeError(
                "All values in gate_error_probabilities must be callable. "
                f"Keys with invalid value: {invalid_values}"
            )

        # fill with 0 for missing gates
        return defaultdict(lambda: lambda _: [0.0], v)

    # Check that the global_time_error_probability is valid for t=0
    @field_validator("global_time_error_probability", mode="before")
    @classmethod
    def validate_global_time_error_probability(cls, v) -> ErrorProbProtocol:
        """Validate the tick error probabilities callable.
        Ensure it returns a float or a list of floats for t=0."""
        if v is None:
            return lambda _, __: [
                0.0
            ]  # Default to single zero, will be reformatted by model validator if needed
        if not callable(v):
            raise TypeError("global_time_error_probability must be a callable")
        try:

            def is_valid_prob(p):
                return isinstance(p, float) and 0.0 <= p <= 1.0

            result = v(0.0, 0.0)
            if isinstance(result, list):
                if not all(is_valid_prob(p) for p in result):
                    raise ValueError(
                        "global_time_error_probability must return a list of"
                        "floats between 0 and 1"
                    )
            else:
                raise TypeError(
                    "global_time_error_probability must return a list of floats"
                )
        except Exception as e:
            raise ValueError(
                f"global_time_error_probability callable failed: {e}"
            ) from e

        return v

    # ====== Validators that run last (to reformat default values) =====

    @model_validator(mode="after")
    @classmethod
    def validate_gate_error_probabilities_output(cls, model):
        """
        Validate that each lambda in gate_error_probabilities returns the correct
        number of parameters. The previous validators will typically assign default
        value of [0.0] to the gate error probabilities for each undefined gate, but
        some error types may require multiple parameters (e.g., PAULI_CHANNEL).
        This validator will reformat the default value to return a list of zeros with
        the correct length.
        """
        # Create a new dictionary to hold potentially modified functions
        updated_gate_error_probabilities = {}
        expected_count = model.error_type.param_count
        for gate_name, error_func in model.gate_error_probabilities.items():

            try:
                # Test the function with a dummy time value
                if model.is_time_dependent:
                    result = error_func(0.0)
                else:
                    result = error_func(None)

                # Check if this is a default value [0.0] that needs reformatting
                if (
                    len(result) == 1
                    and result[0] == 0.0
                    and model.error_type.param_count > 1
                ):
                    # Create a new function that returns the correct number of zeros

                    if model.is_time_dependent:
                        updated_gate_error_probabilities[gate_name] = (
                            lambda t: [0.0] * expected_count
                        )
                    else:
                        updated_gate_error_probabilities[gate_name] = (
                            lambda _: [0.0] * expected_count
                        )
                else:
                    # Validate using the error type's validation method
                    model.error_type.validate_params(result)
                    updated_gate_error_probabilities[gate_name] = error_func

            except Exception as e:
                raise ValueError(
                    f"Gate '{gate_name}' error probability function failed validation: {e}"
                ) from e

        # Update the model with the potentially modified functions
        if updated_gate_error_probabilities != model.gate_error_probabilities:
            object.__setattr__(
                model,
                "gate_error_probabilities",
                defaultdict(
                    lambda: lambda _: [0.0] * model.error_type.param_count,
                    updated_gate_error_probabilities,
                ),
            )

        return model

    @model_validator(mode="after")
    @classmethod
    def validate_global_time_error_probability_output(cls, model):
        """Validate that global_time_error_probability returns the correct number of
        parameters. The previous validators will typically assign default value of [0.0]
        to the global time error probability, but some error types may require multiple
        parameters (e.g., PAULI_CHANNEL). This validator will reformat the default value
        to return a list of zeros with the correct length."""

        # Define a dummy zero value for tick_time and global_time in order to test the
        # function's behavior.
        tick_time = (
            0.0
            if model.application_mode
            in {ApplicationMode.END_OF_TICK, ApplicationMode.IDLE_END_OF_TICK}
            else None
        )
        global_time = 0.0 if model.is_time_dependent else None
        try:
            result = model.global_time_error_probability(global_time, tick_time)

            # Check if this is a default value [0.0] that needs reformatting
            if (
                len(result) == 1
                and result[0] == 0.0
                and model.error_type.param_count > 1
            ):
                # Create a new function that returns the correct number of zeros
                expected_count = model.error_type.param_count
                # pylint: disable=unnecessary-lambda-assignment
                if model.is_time_dependent:
                    new_func = lambda t, _: [0.0] * expected_count
                else:
                    new_func = lambda _, __: [0.0] * expected_count
                object.__setattr__(model, "global_time_error_probability", new_func)
            else:
                # Validate using the error type's validation method
                model.error_type.validate_params(result)

        except Exception as e:
            raise ValueError(
                f"global_time_error_probability function failed validation: {e}"
            ) from e

        return model

    # ====== Methods =====
    def get_idle_tick_error_probability(
        self, tick_index: int, channel_id: str
    ) -> list[float] | None:
        """
        Get the error probability based on the time a specific channel was idle during a tick.

        Parameters
        ----------
        tick_index : int
            The index of the tick for which to get the error probability.
        channel_id : str
            The ID of the channel for which to get the idle time error probability.

        Returns
        -------
        list[float] | None
            List of floats representing the error probabilities for the idle tick.
            If the application mode is not IDLE_END_OF_TICK, returns None.
        """
        if self.application_mode != ApplicationMode.IDLE_END_OF_TICK:
            return None

        if channel_id not in self._idle_times:
            raise ValueError(f"Channel {channel_id} not found in idle times mapping.")

        # time dependent, check the tick index
        if tick_index < 0 or tick_index >= len(self._tick_durations):
            raise IndexError("Tick index out of range.")
        # compute the time at which the tick occurs (time at the end of tick)

        idle_in_tick = self._idle_times[channel_id][tick_index]

        time = sum(self._tick_durations[: tick_index + 1])
        p = self.global_time_error_probability(time, idle_in_tick)
        # if the error probability is 0, we return None so that it gets ignored instead
        if not p or all(x == 0.0 for x in p):
            return None

        return p

    def get_tick_error_probability(self, tick_index: int = None) -> list[float] | None:
        """
        Get the error probability for a specific tick in the circuit.

        Parameters
        ----------
        int
            The index of the tick for which to get the error probability.

        Returns
        -------
        list[float] | None
            List of floats representing the error probabilities for the tick.
            If the application mode is not END_OF_TICK, returns None.
        """
        if self.application_mode != ApplicationMode.END_OF_TICK:
            return None  # Only applicable for END_OF_TICK application mode

        if not self.is_time_dependent:
            p = self.global_time_error_probability(None, None)
        else:
            if tick_index is None:
                raise ValueError(
                    "tick_index must be provided for time-dependent models."
                )
            # time dependent, check the tick index
            if tick_index < 0 or tick_index >= len(self._tick_durations):
                raise IndexError("Tick index out of range.")
            # compute the time at which the tick occurs (time at the end of tick)
            time = sum(self._tick_durations[: tick_index + 1])
            p = self.global_time_error_probability(
                time, self._tick_durations[tick_index]
            )

        # if the error probability is 0, we return None so that it gets ignored instead
        if not p or all(x == 0.0 for x in p):
            return None

        return p

    def get_gate_error_probability(self, gate: Circuit) -> list[float] | None:
        """
        Get the error type (instruction) and probability for a given gate.

        Parameters
        ----------
        Circuit
            The quantum gate for which to get the error type and probability.

        Returns
        -------
        list[float] | None
            List of floats representing the error probabilities parameters given
            the gate name
            if the error probability is 0, return None so that it gets ignored
            if the application mode is END_OF_TICK, return None
        """
        if not isinstance(gate, Circuit):
            raise TypeError("gate must be an instance of Circuit")
        if gate.circuit != ():
            raise ValueError("gate must have empty children (no sub-circuit)")

        if self.application_mode == ApplicationMode.END_OF_TICK:
            # If the application mode is END_OF_TICK, return None
            return None

        if self.is_time_dependent:
            if gate.id not in self._op_time:
                raise ValueError(f"Gate {gate.id} not found in operation time mapping.")
            time = self._op_time.get(gate.id)
            if self.application_mode == ApplicationMode.AFTER_GATE:
                # If the error is applied after the gate, we use the time of the gate.
                time = self.gate_durations[gate.name] + time
            error_probability = self.gate_error_probabilities[gate.name](time)
        else:
            error_probability = self.gate_error_probabilities[gate.name](None)
        # if the error probability is 0, we return None so that it gets ignored instead
        # of  being converted to a noise instruction with 0 probability.
        if not error_probability or all(x == 0.0 for x in error_probability):
            return None

        return error_probability

    @property
    def total_time(self) -> float:
        """
        Total time of the circuit, computed as the sum of all tick durations.

        Returns
        -------
        float
            The total time of the circuit.

        Raises
        ------
        ValueError
            If the circuit is not time-dependent or tick durations are not set.
        """
        if not self.is_time_dependent or self._tick_durations is None:
            raise ValueError(
                "Circuit is not time-dependent or tick durations are not set."
            )

        return sum(self._tick_durations)

    # ====== Utilities ======

    def _compute__op_times_and__tick_durations(
        self,
    ) -> tuple[dict[str, float], list[float], dict[str, list[float]]]:
        """
        Retrieve for each operation, the time elapsed from the start of the circuit until
        the start of the operation. Also computes the duration of each tick in the circuit
        and the idle time for each channel during each tick.

        Returns
        -------
        tuple[dict[str, float], list[float], dict[str, list[float]]]
            A tuple containing:
            - dict[str, float]: A dictionary mapping gate IDs to the execution times
              (time elapsed before the gate's execution starts).
            - list[float]: A list of tick durations for each layer in the circuit.
            - dict[str, list[float]]: A dictionary mapping channel IDs to a list of
              idle times during each tick.
        """
        _op_time = {}
        _tick_durations = []
        _channel_idle_duration_in_tick = defaultdict(lambda: [])
        time_stack = 0
        unrolled = Circuit.unroll(self.circuit)
        for tick in unrolled:
            channel_usage_time = defaultdict(lambda: 0)
            for operation in tick:
                if operation.name not in self.gate_durations:
                    raise ValueError(
                        f"Gate {operation.name} not found in gate_duration dictionary."
                    )
                _op_time[operation.id] = time_stack
                for ch in operation.channels:
                    channel_usage_time[ch.id] += self.gate_durations[operation.name]
            _tick_duration = max(
                self.gate_durations[op.name] for op in tick if op is not None
            )
            _tick_durations.append(_tick_duration)
            # Store the idle time for each channel in the tick by subtracting the
            # total usage time from the tick duration.
            for ch in self.circuit.channels:
                if ch.is_quantum():
                    _channel_idle_duration_in_tick[ch.id].append(
                        _tick_duration - channel_usage_time[ch.id]
                    )
            time_stack += _tick_duration

        return _op_time, _tick_durations, _channel_idle_duration_in_tick


class HomogeneousTimeIndependentCEM(CircuitErrorModel):
    """
    A constant probability error model that applies a fixed error probability to all gates
    in the circuit.
    This model is not time-dependent, meaning the error probability does not change
    over time.

    Enforces the following properties:
        - The error model is not time-dependent.

    Parameters
    ----------
    error_type : ErrorType
        The type of error that the model will apply to the circuit.
    application_mode : ApplicationMode
        The mode in which the error is applied to the circuit.
    error_probability : list[float]
        The error probability parameter(s) for the error model. This will be assigned
        to all target gates.
    target_gates : list[str]
        A list of gate names to which the error probability applies. Other gates will have
        an error probability of 0.0 by default.
    """

    is_time_dependent: bool = Field(
        init=False, default=False
    )  # This model is not time-dependent.

    # User need to define the error type and application mode
    error_type: ErrorType
    application_mode: ApplicationMode
    error_probability: float | list[float]
    global_time_error_probability: ErrorProbProtocol = Field(
        default=lambda _, __: [0.0], init=False
    )

    gate_error_probabilities: dict[str, GateErrorProbProtocol] | None = Field(
        default_factory=lambda: defaultdict(lambda: lambda _: [0]), init=False
    )

    @field_validator("error_probability")
    @classmethod
    def validate_error_probability(cls, v: float | list[float]) -> list[float]:
        """Validate the error probability to ensure it is a list of floats."""
        if isinstance(v, float):
            return [v]
        if isinstance(v, list) and all(isinstance(x, float) for x in v):
            return v
        raise ValueError("error_probability must be a float or a list of floats.")

    # Define the target gates to which the error probability applies. For the rest of the
    # gates, the error probability will be 0.0. By default, it applies to nothing
    target_gates: list[str] = []

    def model_post_init(self, __context):
        """Post-initialization for HomogeneousTimeIndependentCEM to set constant gate errors."""
        object.__setattr__(
            self,
            "gate_error_probabilities",
            self.validate_gate_error_probabilities(
                {g: lambda _: self.error_probability for g in self.target_gates}
            ),
        )
        # set the global time error probability to a constant value.
        object.__setattr__(
            self,
            "global_time_error_probability",
            lambda _, __: self.error_probability,  # error_probability is already a list
        )

        super().model_post_init(__context)


class HomogeneousTimeDependentCEM(CircuitErrorModel):
    """
    A constant probability error model that applies a fixed error probability function
    to all gates in the circuit.

    Enforces the following properties:
        - The error model is time-dependent.

    Parameters
    ----------
    error_type : ErrorType
        The type of error that the model will apply to the circuit.
    application_mode : ApplicationMode
        The mode in which the error is applied to the circuit.
    error_probability : ErrorProbProtocol
        The error probability parameter(s) for the error model. This will be assigned
        to all target gates.
    target_gates : list[str]
        A list of gate names to which the error probability applies. Other gates will have
        an error probability of 0.0 by default.
    """

    is_time_dependent: bool = Field(
        init=False, default=True
    )  # This model is time-dependent.
    # Define the error type and application mode
    error_type: ErrorType
    application_mode: ApplicationMode

    target_gates: list[str] = []

    error_probability: ErrorProbProtocol

    def model_post_init(self, __context):
        """
        Post-initialization for HomogeneousTimeDependentCEM to set constant gate errors.
        """

        object.__setattr__(
            self,
            "gate_error_probabilities",
            self.validate_gate_error_probabilities(
                {gate: self.error_probability for gate in self.target_gates}
            ),
        )
        # Set the global time error probability to a constant value according to the
        # given parameters.
        object.__setattr__(
            self,
            "global_time_error_probability",
            self.error_probability,
        )

        super().model_post_init(__context)


class AsymmetricDepolarizeCEM(CircuitErrorModel):
    """
    Error model that applies asymmetric depolarizing noise to the circuit.

    Enforce the following properties:
    - The error model is time-dependent.
    - The error type is PAULI_CHANNEL.
    - The application mode is END_OF_TICK.

    Parameters
    ----------
    t1 : float
        The time constant for the X and Y errors.
        Must be positive.
    t2 : float
        The time constant for the Z error.
        Must be positive.
    """

    t1: float
    t2: float
    gate_error_probabilities: dict[str, GateErrorProbProtocol] | None = Field(
        default_factory=lambda: defaultdict(lambda: lambda _: [0]), init=False
    )
    global_time_error_probability: ErrorProbProtocol = Field(
        default=lambda _, __: [0], init=False
    )

    @model_validator(mode="after")
    @classmethod
    def validate_time_constants(cls, model) -> float:
        """t2 must be smaller or equal 2*t1."""
        if model.t2 > 2 * model.t1:
            raise ValueError("t2 must be smaller or equal than 2*t1.")
        return model

    @field_validator("t1", "t2")
    @classmethod
    def check_positive(cls, v, info):
        """Ensure that t1 and t2 are positive."""
        if v < 0:
            raise ValueError(f"{info.field_name} must be non-negative.")
        return v

    def _p(self, t: float) -> list[float]:
        """
        Internal method to compute error probabilities given time t.
        Using the following model:
        p_x = p_y = (1 - exp(-t / t1)) / 4
        p_z = (1 - exp(-t / t2)) / 2 - p_x

        Parameters
        ----------
        t : float
            The time used to compute the error probabilities.

        Returns
        -------
            List[float]: [p_x, p_y, p_z]
        """
        if self.t1 == 0:
            exp_t_t1 = 0
        else:
            exp_t_t1 = math.exp(-t / self.t1)
        p_x = p_y = (1 - exp_t_t1) / 4
        if self.t2 == 0:
            exp_t_t2 = 0
        else:
            exp_t_t2 = math.exp(-t / self.t2)
        p_z = (1 - exp_t_t2) / 2 - p_x
        return [p_x, p_y, p_z]

    is_time_dependent: bool = Field(
        init=False, default=True
    )  # This model is time-dependent.
    error_type: ErrorType = Field(init=False, default=ErrorType.PAULI_CHANNEL)
    application_mode: ApplicationMode = Field(
        init=False, default=ApplicationMode.END_OF_TICK
    )

    def model_post_init(self, __context):
        """
        Post-initialization for AsymmetricDepolarizeCEM to set gate error probabilities.
        """
        object.__setattr__(
            self,
            "global_time_error_probability",
            lambda _, t: self._p(t),
        )
        super().model_post_init(__context)
