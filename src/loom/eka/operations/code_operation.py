"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

from functools import partial

from pydantic.dataclasses import dataclass
from pydantic import Field, field_validator


from .base_operation import Operation
from .logical_measurement import LogicalMeasurement
from ..utilities import (
    SingleQubitPauliEigenstate,
    Direction,
    Orientation,
    ResourceState,
    dataclass_params,
    larger_than_zero_error,
)


# CodeOperation act on the code itself
@dataclass(**dataclass_params)
class CodeOperation(Operation):
    """
    Parent class for all code operations. All code operations act on blocks

    Properties
    ----------

    _inputs : tuple[str, ...]
        Standardized way to access the input blocks names.
    _outputs : tuple[str, ...]
        Standardized way to access the output blocks names.
    """

    @property
    def _inputs(self):
        """
        Standardized way to access the input block(s) names.

        Returns
        -------
        tuple[str, ...]
            Names of the input blocks
        """
        if hasattr(self, "input_block_name"):
            return (self.input_block_name,)
        if hasattr(self, "input_blocks_name"):
            return self.input_blocks_name

        raise ValueError(f"No block inputs specified for {self.__class__.__name__}")

    @property
    def _outputs(self):
        """
        Standardized way to access the output block(s) names.

        Returns
        -------
        tuple[str, ...]
            Names of the output blocks
        """
        if hasattr(self, "output_block_name"):
            return (self.output_block_name,)
        if hasattr(self, "output_blocks_name"):
            return self.output_blocks_name

        return self._inputs


# Readout operations
@dataclass(**dataclass_params)
class MeasureBlockSyndromes(CodeOperation):
    """
    Performs a given number of rounds of syndrome measurements on a block.

    Parameters
    ----------
    input_block_name : str
        Name of the block to measure.
    n_cycles : int
        Number of cycles to measure. Default is 1.
    """

    input_block_name: str
    n_cycles: int = 1


@dataclass(**dataclass_params)
class MeasureLogicalX(CodeOperation):
    """
    Measure the logical X operator of a block.

    Parameters
    ----------
    input_block_name : str
        Name of the block where the logical operator to be measured is located.
    logical_qubit : int | None, optional
        Index of the logical qubit inside the specified block which should be measured.
        For blocks with a single logical qubit, this parameter does not need to be
        provided. Then by default the index 0 is chosen for this single logical qubit.
        For blocks with multiple logical qubits, this parameter is required.
    """

    input_block_name: str
    logical_qubit: int = Field(default=0)


@dataclass(**dataclass_params)
class MeasureLogicalZ(CodeOperation):
    """
    Measure the logical Z operator of a block.

    Parameters
    ----------
    input_block_name : str
        Name of the block where the logical operator to be measured is located.
    logical_qubit : int | None, optional
        Index of the logical qubit inside the specified block which should be measured.
        For blocks with a single logical qubit, this parameter does not need to be
        provided. Then by default the index 0 is chosen for this single logical qubit.
        For blocks with multiple logical qubits, this parameter is required.
    """

    input_block_name: str
    logical_qubit: int = Field(default=0)


@dataclass(**dataclass_params)
class MeasureLogicalY(CodeOperation):
    """
    Measure the logical Y operator of a block.

    Parameters
    ----------
    input_block_name : str
        Name of the block where the logical operator to be measured is located.
    logical_qubit : int | None, optional
        Index of the logical qubit inside the specified block which should be measured.
        For blocks with a single logical qubit, this parameter does not need to be
        provided. Then by default the index 0 is chosen for this single logical qubit.
        For blocks with multiple logical qubits, this parameter is required.
    """

    input_block_name: str
    logical_qubit: int = Field(default=0)


@dataclass(**dataclass_params)
class LogicalX(CodeOperation):
    """
    Apply a logical X operator to a block.

    Parameters
    ----------
    input_block_name : str
        Name of the block where the logical operator should be applied.
    logical_qubit : int | None, optional
        Index of the logical qubit inside the specified block which should be acted on.
        For blocks with a single logical qubit, this parameter does not need to be
        provided. Then by default the index 0 is chosen for this single logical qubit.
        For blocks with multiple logical qubits, this parameter is required.
    """

    input_block_name: str
    logical_qubit: int = Field(default=0)


@dataclass(**dataclass_params)
class LogicalY(CodeOperation):
    """
    Apply a logical Y operator to a block.

    Parameters
    ----------
    input_block_name : str
        Name of the block where the logical operator should be applied.
    logical_qubit : int | None, optional
        Index of the logical qubit inside the specified block which should be acted on.
        For blocks with a single logical qubit, this parameter does not need to be
        provided. Then by default the index 0 is chosen for this single logical qubit.
        For blocks with multiple logical qubits, this parameter is required.
    """

    input_block_name: str
    logical_qubit: int = Field(default=0)


@dataclass(**dataclass_params)
class LogicalZ(CodeOperation):
    """
    Apply a logical Z operator to a block.

    Parameters
    ----------
    input_block_name : str
        Name of the block where the logical operator should be applied.
    logical_qubit : int | None, optional
        Index of the logical qubit inside the specified block which should be acted on.
        For blocks with a single logical qubit, this parameter does not need to be
        provided. Then by default the index 0 is chosen for this single logical qubit.
        For blocks with multiple logical qubits, this parameter is required.
    """

    input_block_name: str
    logical_qubit: int = Field(default=0)


@dataclass(**dataclass_params)
class ResetAllDataQubits(CodeOperation):
    """
    Reset all data qubits to a specific SingleQubitPauliEigenstate.

    input_block_name : str
        Name of the block where the logical operator should be applied.
    state: SingleQubitPauliEigenstate | None, optional
        State to which the logical qubit should be reset. Default is
        SingleQubitPauliEigenstate.ZERO, i.e. the zero eigenstate of the Pauli Z operator.
    """

    input_block_name: str
    state: SingleQubitPauliEigenstate = Field(default=SingleQubitPauliEigenstate.ZERO)


@dataclass(**dataclass_params)
class ResetAllAncillaQubits(CodeOperation):
    """
    Reset all ancilla qubits to a specific SingleQubitPauliEigenstate.

    Parameters
    ----------
    input_block_name : str
        Name of the block where the logical operator should be applied.
    state: SingleQubitPauliEigenstate | None, optional
        State to which the ancilla qubit should be reset. Default is
        SingleQubitPauliEigenstate.ZERO, i.e. the zero eigenstate of the Pauli Z operator.
    """

    input_block_name: str
    state: SingleQubitPauliEigenstate = Field(default=SingleQubitPauliEigenstate.ZERO)


@dataclass(**dataclass_params)
class Grow(CodeOperation):
    """
    Grow operation.

    Parameters
    ----------
    input_block_name : str
        Name of the block to grow.
    direction : Direction
        Direction in which to grow the block.
    length : int
        Length by which to grow the block.
    """

    input_block_name: str
    direction: Direction
    length: int

    _position_error = field_validator("length", mode="before")(
        partial(larger_than_zero_error, arg_name="length")
    )


@dataclass(**dataclass_params)
class Shrink(CodeOperation):
    """
    Shrink operation.

    Parameters
    ----------
    input_block_name : str
        Name of the block to shrink.
    direction : Direction
        Direction in which to shrink the block.
    length : int
        Length by which to shrink the block.
    """

    input_block_name: str
    direction: Direction
    length: int
    _position_error = field_validator("length", mode="before")(
        partial(larger_than_zero_error, arg_name="length")
    )


@dataclass(**dataclass_params)
class Merge(CodeOperation):
    """
    Merge operation.

    Parameters
    ----------
    input_blocks_name : tuple[str, str]
        Names of the two blocks to merge.
    output_block_name : str
        Name of the resulting block.
    orientation : Orientation, optional
        Orientation along which to merge the blocks. E.g. if Orientation.HORIZONTAL,
        the blocks will be merged using their left and right boundaries (whichever is
        easiest). If None, the orientation will be derived from the blocks positions.
    """

    input_blocks_name: tuple[str, str]
    output_block_name: str
    orientation: Orientation | None = Field(default=None, validate_default=True)


@dataclass(**dataclass_params)
class Split(CodeOperation):
    """
    Split operation.

    Parameters
    ----------
    input_block_name : str
        Name of the block to split.
    output_blocks_name : tuple[str, str]
        Names of the resulting blocks.
    orientation : Orientation
        Orientation along which to split the block. E.g. if Orientation.HORIZONTAL, the
        block will be split in a horizontal cut, leaving two blocks with adjacent top
        and bottom boundaries.
    split_position : int
        Position at which to split the block, distance to the (0,0) corner of the block.
    """

    input_block_name: str
    output_blocks_name: tuple[str, str]
    orientation: Orientation
    split_position: int
    _position_error = field_validator("split_position", mode="before")(
        partial(larger_than_zero_error, arg_name="split_position")
    )


@dataclass(**dataclass_params)
class StateInjection(CodeOperation):
    """
    Inject the given resource state into the specified block. This operation resets the
    central qubit of the block into the specified resource state and maximizes the
    number of stabilizers that can be initialized in a deterministic way.

    E.g. a T state injection in a RotatedSurfaceCode block will reset the central qubit
    into the T state and reset the rest of the data qubits into four quadrants,
    such that two quadrants are in the ``|0⟩`` state and two quadrants are in the
    ``|+⟩`` state. This ensures that the Z stabilizer measurements are deterministic in
    the ``|0⟩`` quadrants and the X stabilizer measurements are deterministic in the
    ``|+⟩`` quadrants.

    Parameters
    ----------
    input_block_name : str
        Name of the block to inject the resource state into.
    resource_state : ResourceState
        The resource state to inject into the block. This can be one of the following:
        - ResourceState.T: T state
        - ResourceState.S: S state
    """

    input_block_name: str
    resource_state: ResourceState


@dataclass(**dataclass_params)
class ConditionalLogicalX(CodeOperation):
    """
    Apply a conditional logical X operator to a block.

    Parameters
    ----------
    input_block_name : str
        Name of the block where the logical operator should be applied.
    logical_qubit : int | None, optional
        Index of the logical qubit inside the specified block which should be acted on.
        For blocks with a single logical qubit, this parameter does not need to be
        provided. Then by default the index 0 is chosen for this single logical qubit.
        For blocks with multiple logical qubits, this parameter is required.
    condition : LogicalMeasurement | None, optional
        Condition for logical pauli operation to be applied based on the value of the
        LogicalMeasurement provided.
    """

    input_block_name: str
    condition: LogicalMeasurement
    logical_qubit: int = Field(default=0)


@dataclass(**dataclass_params)
class ConditionalLogicalY(CodeOperation):
    """
    Apply a conditional logical Y operator to a block.

    Parameters
    ----------
    input_block_name : str
        Name of the block where the logical operator should be applied.
    logical_qubit : int | None, optional
        Index of the logical qubit inside the specified block which should be acted on.
        For blocks with a single logical qubit, this parameter does not need to be
        provided. Then by default the index 0 is chosen for this single logical qubit.
        For blocks with multiple logical qubits, this parameter is required.
    condition : LogicalMeasurement
        Condition for logical pauli operation to be applied based on the value of the
        LogicalMeasurement provided.
    """

    input_block_name: str
    condition: LogicalMeasurement
    logical_qubit: int = Field(default=0)


@dataclass(**dataclass_params)
class ConditionalLogicalZ(CodeOperation):
    """
    Apply a conditional logical Z operator to a block.

    Parameters
    ----------
    input_block_name : str
        Name of the block where the logical operator should be applied.
    logical_qubit : int | None, optional
        Index of the logical qubit inside the specified block which should be acted on.
        For blocks with a single logical qubit, this parameter does not need to be
        provided. Then by default the index 0 is chosen for this single logical qubit.
        For blocks with multiple logical qubits, this parameter is required.
    condition : LogicalMeasurement
        Condition for logical pauli operation to be applied based on the value of the
        LogicalMeasurement provided.
    """

    input_block_name: str
    condition: LogicalMeasurement
    logical_qubit: int = Field(default=0)
