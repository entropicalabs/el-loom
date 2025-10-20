"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

from importlib.metadata import entry_points

from loom.eka import Circuit, Eka
from loom.eka.operations import Operation, BaseOperation, CodeOperation

from .applicator import BaseApplicator, CodeApplicator
from .interpretation_step import InterpretationStep


def load_plugin_options():
    """
    Load plugin options from entry points defined in the 'loom.selectors' group.
    If plugins are installed, they are detected by this method. This allows the user to
    use `interpret_eka` with plugin specific applicators.
    """
    plugin_options = {}
    for ep in entry_points().select(group="loom.selectors"):
        data_provider = ep.load()
        plugin_options.update(data_provider)
    return plugin_options


def interpret_operation(
    eka: Eka,
    step: InterpretationStep,
    op: Operation,
    same_timeslice: bool,
    debug_mode: bool,
) -> InterpretationStep:
    """
    This function interprets the given operation and returns a new InterpretationStep
    which contains all modifications due to the operation. The function itself finds the
    right applicator for the given operation and calls it. The actual implementation of
    the operation is done in the applicator.

    Parameters
    ----------
    eka : Eka
        The Eka object for which the operation should be interpreted. The Eka is
        passed on to the applicator. This allows the applicator to access the lattice
        and the syndrome circuits
    step : InterpretationStep
        Current InterpretationStep to which the modifications due to the operation
        should be applied
    op : :class:`loom.eka.operations.base_operation.Operation`
        Operation to be interpreted
    same_timeslice : bool
        Flag indicating whether the operation is part of the same timestep as the
        previous operation.
    debug_mode : bool
        Flag indicating whether the interpretation should be done in debug mode.
        Currently, the effects of debug mode are:
        - Disabling the commutation validation of Block

    Returns
    -------
    InterpretationStep
        New InterpretationStep containing all modifications due to the operation
    """
    match op:
        case BaseOperation():
            # BaseOperations are implemented at the level of the BaseApplicator
            return BaseApplicator(eka).apply(step, op, same_timeslice, debug_mode)
        case CodeOperation():
            # Get the class name of the blocks
            block_class_names = list(
                set(
                    step.get_block(block_label).__class__.__name__
                    for block_label in op._inputs  # pylint: disable=protected-access
                )
            )
            # For CodeOperations, check that the code types of all input blocks are the
            # same and call the appropriate applicator
            if len(block_class_names) > 1:
                raise NotImplementedError(
                    "Operations including blocks of different classes are not "
                    f"supported at the moment. Block classes: {block_class_names}"
                )
            if block_class_names[0] == "Block":
                return CodeApplicator(eka).apply(step, op, same_timeslice, debug_mode)
            plugin_options = load_plugin_options()
            if plugin_options:
                # If there are plugins available, we can try to load the appropriate
                # applicator for the block class name
                for plugin_value in plugin_options.values():
                    if plugin_value["block_class_name"] == block_class_names[0]:
                        return plugin_value["applicator"](eka).apply(
                            step, op, same_timeslice, debug_mode
                        )
            raise NotImplementedError(
                f"The Block type '{block_class_names[0]}' is not supported at the moment"
            )


def cleanup_final_step(final_step: InterpretationStep) -> InterpretationStep:
    """
    Clean up the final interpretation step before it is returned to the user.

    Parameters
    ----------
    final_step : InterpretationStep
        Final interpretation step which should be cleaned up before returning it to the
        user

    Returns
    -------
    InterpretationStep
        Cleaned up interpretation step
    """
    cleaned_final_step = final_step
    # Remove all channels from the channel_dict that are not part of the circuit
    # This can only be done if the circuit is not None (which is true for all realistic
    # use cases). For empty circuits, the dict is set to an empty dict.

    # circuit_seq is a tuple of tuples of composite circuits, we need to align the empty
    # tuples to make sure that the final circuit has the correct duration and that
    # parallel operations are indeed in parallel
    circuit_seq = final_step.intermediate_circuit_sequence
    full_circuit = ()
    for timeslice in circuit_seq:
        timespan = max(composite_circuit.duration for composite_circuit in timeslice)
        # Create a template circuit
        template_circ = (
            tuple(composite_circuit for composite_circuit in timeslice),
        ) + ((),) * (timespan - 1)
        full_circuit += template_circ

    cleaned_final_step.final_circuit = Circuit(
        name="Final circuit",
        circuit=full_circuit,
        channels=sorted(  # Ensure channels are sorted
            list(
                set(
                    channel
                    for timeslice in full_circuit
                    for circuit in timeslice
                    for channel in circuit.channels
                )
            ),
            key=lambda ch: ch.label,
        ),
    )

    cleaned_final_step.channel_dict = (
        {
            ch.id: ch
            for ch in cleaned_final_step.channel_dict.values()
            if ch in cleaned_final_step.final_circuit.channels
        }
        if cleaned_final_step.final_circuit is not None
        else {}
    )
    return cleaned_final_step


def interpret_eka(eka: Eka, debug_mode: bool = True) -> InterpretationStep:
    """
    Interpret the Eka object and return the final InterpretationStep. The function
    iterates over all operations in the Eka and applies them to the current
    InterpretationStep. The function also handles the case where multiple operations
    are applied in parallel during the same timestep.

    Parameters
    ----------
    eka : Eka
        Eka object describing the operations to perform, as well as the initial state
         of the blocks.
    debug_mode : bool
        Flag indicating whether the interpretation should be done in debug mode.
        Activating debug mode will enable commutation validation for Block

    Returns
    -------
    InterpretationStep
        Final InterpretationStep containing the interpreted operations and the final
        circuit.
    """
    # Initialize the interpretation step by passing the initial blocks as the first
    # element of the block history
    step = InterpretationStep(block_history=(eka.blocks,))
    for timestep in eka.operations:
        # Reset the same_timeslice flag for each new timestep
        same_timeslice = False
        for op in timestep:
            step = interpret_operation(eka, step, op, same_timeslice, debug_mode)
            # Set the same_timeslice flag to True after the first operation in the
            # timestep has been interpreted to indicate that the following operations
            # are part of the same timestep
            same_timeslice = True

    # Return a cleaned up version of the final step
    step = cleanup_final_step(step)
    # Freeze the InterpretationStep to prevent further modifications
    step.is_frozen = True
    return step
