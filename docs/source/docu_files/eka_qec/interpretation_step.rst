.. _interpretation_step:

InterpretationStep description
==============================

:class:`~loom.interpreter.interpretation_step.InterpretationStep` is a **mutable** data structure that is used to construct the QEC circuit and by-products (like syndromes and detectors) that are required to run a QEC experiment.
As its name suggest it represents a step in the interpretation process.

When interpreting an :class:`~loom.eka.eka.Eka` object, we need to translate the :class:`~loom.eka.operations.base_operation.Operation` that describe actions on the quantum circuit from a higher level of abstraction (either logical or code level) down to the circuit level. In addition to the circuit, we also need to generate components that are necessary for the QEC routines.
These components can either be :class:`~loom.interpreter.syndrome.Syndrome` or :class:`~loom.interpreter.detector.Detector` and they are used to locate and reference measurements in the circuit that are inputs to the decoding process.
:class:`~loom.interpreter.interpretation_step.InterpretationStep` serves exactly this purpose, it accumulates gates and sub-circuits with each operations, creates the relevant decoding objects and keeps track of the evolution of the code in time. 
At the output of the interpretation, we get an :class:`~loom.interpreter.interpretation_step.InterpretationStep` instance (often called :code:`interpreted_eka` or :code:`final_step`) that encompasses the whole QEC circuit and can then be sent for execution or simulation.

Since :class:`~loom.interpreter.interpretation_step.InterpretationStep` accumulates information, most fields of the object are mutable and connect different processes together.

Output components: What the world needs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently, the interpretation process returns the final instance of :class:`~loom.interpreter.interpretation_step.InterpretationStep` that bears all changes that happened during interpretation.
This is very convenient for testing and debugging but it is not the minimum information: only a few fields are actually necessary to run a QEC routine on a QPU and/or simulator.
This is what we will describe here as output components.

- :attr:`~loom.interpreter.interpretation_step.InterpretationStep.final_circuit` : :code:`Circuit`

This is the circuit that we want to run to perform the task described in :code:`Eka`.
At the end of interpretation, this circuit may still be in a recursive format (for ease of reading and conciseness) but it can be both expanded into base gates and/or converted into different executable formats (stim, qasm, pennylane, etc.).

- :attr:`~loom.interpreter.interpretation_step.InterpretationStep.syndromes` : :code:`tuple[Syndrome, ...]`

The syndromes are collection of measurements that describe stabilizer measurements.
It is assumed that the value of the stabilizer measurement is given by the sum modulo 2 of all physical measurements (stored as :class:`~loom.interpreter.Cbit`s).
These are the values we input to the decoding algorithms (as well as :class:`~loom.interpreter.detector.Detector` ).

- :attr:`~loom.interpreter.interpretation_step.InterpretationStep.detectors` : :code:`tuple[Detector, ...]`

The detectors are collections of syndromes.
It is a different paradigm for detecting errors by comparing state of syndromes rather than measurements directly.
Detectors are typically created such that if the value of a syndrome changes in time, the subsequent syndromes are compared to this new value.
They are typically used in matching algorithms to detect errors in space and time.
These are the values we input to the decoding algorithms (as well as :class:`~loom.interpreter.syndrome.Syndrome`).

- :attr:`~loom.interpreter.interpretation_step.InterpretationStep.logical_observables` : :code:`tuple[LogicalObservable, ...]`

The logical observables are collections of physical measurements that add up to the value of the measurement of a logical qubit.
The value of a logical observable is given by the sum modulo all the physical measurements it is composed of.

Building components: What the world does not want to see
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:class:`~loom.interpreter.interpretation_step.InterpretationStep` also serves as an accumulator object that is populated during interpretation of the different operations.
This offers some convenience to access objects for debugging and plotting utilities.

- :attr:`~loom.interpreter.interpretation_step.InterpretationStep.intermediate_circuit_sequence` : :code:`tuple[tuple[Circuit, ...], ...]`

This is a time-ordered sequence of gate that is populated throughout interpretation.
The circuit representing an operation is added to it directly, either in an existing time step or in the subsequent one depending on the time-ordering of operations themselves.
This object is also used for temporary storage of sub-components of composite operations. Composite operations will first populate this object with sub-operation circuits, pop the circuit sequence off to wrap it correctly and then add it back as a composite circuit.
The final circuit is created from the final value of this sequence.

- :attr:`~loom.interpreter.interpretation_step.InterpretationStep.stabilizer_evolution` : :code:`dict[str, tuple[str, ...]]`

This dictionary keeps track of the evolution of stabilizers during lattice surgery operations.
It maps the uuid of a new stabilizer (key) to the previous stabilizer(s) value(s).
A stabilizer being divided into multiple will be stored as multiple keys with the same value.
If two stabilizers are merged into one, the key will be the resulting stabilizer and the value will be a tuple of the two initial stabilizers.

Note that keys are single uuid and values are always tuples of uuids (can also be tuples of a single value).

- :attr:`~loom.interpreter.interpretation_step.InterpretationStep.logical_x_evolution` / :attr:`~loom.interpreter.interpretation_step.InterpretationStep.logical_z_evolution` : :code:`dict[str, tuple[str, ...]]`

This dictionary is used to describe the evolution of logical operators during lattice surgery operations.
It maps the uuid of the new operator (key) to the previous operator(s) and stabilizer(s) that create the new operator.
In the case an operator is transformed by multiplying it with stabilizers, these stabilizers will be added to the evolution dictionary.

The evolution dictionaries are also used to propagate updates using :meth:`~loom.interpreter.interpretation_step.InterpretationStep.update_logical_operator_updates_MUT`.
If a logical operator is included in the evolution of another, then its updates will be propagated to the next.
In the following examples, the propagation is ignored unless explicitly written.

- :attr:`~loom.interpreter.interpretation_step.InterpretationStep.block_evolution` : :code:`dict[str, tuple[str, ...]]`

This dictionary keeps track of the evolution of blocks using their UUID.
Similarly to stabilizers, blocks that are transformed during an operation will have their id stored in the dictionary in a final-to-initial mapping.

- :attr:`~loom.interpreter.interpretation_step.InterpretationStep.block_qec_rounds` : :code:`dict[str, int]`

This counter keeps track of how many rounds of syndrome extraction were performed on a code block.
It is used to create the measurement registers in which the classical bits referring to syndromes are stored.
The :attr:`~loom.interpreter.syndrome.Syndrome.round` field of :py:attr:`~loom.interpreter.syndrome.Syndrome` objects created at interpretation is determined by the value of this object.
Note that the counter is reset every time a block is transformed (any :class:`~loom.eka.operations.base_operation.Operation` outside of :class:`~loom.eka.operations.code_operation.MeasureBlockSyndromes` modifies the code block).

This is a counter that is local in time and not bound to the physical location of the qubits (or :class:`~loom.eka.circuit.Channel` ).

- :py:attr:`~loom.interpreter.interpretation_step.InterpretationStep.cbit_counter` : :code:`dict[str, int]`

This counter keeps track of how many times a qubit is measured during the whole set of operations.
It maps the label of the classical channel that is the target of the measurement operation to the index at which that measurement is stored.
The :py:attr:`~loom.interpreter.interpretation_step.InterpretationStep.cbit_counter` field is used to generate the :code:`Cbit` s automatically during interpretation.
The method :code:`get_new_cbit_MUT` is responsible of incrementing the counter and returning the :code:`Cbit`.

E.g. qubit :code:`"(4, 2, 0)"` is measured in the classical register :code:`"c_(4, 2, 0)"` for the third time, the channel used will be labelled :code:`"c_(4, 2, 0)_2"` and the resulting cbit counter is :code:`{"c_(4, 2, 0)": 2}`.
The associated :code:`Cbit` is :code:`(c_(4, 2, 0), 2)`.
During the next round of syndrome extraction, if :code:`"(4, 2, 0)"` is measured again onto the register :code:`"c_(4, 2, 0)"`, :py:attr:`~loom.interpreter.interpretation_step.InterpretationStep.get_new_cbit_MUT` will be called and :py:attr:`~loom.interpreter.interpretation_step.InterpretationStep.cbit_counter` will be incremented. The channel created will be labelled :code:`"c_(4, 2, 0)_3"`.
The resulting :code:`Cbit` will be :code:`(c_(4, 2, 0), 3)`.

- :attr:`~loom.interpreter.interpretation_step.InterpretationStep.channel_dict` : :code:`dict[str, Channel]`

This dictionary is a mapping between UUIDs and their respective channel.
This is used for convenience to access the channels outside of the right context.

- :attr:`~loom.interpreter.interpretation_step.InterpretationStep.stabilizer_updates` : :code:`dict[str, tuple[Cbit, ...]]`

This dictionary is used to describe the measurements we keep track of to ensure that the output stabilizers are in a deterministic state (in the noiseless case).
The measurements themselves may not be, but the final value associated with the stabilizer measurement should be (assuming they have been measured in the past and the block is in a quiescent state).
Whenever data qubits that are part of a stabilizer are measured, they need to be included in the update dictionaries in some way.

- :attr:`logical_x/z_operator_updates<loom.interpreter.interpretation_step.InterpretationStep.logical_x_operator_updates>` : :code:`dict[str, tuple[Cbit, ...]]`

This dictionary is used to describe the measurements we keep track of to ensure that the output observable is deterministic.
The measurements themselves may not be deterministic but the final product is.
Whenever data qubits that are included in logical operators are measured, they need to be included in the update dictionaries in some way.
This is also the case when a logical operator is displaced by multiplying it with a stabilizer.
In that case, the most recent syndrome should be included in the updates too.

In some cases like in the lattice surgery phase, updates of the :math:`Z` operator should be appended to the :math:`X` operator updates.

- :attr:`~loom.interpreter.interpretation_step.InterpretationStep.block_history` : :code:`tuple[tuple[Block, ...]]`

This fields stores the full history of blocks for the given list of operations.
It currently creates a tuple of blocks every time an operation is executed.

This could be improved by only reflecting changes of a full time step in the block history (and not every single operation within these time steps).

- :attr:`~loom.interpreter.interpretation_step.InterpretationStep.is_frozen` : :code:`bool`

This flag is used to signal that interpretation is completed and the fields may not be mutated anymore.

Usage in the wild
^^^^^^^^^^^^^^^^^

Here are examples of fields that are updated throughout interpretation of lattice surgery operations.
The two most important (and tricky) fields to modify are :code:`updates` and :code:`evolution`.
These fields are responsible for the automatic generation of :class:`~loom.interpreter.syndrome.Syndrome` and corrections.

A convention that we follow for the two codes described here is that logical operators are tied to the top-left qubit (minimum coordinates in all directions).
The consequence of this is that these operators are displaced if we add new qubits to the top-left of the block.
E.g. growing a rotated surface code to the left will displace the logical operators by the same distance as specified in grow.

Examples for the repetition code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Notation**:

`Long logical` describes the logical that spans all qubits and which pauli is opposite to the check type.

`Short logical` describes the logical operator that only acts on a single qubit and is of the same pauli type as the checks/stabilizers.

`Required stabilizers` describe the :class:`~loom.eka.stabilizer.Stabilizer` objects (sometimes by their UUID) that are required to go from one logical operator to the other.
They satisfy the relation: :math:`\{S_{\text{required}}\} = \{S_i | L' = L\prod_{i}S_i\}` , where :math:`L` and :math:`L'` are respectively the initial and final logical operators.

`Required syndromes` describe the :class:`~loom.interpreter.syndrome.Syndrome` objects corresponding to the required stabilizers measurements.

- **Grow** is basically an identity operation, though one of the operators (the long operator) will get longer.
  
  The operation consists of resetting qubits in the grow direction (left or right) in the basis that matches the code stabilizers.

  The evolution of operators is then:

  .. code-block:: python

        # The long operator gets longer
        evolution = {
            new_long_op.uuid: (old_long_op.uuid,),
        }

  where :code:`old_long_op` grew into :code:`new_long_op`.

  Nothing happens to the other operators.

  Note that by default a grow towards the left will displace the associated logical operator (short/single qubit operator), populating the evolution dictionary with the required stabilizers and update dictionary with the most recent syndromes:

  .. code-block:: python

        # The long operator gets longer and the short one is displaced
        evolution = {
            new_long_op.uuid: (old_long_op.uuid,)
            new_short_op.uuid: (old_short_op.uuid,) + tuple(stab.uuid for stab in stabs_required)
        }

        updates = {
            new_short_op.uuid: tuple(meas for meas in required_syndromes)
        }

  if and only if the (short) logical operator is displaced.

- **Shrink** can also be considered an identity operation, though we are measuring some data qubits and need to keep these into account in the update dictionaries.
  If the operator is not displaced by the shrink, we have:

  .. code-block:: python

        evolution = {
            new_long_op.uuid: (old_long_op.uuid,)
        }
        updates = {
            new_long_op.uuid: tuple(meas for meas in shrink_measurements)
        }

  where :code:`shrink_measurements` are all the measurements on data qubits that are part of :code:`old_long_op.data_qubits`.

  The new operator :code:`new_long_op` also inherits updates from :code:`old_long_op`.
  If we shrink a repetition code from the left, the short operator is displaced, resulting in:

  .. code-block:: python

        evolution = {
        new_short_op.uuid: (old_short_op.uuid,) + tuple(stab.uuid for stab in stabs_required)
        }
        updates = {
            new_short_op.uuid: tuple(meas for meas in shrink_measurements) + tuple(meas for meas in required_syndromes)
        }


- **Merge** is equivalent to a joint measurement of two blocks in the basis of the shared logical operator (the short one).
  This is not a unitary operation because the number of logical operators after the operation is reduced.
  
  In the case of a merge, the final long operator is made up of the two initial long operators.
  If the short operator is not displaced, we have:
  
  .. code-block:: python
        
        evolution = {
            new_long_op.uuid: (old_long_op_1.uuid, old_long_op_2.uuid,)
        }

  The new operator will inherit updates from the previous operators.
  As usual, if the preserved (short) operator needs to be displaced, we need to account for it in the evolution and update dictionaries, resulting in:

  .. code-block:: python
    
        evolution = {
            new_long_op.uuid: (old_long_op_1.uuid, old_long_op_2.uuid,),
            new_short_op.uuid: (old_short_op_1,) + tuple(stab.uuid for stab in stabs_required)
        }
        updates = {
            new_short_op.uuid: tuple(meas for meas in required_syndromes)
        }

- **Split** distributes the logical information in two blocks that are separated in space.
  This is done through physical qubits measurements.
  Split can be understood as the converse operation to merge.
  The measurements occurring during a split need to be kept track of to update the state of logical operators and stabilizers.
  
  If no logical operator is measured out (:code:`new_short_op_1` is the same as :code:`old_short_op_1`), we have:\

  .. code-block:: python
    
        # Only one of the operators inherits updates from the initial operator
        evolution = {
            new_long_op_1.uuid: (old_long_op.uuid,)
            new_long_op_2.uuid: (old_long_op.uuid,)
            new_short_op_2.uuid: (old_short_op.uuid,) + tuple(stab.uuid for stab in stabs_required)
        }
        updates = {
            new_long_op_1: (split_meas,)
            new_short_op_2.uuid: tuple(meas for meas in required_syndromes)
        }

  Note that for the repetition code, no stabilizer is partially measured and thus we do not have to keep track of any stabilizer in evolution or updates.

Examples for the surface code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Notation**:

`Standard code block` describes a `RotatedSurfaceCode` instance for a single logical qubit where the logical operators are tied to the upper-left qubit of the code bock.
This means that there is exactly one :class:`~loom.eka.pauli_operator.PauliOperator` in :py:attr:`~loom.eka.block.Block.logical_x_operators` and one in :py:attr:`~loom.eka.block.Block.logical_z_operators`.

`Vertical logical` describes the logical operator that spans the vertical (left) boundary of the rotated surface code block.

`Short logical` describes the logical operator that spans the horizontal (top) boundary of the rotated surface code block.

`Required stabilizers` describe the :class:`~loom.eka.stabilizer.Stabilizer` objects (sometimes by their UUID) that are required to go from one logical operator to the other.
They satisfy the relation: :math:`\{S_{\text{required}}\} = \{S_i | L' = L\prod_{i}S_i\}` , where :math:`L` and :math:`L'` are respectively the initial and final logical operators.

`Required syndromes` describe the :class:`~loom.interpreter.syndrome.Syndrome` objects corresponding to the required stabilizers measurements.

- **Grow** is basically an identity operation, though one of the operators (the operator that is parallel to the grow direction -- horizontal for left/right) will get longer.

  This results in an increased distance for the other operator.
  The operation consists of resetting qubits in the grow direction (left/right/top/bottom) in the basis that matches the code stabilizers.
  For a grow to the right, the evolution of operators is:

  .. code-block:: python

        # The horizontal operator gets longer
        evolution = {
            new_horizontal_op.uuid: (old_horizontal_op.uuid,),
        }

  where :code:`old_horizontal_op` grew into :code:`new_horizontal_op`.
  Nothing happens to the other (vertical) operator.
  Note that by default a grow towards the left does not displace the vertical logical operator to the left.

- **Shrink** can also be considered an identity operation, though we are measuring some data qubits and need to keep these into account in the update dictionaries.
  Consider a horizontal shrink that does not displace the vertical logical operator (e.g. from the right):

  .. code-block:: python

        evolution = {
            new_horizontal_op.uuid: (old_horizontal_op.uuid,),
        } 
        updates = {
            new_horizontal_op.uuid: tuple(meas for meas in shrink_measurements)
        }

  where :code:`shrink_measurements` are all the measurements on data qubits that are part of :code:`old_horizontal_op.data_qubits`.
  The new operator :code:`new_horizontal_op` also inherits updates from :code:`old_horizontal_op`.

  If we shrink a surface code from the left, the vertical operator is displaced, resulting in:

  .. code-block:: python

        evolution = {
            new_horizontal_op.uuid: (old_horizontal_op.uuid,)
            new_vertical_op.uuid: (old_vertical_op,) + tuple(stab.uuid for stab in stabs_required)
        }
        updates = {
            new_horizontal_op.uuid: tuple(meas for meas in shrink_measurements)
            new_vertical_op.uuid: tuple(meas for meas in required_syndromes)
        }

- **Merge** is equivalent to a joint measurement of two blocks in the basis of the shared logical operator (aligned with the merge orientation).
  This is not a unitary operation because the number of logical operators after the operation is reduced.
  In the case of a horizontal merge, the final horizontal operator is made up of the two initial horizontal operators.
  In case operators are aligned already:

  .. code-block:: python

        evolution = {
            new_horizontal_operator.uuid: (old_horizontal_op_1.uuid, old_horizontal_op_2.uuid,)
        }

  The new (horizontal) operator will also inherit updates from the previous operators.
  Note that the resulting vertical operator will be inherited from the left-most (or top) block.

  If the blocks are aligned but the operators to be merged (horizontal) are not, we first need to account for stabilizers required to align them.
  The stabilizers will bet tracked in the evolution and the syndromes in updates:

  .. code-block:: python

        evolution = {
            new_long_op.uuid: (old_horizontal_op_1.uuid, old_horizontal_op_2.uuid,) + tuple(stab.uuid for stab in stabs_required)
        }
        updates = {
            new_horizontal_operator.uuid: tuple(meas for meas in required_syndromes)
        }

  The vertical operators are not modified.

- **Split** distributes the logical information in two blocks that are separated in space.
  This is done through physical qubits measurements.
  Split can be understood as the converse operation to merge.
  The measurements occurring during a split need to be kept track of to update the state of logical operators and stabilizers.
  In the case of a horizontal split, if no logical operator is measured out (:code:`new_horizontal_op_1` is the same as :code:`old_horizontal_op_1`), we have:

  .. code-block:: python

        # Only one of the operators inherits updates from the initial operator
        evolution = {
            new_horizontal_op_1.uuid: (old_horizontal_op.uuid,)
            new_horizontal_op_2.uuid: (old_horizontal_op.uuid,)
            new_vertical_op_2.uuid: (old_vertical_op.uuid,) + tuple(stab.uuid for stab in stab_required)
        }
        # new_vertical_op_1 is inherited from the initial block
        updates = {
            new_long_op_1: (split_meas,)
            new_vertical_op_2.uuid: tuple(meas for meas in required_syndromes)
        }

  If the initial vertical operator is fully measured by the split, both new vertical operators are displaced using stabilizers/syndromes.

