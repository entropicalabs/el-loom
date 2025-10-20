Circuit
=======

The lowest layer of representation, used as foundation for QEC, is defined to be a circuit. In Loom, a circuit is defined as a sequence of operations that are applied to a set of data registers over discrete time steps, with the only constraint being that no two operations can be applied to the same data register at the same time step. This is a very flexible definition, which allows us to represent any quantum and/or classical circuit and is used to create more complex representations, such as quantum error correction schemes and high-level quantum algorithms involving QEC.

Channels
^^^^^^^^
The data registers, called channels in Loom, are represented by the :class:`~loom.eka.circuit.Channel` class. A channel can hold either classical or quantum data. A :class:`~loom.eka.circuit.Channel` is uniquely identified by its `id`, it can also have a `label` and a `type`
(either classical or quantum). The :class:`~loom.eka.circuit.ChannelType` of a channel is assumed to be constant throughout the lifetime of the channel. The example below shows how to create channels in Loom:

.. literalinclude:: ../../python/circuit/channel_creation.py
    :language: python

Circuits
^^^^^^^^

The circuits in Loom are represented by the :class:`~loom.eka.circuit.Circuit` class, which uses a recursive structure. An instance of :class:`~loom.eka.circuit.Circuit` is defined as a sequence of circuits and the set of channels it acts on. At the lowest level, we have operations (often called gates) that are represented as :class:`~loom.eka.circuit.Circuit` with an empty sequence and a set of targeted data channels. The following example shows how to create gates and simple circuits in Loom :

.. literalinclude:: ../../python/circuit/simple_circuit.py
    :language: python

Loom's framework is meant to be as flexible as possible; therefore, one can create operations with any names and any number of classical/quantum channels. These are only abstract representations.

In circuits, we typically want to have operations acting on disjoint channels to be run in parallel. This can be done by wrapping the parallel circuit elements in a tuple. Within an instance of :class:`~loom.eka.circuit.Circuit`, the sequence of operations has to contain either :class:`~loom.eka.circuit.Circuit` instances or tuples of :class:`~loom.eka.circuit.Circuit` instances, but not both, so make sure to wrap all the elements in tuples. The example below shows how to create a circuit with parallel operations:

.. literalinclude:: ../../python/circuit/parallel_operation.py
    :language: python

**Important note**: When using parallel execution (i.e., providing a `tuple(tuple(Circuit, ...), ...)` as circuit parameter), the elements within the sequence of tuples will be executed on the step corresponding to their index in the sequence, regardless of the duration of the previous elements. This can lead to an unexpected error situation where 2 gates are applied to the same channel at the same time step. Loom leaves the user freedom on how to pad the circuit (you may use empty tuples to do so). The example below illustrates this:

.. literalinclude:: ../../python/circuit/scheduling_issue.py
   :language: python

The :class:`~loom.eka.circuit.Circuit` class provides a method to automatically pad the circuit by padding with empty tuples after elements of duration more than 1. This may result in a suboptimal circuit :

.. literalinclude:: ../../python/circuit/auto_padding.py
   :language: python


Utilities
^^^^^^^^^
The :class:`~loom.eka.circuit.Circuit` class provides a set of utilities to manipulate circuits:

- :meth:`~loom.eka.circuit.Circuit.flatten` : 
    Returns a flattened copy of the circuit where the sub-sequence of circuit is a list of base operations (i.e., circuits with no sub-circuits). The parallel operations will be flattened and get executed in series.
- :meth:`~loom.eka.circuit.Circuit.unroll` : 
    Unroll the recursive structure and provide a representation of the circuit as a sequence containing only base operations while preserving the time structure.
- :meth:`~loom.eka.circuit.Circuit.clone` : 
    This allows us to clone a circuit structure and have it assigned to other channels.
- :meth:`~loom.eka.circuit.Circuit.detailed_str` : 
    Which provides a string expression for visualization of the circuit in more detail, including which channel the operations are applied to.

Some extra utilities exist for defining circuits independently of channels.

- :meth:`~loom.eka.circuit.Circuit.from_circuits` : 
    Build a :class:`~loom.eka.circuit.Circuit` object from a representation of its content with relative qubits indices.
- :meth:`~loom.eka.circuit.Circuit.as_gate`: 
    Create a gate without needing to specify channels (they will be automatically generated).


