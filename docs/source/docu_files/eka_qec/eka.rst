.. _eka:

Eka
===

:class:`~loom.eka.eka.Eka` is a data structure that contains all the information required to describe a logical quantum circuit with embedded error correction code.

The data structure itself and all its sub-components implement a comprehensive set of validations that reduces the room for low-level errors when using these abstractions with the rest of Loom.

:class:`~loom.eka.eka.Eka` was created to compactly represent an error-corrected quantum circuit. It allows the creation of QEC codes using the :class:`~loom.eka.block.Block` component.

It is composed of three main components:

- lattice: a :class:`~loom.eka.lattice.Lattice` object that defines the geometry (and coordinates) of the system.
- blocks: a set of :class:`~loom.eka.block.Block` objects that represent all the logical qubits in the initial state of our logical algorithm. They are given in the form of a tuple.
- operations: a set of :mod:`~loom.eka.operations` that represent the different transformations that will be applied on the system to perform the intended computation. These can be code-agnostic (e.g. measuring syndromes) or code-specific (e.g. lattice surgery, state preparation, etc.). They are given in the form of timeslices, e.g. :code:`operations[i]` is as tuple of operations executed in parallel at time :math:`i`.

More automations are provided as plugins to the main :mod:`~loom.eka.eka.Eka` module.

Example
^^^^^^^
Let's create a quantum memory experiment using a simple repetition code.

We start by defining the geometry of the system using :class:`~loom.eka.lattice.Lattice`. Since the repetition code can be represented on a line, we choose the linear lattice:

.. code-block:: python

    from loom.eka import Lattice

    lattice = Lattice.linear(lattice_size=(3,))

Then we define our logical qubits using :class:`~loom.eka.block.Block` (see the detailed documentation for the repetition code :ref:`example <block_example>`). We can define a logical qubit:

.. code-block:: python
    
    from loom.eka import Block, Stabilizer, PauliOperator

    logical_qubit = Block(
        unique_label="q1",
        stabilizers=(
            Stabilizer("ZZ", ((0, 0), (1, 0)), ancilla_qubits=((0, 1),)),
            Stabilizer("ZZ", ((1, 0), (2, 0)), ancilla_qubits=((1, 1),)),
        ),
        logical_x_operators=(PauliOperator("XXX", ((0, 0), (1, 0), (2, 0))),),
        logical_z_operators=(PauliOperator("Z", ((0, 0),)),)
    )

Finally we need to define the operations we want to apply on the code using :class:`~loom.eka.operations.base_operation.Operation`. We will first reset the qubit in a known state, e.g. :math:`|0\rangle` measure its syndromes multiple times and finally measure the logical operator:

.. code-block:: python

    from loom.eka.operations import ResetAllDataQubits, MeasureBlockSyndromes, MeasureLogicalZ

    reset_op = ResetAllDataQubits("q1", state="0")
    measure_synd_op = MeasureBlockSyndromes("q1", n_cycles=3)
    measure_log_op = MeasureLogicalZ("q1")

We have all the components and can now build the :class:`~loom.eka.eka.Eka` dataclass:

.. code-block:: python

    from loom.eka import Eka

    my_eka = Eka(
        lattice=lattice, 
        blocks=[logical_qubit], 
        operations=[reset_op, measure_synd_op, measure_log_op]
    )

Now we have a compact representation of the logical circuit that implements a simple quantum memory with a custom defined repetition code.

The next step would be to interpret it and execute it on a backend. 

Validations
^^^^^^^^^^^

The :class:`~loom.eka.eka.Eka` dataclass implements some simple validation functions. This is done to ensure that certain constraints enforced in the later stages are satisfied and to prevent the user from starting from scratch due to an invalid definition of :class:`~loom.eka.eka.Eka`.

The list of validators includes:

- All physical qubits indices are compatible with the choice of lattice and its size.
- All code blocks have a unique label.
- All operations are disjoint: there is no two operations acting on the same logical qubits at the same time step.
