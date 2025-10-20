Build a Quantum Error Correction Experiment
===========================================

Loom provides a high-level abstraction dataclass, named :class:`~loom.eka.eka.Eka`, to work with the stabilizers for your quantum error correction code. The tool can be accessed via the Eka package.

.. code-block:: python

    from loom.eka import Eka

The :class:`~loom.eka.eka.Eka` class takes in three inputs — a :class:`~loom.eka.lattice.Lattice` object, a list of :class:`~loom.eka.block.Block` objects, and a list of :class:`~loom.eka.operations.base_operation.Operation` objects — and builds all of the necessary components for the experiment.

.. code-block:: python

    eka_obj = Eka(
        lattice=...,
        blocks=[...],
        operations=[...],
    )

This provides a simple yet compact representation of a logical circuit, though the logical circuit needs to be interpreted before it can be executed using a backend. 

To learn more about using :class:`~loom.eka.eka.Eka`, please refer to the :doc:`Eka Architecture <../eka_qec/eka>` documentation. 
To proceed on to visualization features, please refer to the :doc:`Visualize stabilizers and qubits <loom_visualize>` documentation.
To learn more about backends and interpretation, please refer to the :doc:`Run an experiment <loom_run>` documentation.



.. Lattice
.. --------
.. This object defines the geometry and coordinates of the system. 

.. .. code-block:: python

..     from loom.eka import Lattice

.. It can be created using one of the built-in methods, such as :meth:`~loom.eka.lattice.Lattice.square_2d` (especially if you're working with the Rotated Surface Code), or :meth:`~loom.eka.lattice.Lattice.linear` (for the Repetition Code). 
.. Simply provide a shape argument to define the size of the lattice. If no shape is provided, an infinite lattice will be created.

.. .. code-block:: python

..     lattice_1 = Lattice.linear((7, )) # Finite lattice containing 7 data qubits
..     lattice_2 = Lattice.square_2d((5, 4)) # Finite lattice containing 5x4 data qubits
..     lattice_3 = Lattice.square_2d() # Infinite lattice

.. Note that the :mod:`~loom.visualizer` package in Loom can only visualize finite lattices. Refer to :doc:`Visualize stabilizers and qubits <loom_visualize>` for more on visualization.

.. Block
.. --------
.. This object represents a collection of qubits and their associated stabilizers and logical operators.

.. .. code-block:: python

..     from loom.eka import Block

..     block_1 = Block(
..         stabilizers=(Stabilizer("ZZ", ((0, 0, 0), (1, 0, 0))),),
..         logical_x_operators=(PauliOperator("XX", ((0, 0, 0), (1, 0, 0))),),
..         logical_z_operators=(PauliOperator("Z", ((0, 0, 0),)),),
..         unique_label="q1",  # optional argument to identify the block
..     )

.. The :class:`~loom.eka.block.Block` class takes in three inputs — a list of :class:`~loom.eka.stabilizer.Stabilizer` objects, a list of :class:`~loom.eka.pauli_operator.PauliOperator` objects representing logical X operators, and a list of :class:`~loom.eka.pauli_operator.PauliOperator` objects representing logical Z operators. 
.. These inputs will be validated to ensure that the QEC code is well-defined. Please refer to :doc:`Block Architecture <../eka_qec/block>` for more details on the list of validators.

.. Operation
.. ----------
.. This object represents a quantum operation that can be performed on the :class:`~loom.eka.block.Block`. We provide a set of pre-built :class:`~loom.eka.operation.Operation` objects that can be used to manipulate the blocks, such as measuring syndromes or performing logical operations.

.. .. code-block:: python

..     from loom.eka.operations import *

..     measure_syndrome = MeasureBlockSyndromes("q1", n_cycles=3)
..     measure_logical_z = MeasureLogicalZ("q1")
..     logical_hadamard = Hadamard("q1")

.. Logical gates are provided within the :mod:`~loom.eka.operations.logical_operation` module, while :mod:`~loom.eka.operations.code_operation` contains lower-level operations that manipulate the code itself.
.. For instance, :class:`~loom.eka.operations.logical_operation.Hadamard` is a logical operation that applies a Hadamard gate to a specified :class:`~loom.eka.block.Block`, while :class:`~loom.eka.operations.code_operation.MeasureBlockSyndromes` is a code operation that measures the syndromes of the code for error correction.

.. Putting it all together
.. -------------------------

.. We can construct an experiment that performs a state initialization on a Rotated Surface Code block.


.. .. code-block:: python
..     import loom.eka.operations.code_operation as code_op
..     from loom.eka import Eka
..     from loom.eka import Lattice
..     from loom_rotated_surface_code.code_factory import RotatedSurfaceCode   # if you have loom_rotated_surface_code installed

..     lattice = Lattice.square_2d((4, 4))
..     alpha = RotatedSurfaceCode.create(...)

..     operations = (
..         (   # Encode pt1: Reset all data qubits
..             code_op.ResetAllDataQubits('alpha', state="0"),
..         ),
..         (   # Encode pt2: Encode data qubits via measurements
..             code_op.MeasureBlockSyndromes('alpha', n_cycles=1),
..         ),
..         (   # Syndromes: Information collection
..             code_op.MeasureBlockSyndromes('alpha', n_cycles=2),
..         ),
..         (   # Measure logicals
..             code_op.MeasureLogicalZ('alpha'),
..         ),
..     )

..     eka_obj = Eka(lattice=lattice, blocks=[alpha], operations=operations)

.. import loom.eka.operations.logical_operation as log_op
