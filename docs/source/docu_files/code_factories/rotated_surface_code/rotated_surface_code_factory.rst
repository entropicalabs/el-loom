Rotated Surface Code
===========================
Rotated Surface Code Block
--------------------------

.. contents::
    :local:
    :depth: 3

The rotated surface code is a popular QECC as it provides a high threshold with efficient usage of the physical qubits. 
This package provides tools to build :class:`Block` objects for rotated surface codes, which describe one or more logical qubits, and provide information on how to perform logical operations and error correction at the physical level. 
These :class:`Block` s can then be used as an abstraction for logical qubits, and Loom provides tools to easily apply gates to it in order to perform circuit simulation on corrected logical qubits.

The example below shows how to create a :class:`Block` for a rotated surface code of distance 3 (that represents one logical qubit).

.. code-block:: python

    import loom_rotated_surface_code as loom_rsc
    from loom_rsc.code_factory import RotatedSurfaceCode
    from loom.eka import Lattice

    # Create a block for a rotated surface code of distance 3
    lattice = Lattice.square_2d((5,5))
    rsc_block = RotatedSurfaceCode.create(
        3,
        3,
        lattice,
        unique_label="rotated_surface_code",
        position=(0, 0),
        
    )

Operations on Rotated Surface Codes
------------------------------------

The :mod:`~loom_rotated_surface_code.applicator` module provides tools to apply gates and perform operations on :class:`~loom.eka.block.Block`  objects representing rotated surface codes. 
These applicators are designed to work seamlessly with the :class:`~loom_rotated_surface_code.code_factory.rotated_surface_code.RotatedSurfaceCode` class.

Applicators provide implementations of Loom's high-level :class:`Operation` for a specific code. This package provide implementations for the following operations for the rotated surface code:

- :class:`Grow`: Expand the size of a rotated surface code :class:`Block` by adding qubits and stabilizers around its perimeter, increasing its logical qubit capacity or error-correcting capabilities.
- :class:`Merge`: Combine two adjacent rotated surface code :class:`Block` into a single larger block by joining their boundaries, enabling operations on the resulting logical qubit.
- :class:`Shrink`: Reduce the size of a rotated surface code :class:`Block` by removing qubits and stabilizers from its edges, typically to prepare for operations like merging that require boundaries properly aligned and of equal size.
- :class:`Split`: Divide a rotated surface code :class:`Block` into two smaller :class:`Block` by creating boundaries within the original block, allowing independent manipulation of the resulting logical qubits.

Growing, merging, shrinking, and splitting are operations that modify the structure of the rotated surface code, allowing for dynamic changes to the logical qubit configuration.

Example of application :

.. code-block:: python

    from loom_rotated_surface_code.code_factory import RotatedSurfaceCode
    from loom.eka import Eka, Lattice
    from loom.interpreter import interpret_eka
    from loom.eka.operations import *

    # Create a square 2D lattice of qubits
    lattice = Lattice.square_2d((15,15))

    # Create two rotated surface code blocks
    rsc_block_1 = RotatedSurfaceCode.create(5, 5, lattice, unique_label="rsc_block_1")
    rsc_block_2 = RotatedSurfaceCode.create(5, 5, lattice, unique_label="rsc_block_2", position=(6, 0))

    # Add some operations on the blocks
    operations = [
        ResetAllDataQubits(rsc_block_1.unique_label),
        MeasureBlockSyndromes(rsc_block_1.unique_label, n_cycles=1),
        Shrink(rsc_block_1.unique_label, direction="right", length=2),
        Merge([rsc_block_1.unique_label, rsc_block_2.unique_label], "rsc_block_3"),
        MeasureBlockSyndromes("rsc_block_3", n_cycles=1),
        MeasureLogicalZ("rsc_block_3"),
    ]

    # Interpret the operations on the rotated surface code blocks
    eka_experiment = Eka(lattice, blocks=[rsc_block_1, rsc_block_2], operations=operations)
    # This will contain the circuit, syndromes and detectors of the system resulting from the operations.
    final_state = interpreter.interpret_eka(eka_experiment)

To view further examples on some sample experiments using the rotated surface code factory, 
please refer to :doc:`Rotated Surface Code Examples <examples>`.