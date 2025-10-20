Repetition Code
=======================
Repetition Code Block
----------------------
The repetition code is a simple :math:`[n, 1, n]` error-correcting code that encodes a single logical qubit with :math:`n` physical qubits. 
By specifying the code distance :math:`d`, stabilizer type (X or Z), the physical lattice, and lattice position, this code factory allows the user to build :class:`Block` that describe one logical qubit with the specified parameters.

The example below shows how to create a repetition code with 5 data qubits, which encodes a single logical qubit with X stabilizers.

.. code-block:: python

    from loom_repetition_code.code_factory import RepetitionCode
    from loom.eka import Lattice

    lattice = Lattice.linear((10,))

    # Create a block for a repetition code with 5 data qubits
    myLogicalqubit = RepetitionCode.create(
        d = 5,
        check_type = "X",   # Specify the type of stabilizers (X or Z), this code will only detects Z errors
        lattice=lattice,
        unique_label = "repetition_code_5",
        position =  (0,),
    )


Operations on Repetition Code
---------------------------------

The :mod:`loom.applicator` module provides tools to apply gates and perform operations on :class:`Block` objects representing repetition codes. 
These applicators are designed to work seamlessly with the `RepetitionCode` class, providing a simple high-level interface for lattice surgery and logical operations. 
In addition to the basic applicators (Paulis, reset and measurement), this module includes specialized applicators for repetition codes. 
These include:

- grow: Expand the size of a repetition code :class:`Block` by adding qubits and stabilizers around its perimeter, increasing its logical qubit capacity or error-correcting capabilities.
- merge: Combine two adjacent repetition code :class:`Block` into a single larger block by joining their boundaries, enabling operations across the combined logical qubits.
- shrink: Reduce the size of a repetition code :class:`Block` by removing qubits and stabilizers from its edges, typically to free resources or prepare for operations like splitting.
- split: Divide a repetition code :class:`Block` into two smaller :class:`Block` by creating boundaries within the original block, allowing independent manipulation of the resulting logical qubits.

Growing, merging, shrinking, and splitting are operations that modify the structure of the repetition code, allowing for dynamic changes to the logical qubit configuration.

Example of application :

.. code-block:: python

    import loom_repetition_code as loom_repc
    from loom_repc.applicator import RepetitionCodeApplicator
    from loom_repc.code_factory import RepetitionCode
    from loom.eka import Eka, Lattice
    from loom.eka.operations import ResetAllDataQubits, MeasureBlockSyndromes, Shrink, Merge, MeasureBlockSyndromes, MeasureLogicalZ
    from loom.interpreter import interpret_eka

    lattice = Lattice.linear(...)
    # Assuming you have RepetitionCode blocks created
    rep_block_1 = RepetitionCode.create(unique_label="rep_block_1", ...)
    rep_block_2 = RepetitionCode.create(unique_label="rep_block_2", ...)

    ops = [
        ResetAllDataQubits(rep_block_1.unique_label),
        MeasureBlockSyndromes(rep_block_1.unique_label, n_cycles=1),
        Shrink(rep_block_1.unique_label, direction="right", length=2),
        Merge([rep_block_1.unique_label, rep_block_2.unique_label], "rep_block_3", orientation="horizontal"),
        MeasureBlockSyndromes("rep_block_3", n_cycles=1),
        MeasureLogicalZ("rep_block_3"),
    ]

    # Interpret the operations on the repetition code blocks
    eka = Eka(lattice, blocks=[rep_block_1, rep_block_2], operations=ops)
    # This will contain the circuit, syndromes and detectors of the system resulting from the operations.
    final_state = interpret_eka(eka)