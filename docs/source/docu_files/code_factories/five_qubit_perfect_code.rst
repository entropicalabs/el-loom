Five Qubit Perfect Code Factory
===============================
Five Qubit Perfect Code Block
-------------------------------
The five qubit perfect code is a simple :math:`[5, 1, 3]` error-correcting code that encodes a single logical qubit with :math:`5` physical qubits. 

.. code-block:: python
    
    from loom_five_qubit_perfect_code.code_factory import FiveQubitPerfectCode
    from loom.eka import Lattice

    lattice = Lattice.poly_2d((10, 10), 5, anc=4)

    # Create a block for a five qubit perfect code
    myLogicalqubit = FiveQubitPerfectCode.create(
        lattice=lattice,
        unique_label="five_qubit_perfect_code",
        position=(0, 0),
    )

Operations on Five Qubit Perfect Code
-------------------------------------
We currently do not provide any special code operations specific to the Five Qubit Perfect Code. 
The only operations available are the ones that already apply to all `Block` objects in Loom.