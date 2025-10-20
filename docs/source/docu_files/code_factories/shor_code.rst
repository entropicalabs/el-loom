Shor Code Factory
=================
Shor Code Block
-----------------
The Shor code is a simple :math:`[9, 1, 3]` error-correcting code that encodes a single logical qubit with :math:`9` physical qubits. 

.. code-block:: python
    
    from loom_shor_code.code_factory import ShorCode
    from loom.eka import Lattice

    lattice = Lattice.linear((10,))

    # Create a block for a shor code
    myLogicalqubit = ShorCode.create(
        lattice=lattice,
        unique_label="shor_code",
        position=(0,),
    )

Operations on Shor Code
-------------------------
We currently do not provide any special code operations specific to the Shor code. 
The only operations available are the ones that already apply to all `Block` objects in Loom.