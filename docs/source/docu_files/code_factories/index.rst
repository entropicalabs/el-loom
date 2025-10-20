Code Factories
================
Loom provides code factories to generate common quantum error correction (QEC) codes. 
These code factories simplify the process of creating and managing error-corrected circuits by allowing users to create blocks that implement QEC codes without needing to manually implement the underlying error correction protocols.
These blocks can then be manipulated through Loom's applicators, which provide a high-level interface for performing code operations on the encoded qubits, allowing users to focus on higher-level tasks.

Note that as not all QEC codes have the same code operations, the methods available to each code factory applicator will naturally differ.
Currently, we support the following error correcting codes:

.. toctree::
   :maxdepth: 1
   :caption: Quantum Error Correcting Codes:

   Five Qubit Perfect Code <five_qubit_perfect_code.rst>
   Repetition Code <repetition_code.rst>
   Rotated Surface Code <rotated_surface_code/index.rst>
   Shor Code <shor_code.rst>
   Steane Code <steane_code.rst>