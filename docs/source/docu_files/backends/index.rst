Backends
========
Loom supports conversion to different formats to run quantum experiments, allowing users to easily compare and choose the most suitable platform for their needs. These tools are provided in the :mod:`~loom.executor` package.

The following formats are currently supported:

.. contents::
   :local:
   :depth: 1

OpenQASM 3.0
------------
`OpenQASM3 <https://github.com/openqasm/openqasm>`_ is an imperative programming language for describing quantum circuits. :meth:`~loom.executor.eka_circuit_to_qasm_converter` allows you to convert Loom experiments into OpenQASM3 format, enabling execution on any platform that supports OpenQASM3. The following example shows how to convert a Loom's :class:`~loom.interpreter.interpretation_step.InterpretationStep` into a OpenQASM3 string:

.. literalinclude:: ../../python/executor/qasm_conversion.py
   :language: python

Stim
----
`Stim <https://github.com/quantumlib/Stim>`_ is an open-source tool for high-performance simulation of quantum stabilizer circuits. Loom experiments can be converted to Stim format using :class:`~loom.executor.eka_circuit_to_stim_converter.EkaCircuitToStimConverter`. This allows for efficient simulation and analysis of quantum circuits. The following example shows how to convert an :class:`~loom.interpreter.interpretation_step.InterpretationStep` to Stim (check the list of Stim's supported operations `here <https://github.com/quantumlib/Stim/blob/main/doc/gates.md>`_):

.. literalinclude:: ../../python/executor/stim_conversion.py
   :language: python

Pennylane
---------
`Pennylane <https://github.com/PennyLaneAI/pennylane>`_ is an open-source python framework for quantum programming built by Xanadu. Loom experiments can be converted to a format that is compatible with Pennylane's simulators using :meth:`~loom.executor.eka_circuit_to_pennylane_converter.convert_circuit_to_pennylane`. The output format can also be used for Pennylane's catalyst simulator via the ``is_catalyst`` input boolean. Note that in order to use this exector with loom, you are required to have had installed :mod:`pennylane` and :mod:`catalyst` beforehand. We recommend getting :mod:`pennylane-catalyst` version `0.13.0`:

.. literalinclude:: ../../python/executor/pennylane_conversion.py
   :language: python

Cudaq
-----
`CudaQ <https://github.com/NVIDIA/cuda-quantum>`_ is an open-source quantum development platform for the orchestration of software and hardware resources designed for large-scale quantum computing applications built by Nvidia. Loom experiments can be converted to a format that is compatible with the cudaq simulators using :class:`~loom.executor.eka_circuit_to_cudaq_converter.EkaToCudaqConverter`. Note that in order to use this executor with loom, you are required to have had installed :mod:`cudaq` beforehand. We recommend getting :mod:`cudaq` and :mod:`cuda-quantum-cu12` at version `0.12.0`:

.. literalinclude:: ../../python/executor/cudaq_conversion.py
   :language: python