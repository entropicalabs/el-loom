Backends
========
Loom supports conversion to different formats to run quantum experiments, allowing users to easily compare and choose the most suitable platform for their needs. These tools are provided in the :mod:`executor` package.

The following formats are currently supported:

.. contents::
   :local:
   :depth: 1

OpenQASM 3.0
------------
`OpenQASM3 <https://github.com/openqasm/openqasm>`_ is an imperative programming language for describing quantum circuits. :meth:`~executor.eka_circuit_to_qasm_converter` allows you to convert Loom experiments into OpenQASM3 format, enabling execution on any platform that supports OpenQASM3. The following example shows how to convert a Loom's :class:`~interpreter.interpretation_step.InterpretationStep` into a OpenQASM3 string:

.. literalinclude:: ../../python/executor/qasm_conversion.py
   :language: python

Stim
----
`Stim <https://github.com/quantumlib/Stim>`_ is an open-source tool for high-performance simulation of quantum stabilizer circuits. Loom experiments can be converted to Stim format using :class:`~executor.eka_circuit_to_stim_converter.EkaCircuitToStimConverter`. This allows for efficient simulation and analysis of quantum circuits. The following example shows how to convert an :class:`~interpreter.interpretation_step.InterpretationStep` to Stim (check the list of Stim's supported operations `here <https://github.com/quantumlib/Stim/blob/main/doc/gates.md>`_):

.. literalinclude:: ../../python/executor/stim_conversion.py
   :language: python