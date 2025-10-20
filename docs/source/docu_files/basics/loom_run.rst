Run an experiment
==================

Loom experiments are designed to be backend-agnostic, meaning they can run on various quantum computing or simulators platforms, allowing you to easily compare results across different technologies. 
To achieve this, Loom comes packaged with converters that bridge our data representation to other formats.

Firstly, we have to interpret the :class:`~loom.eka.eka.Eka` object into an intermediate representation using the :func:`~loom.interpreter.interpreter.interpret_eka` function.
After obtaining the interpreted representation, you can then convert it into a specific backend format using one of the available converters. 
For instance, to convert to a Stim circuit, you can use the :class:`~loom.executor.eka_circuit_to_stim_converter.EkaCircuitToStimConverter` class.

.. code-block:: python

    from loom.interpreter.interpreter import interpret_eka
    from loom.executor.eka_circuit_to_stim_converter import EkaCircuitToStimConverter

    eka_obj = Eka(...)

    interpreted_eka = interpret_eka(eka_obj)

    converter = EkaCircuitToStimConverter()
    stim_circuit = converter.convert(interpreted_eka)

The resultant ``stim_circuit`` can then be executed using the Stim library.

To see other examples of how these backends can be used, visit the :doc:`Converter </../../notebooks/converter_example>` tutorial. To learn more about the full list of backends, visit the :doc:`backends <../backends/index>` documentation.

