.. _interpretation:

Interpretation
==============

Once we have created our logical algorithm in the form of an :class:`~loom.eka.eka.Eka` object, we can interpret it into a :class:`~loom.eka.circuit.Circuit` and other tools required for error correction.

The machinery behind interpretation can be summed up with two constructs: the function :meth:`~loom.interpreter.interpreter.interpret_eka` and the dataclass :class:`~loom.interpreter.interpretation_step.InterpretationStep`. The latter is an accumulator object that will contain all variables we need to keep track of when constructing the physical circuit and the decoding pipeline. The former is a wrapper that interprets the entirety of an :class:`~loom.eka.eka.Eka` object and translates it to an :class:`~loom.interpreter.interpretation_step.InterpretationStep`.

Usage
^^^^^

Since we make use of the wrapper :meth:`~loom.interpreter.interpreter.interpret_eka`, the workflow is very simple. One can simply pass the logical circuit as an :class:`~loom.eka.eka.Eka` object to the interpretation method and obtain the resulting object:

.. code-block:: python

    from loom.interpreter import interpret_eka

    interpreted_eka = interpret_eka(my_eka)

The resulting object contains the physical circuit in :py:attr:`~loom.interpreter.interpretation_step.InterpretationStep.final_circuit`

One may also use the keyword :code:`debug_mode` to enable or disable certain functionalities of interpretation for debug purposes.
Currently, disabling :code:`debug_mode` disables the validation of intermediate steps. We strongly recommend to use :code:`debug_mode = True` when developing.

How does interpretation work?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Internally, the wrapper :meth:`~loom.interpreter.interpreter.interpret_eka` iterates through all time slices of :code:`Eka.operations` and interprets every operation in a sequential manner. All relevant variables are stored within an :class:`~loom.interpreter.interpretation_step.InterpretationStep` object that accumulates all the changes.

It is important to note that certain processes like :class:`~interpreter.detector.Detector` creation are dependent on different fields of :class:`~loom.interpreter.interpretation_step.InterpretationStep` and one should be very careful when writing down their own interpretation functions. See :ref:`interpretation_step` for more details.

Automatic dispatch
------------------

The interpreter module is built to be agnostic to the type of code targeted. The information on the code type we are using is contained with the :class:`~loom.eka.block.Block` objects themselves. When calling :meth:`~loom.interpreter.interpreter.interpret_eka`, the type of code will be used to call the right applicator such that a specific operation is interpreted as a valid operation on the chosen code. 
For example, a reset operation would be interpreted differently for a repetition code and a Steane code, this is taken care of by the applicator structure.
Other operations are common between different types of code and would be resolved in the same manner, this is the case of the operation responsible for measuring syndromes :class:`~loom.eka.operations.code_operation.MeasureBlockSyndromes`.

Interpreting operations
-----------------------

As mentioned previously, operations are read through sequentially, in the same order as they appear in the timeslice structure. The iteration in :meth:`~loom.interpreter.interpreter.interpret_eka` is similar to the following:

.. code-block:: python

    for timeslice in my_eka.operations:
        for operation in timeslice:
            int_step = interpret_operation(int_step, operation)
