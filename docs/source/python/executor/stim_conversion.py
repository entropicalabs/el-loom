from loom.executor import EkaCircuitToStimConverter

converter = EkaCircuitToStimConverter()

# interpreted_eka: InterpretationStep
stim_circuit = converter.convert(interpreted_eka)
