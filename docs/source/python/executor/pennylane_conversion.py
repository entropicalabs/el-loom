from loom.executor import convert_circuit_to_pennylane

# interpreted_eka: InterpretationStep, is_catalyst: bool
circuit_callable, qbit_register = convert_circuit_to_pennylane(
    interpreted_eka.final_circuit,
    is_catalyst,
)
