# interpreted_eka: InterpretationStep
qasm_string = convert_circuit_to_qasm(
    interpreted_eka.final_circuit,
    interpreted_eka.syndromes,
    interpreted_eka.detectors,
    interpreted_eka.logical_observables,
)
