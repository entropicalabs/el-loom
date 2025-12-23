from loom.executor import EkaToQasmConverter

# interpreted_eka: InterpretationStep
converter = EkaToQasmConverter()

# Convert the Eka circuit to QASM string representation
QASM_string, quantum_reg_mapping, classical_reg_mapping = converter.convert(
    interpreted_eka
)
