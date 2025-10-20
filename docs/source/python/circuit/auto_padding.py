# This can be done automatically:
valid_circuit = Circuit(
    name="valid_circuit",
    circuit=Circuit.construct_padded_circuit_time_sequence(  # Automatically pads the circuit
        (
            (long_subcircuit,),
            (
                Circuit(name="gate_1", channels=[chan1]),
                Circuit(name="gate_2", channels=[chan2]),
            ),
        ),
    ),
)

print(valid_circuit)
# Output:
# valid_circuit
# 0: long_subcircuit
# 1:
# 2: gate_1 gate_2
