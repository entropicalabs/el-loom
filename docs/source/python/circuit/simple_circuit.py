# Create a quantum channel
from loom.eka.circuit import ChannelType, Circuit, Channel


channel = Channel(label="my_channel_1", type=ChannelType.QUANTUM)

# Create a gate that applies to the channel
gate = Circuit(name="my_gate", circuit=(), channels=[channel])

# Create a circuit that applies the gate twice in series
# When creating circuits with non-empty circuit sequence, channels are automatically inferred from the sequence.
circuit = Circuit(
    name="my_circuit",
    circuit=(gate, gate),
)

print(circuit)

# Output:
# my_circuit
# 0: my_gate
# 1: my_gate
