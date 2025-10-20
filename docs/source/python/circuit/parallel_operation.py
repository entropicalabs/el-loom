c1 = Channel(label="my_channel_1", type=ChannelType.QUANTUM)
c2 = Channel(label="my_channel_2", type=ChannelType.QUANTUM)

gate_on_1 = Circuit(name="g1", channels=[c1])
gate_on_2 = Circuit(name="g2", channels=[c2])
# Create multiple channel gate
gate_on_1_and_2 = Circuit(name="g12", channels=[c1, c2])

# All gates applied in series
s_circuit = Circuit(
    name="s_circuit",
    circuit=(gate_on_1, gate_on_2, gate_on_1_and_2),
)

print(s_circuit)

# Output:
# s_circuit
# 0: g1
# 1: g2
# 2: g12


# Gate 1 and 2 applied in parallel
p_circuit = Circuit(
    name="p_circuit",
    circuit=(
        (gate_on_1, gate_on_2),  # parallel gates
        (gate_on_1_and_2,),  # has to be wrapped in a tuple
    ),
)

print(p_circuit)

# Output:
# p_circuit
# 0: g1 g2
# 1: g12
