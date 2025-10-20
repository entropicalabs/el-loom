from loom.eka.circuit import Channel, ChannelType, Circuit

chan1 = Channel(label="my_channel_1", type=ChannelType.QUANTUM)
chan2 = Channel(label="my_channel_2", type=ChannelType.QUANTUM)

# Two gates in series, each applied to a different channel.
long_subcircuit = Circuit(
    name="long_subcircuit",
    circuit=(
        Circuit(name="gate_1", channels=[chan1]),
        Circuit(name="gate_2", channels=[chan2]),
    ),
)

print(long_subcircuit)
# Output:
# long_subcircuit
# 0: gate_1
# 1: gate_2


# failing_circuit = Circuit(
#     name="failing_circuit",
#     circuit=[
#         [
#             long_subcircuit, # 2 ticks long
#         ],
#         [
#             Circuit(name="gate_1", channels=[chan1]),
#             Circuit(name="gate_2", channels=[chan2]),
#         ],
#     ],
# )

# This fails because it tries to do:
# tick 0: long_subcircuit tick 0: [gate_1, chan1]
# tick 1: [gate_1, gate_2] AND (long_subcircuit tick 1: [gate_2, chan2]) => 2 gates applied to chan2 at the same time

# We need to pad with empty tick:
valid_circuit = Circuit(
    name="valid_circuit",
    circuit=(
        (long_subcircuit,),  # 2 ticks long
        (),  # empty tick
        (
            Circuit(name="gate_1", channels=[chan1]),
            Circuit(name="gate_2", channels=[chan2]),
        ),
    ),
)

print(valid_circuit)

# Output:
# valid_circuit
# 0: long_subcircuit
# 1:
# 2: gate_1 gate_2
