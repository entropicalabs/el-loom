from loom.eka.circuit import Channel, ChannelType

# Create a quantum channel, with a given label. Id is generated automatically.
chan = Channel(label="my_channel", type=ChannelType.QUANTUM)

# Default constructor is supported, it creates a quantum channel with a default label.
chan_default = Channel()
