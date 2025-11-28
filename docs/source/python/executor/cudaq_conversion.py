from loom.executor import EkaToCudaqConverter

converter = EkaToCudaqConverter()

# interpreted_eka: InterpretationStep
cudaq_kernel = converter.convert(interpreted_eka)
