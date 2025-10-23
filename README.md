# 📦 Loom

[![PyPI version](https://img.shields.io/pypi/v/el-loom.svg)](https://pypi.org/project/el-loom/)
[![Python versions](https://img.shields.io/pypi/pyversions/el-loom.svg)](https://pypi.org/project/el-loom/)

> Loom is a Python library to create and run quantum error correction (QEC) experiments.
> It provides API that allows users to easily define custom stabilizer codes and corresponding lattice surgery operations.
> It also provides all tools required to convert any experiments to executables compatible with the most used backend (Stim, QASM, etc...).

Disclaimer: major breaking changes to be expected in the first month from release v0.1.0.

- Website: [https://entropicalabs.com/](https://entropicalabs.com/)
- API reference: [https://loom-api-docs.entropicalabs.com/](https://loom-api-docs.entropicalabs.com/)
- Documentation: [https://loom-docs.entropicalabs.com/](https://loom-docs.entropicalabs.com/)
- Bug reports: [https://github.com/entropicalabs/el-loom/issues](https://github.com/entropicalabs/el-loom/issues)
- Support: loom-design-support@entropicalabs.com

Loom is licensed under [Apache 2.0](LICENSE.md).

## ✨ Features

- ✅ Build QEC experiments with lattice surgery, surface codes and more.
- ⚡ Simulate circuits on multiple backends (Stim, QASM3, etc...)
- 🧠 Analyze stabilizers, syndromes, and logical error rates
- 🧰 Extensible API for custom experiments and backends

---

## 📦 Installation

Install the latest release from PyPI:

```bash
pip install el-loom
```

Install the development version directly from GitHub:

```bash
pip install git+https://github.com/entropicalabs/el-loom.git
```

---

## 🚀 Quick Start

Here's a simple example of a experiment designed with Eka. Have a look to the
[documentation]() for more information on the available features.

```python
from loom.eka import Eka, Lattice
from loom.eka.operations import (
    ResetAllDataQubits, 
    Merge, 
    MeasureBlockSyndromes, 
    MeasureLogicalZ
)
from loom.interpreter import interpret_eka
from loom.executor import EkaCircuitToStimConverter
from loom_rotated_surface_code.code_factory import RotatedSurfaceCode

lattice = Lattice.square_2d((15, 15))

# Create rotated surface blocks on a lattice
rsc_block_1 = RotatedSurfaceCode.create(5, 5, lattice, unique_label="rsc_block_1")
rsc_block_2 = RotatedSurfaceCode.create(
    5, 5, lattice, unique_label="rsc_block_2", position=(6, 0)
)

# Define lattice surgery operations to process
operations = [
    ResetAllDataQubits(rsc_block_1.unique_label),
    Merge([rsc_block_1.unique_label, rsc_block_2.unique_label], "rsc_block_3"),
    MeasureBlockSyndromes("rsc_block_3", n_cycles=1),
    MeasureLogicalZ("rsc_block_3"),
]

# Interpret the operations on the rotated surface code blocks
eka_experiment = Eka(
    lattice, blocks=[rsc_block_1, rsc_block_2], operations=operations
)
# This will contain the circuit, syndromes and detectors of the system resulting from the operations.
final_state = interpret_eka(eka_experiment)

# Get the Stim code ready for simulation
stim_circuit = EkaCircuitToStimConverter().convert(final_state)

```

---

## 🧑‍💻 Development Setup

To contribute to Loom, set up a development environment using **Poetry**:

```bash
# Clone the repository
git clone https://github.com/entropicalabs/el-loom.git
cd el-loom

# Install Poetry if needed
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies including dev dependencies
poetry install --with dev

# Get the command to activate the virtual environment
poetry env activate     # or run all command with "poetry run" prefix

# Run tests
pytest

# Run linting/formatting
poetry run black src tests
```

---

## 🧱 Project Structure

```
el-loom/
├── .github/workflows/                  # CI/CD workflows
├── docs/                               # Documentation
├── src/
│   ├── checkers/                       # Pylint checkers
│   ├── loom/
│   │   ├── __init__.py
│   │   ├── eka/                        # Error correction algorithms
│   │   ├── executor/                   # Circuit execution and conversion
│   │   ├── interpreter/                # Syndrome interpretation
│   │   └── visualizer/                 # Plotting and visualization
│   ├── loom_five_qubit_perfect_code/
│   │   ├── __init__.py
│   │   ├── selectors.py                    
│   │   ├── applicator/                 # Code-specific interpretation instructions
│   │   └── code_factory/               # Code-specific algorithmic instructions
│   ├── loom_repetition_code/
│   │   └── ...
│   ├── loom_rotated_surface_code/
│   │   └── ...
│   ├── loom_shor_code/
│   │   └── ...
│   └── loom_steane_code/
│       └── ...
├── tests
├── poetry.lock
├── pyproject.toml
└── README.md
 
```

---

## 🪪 License

Loom is licensed under [Apache 2.0](LICENSE.md).

---

## 📬 Contact

Created by [Entropica Labs](https://github.com/entropicalabs).
Feel free to open issues, request features, or contribute via pull requests.
