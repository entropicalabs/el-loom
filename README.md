# loom

An integrated software platform designed to handle the complexities of quantum error correction, seamlessly transforming applications into fault-tolerant, hardware-ready versions.

Disclaimer: major breaking changes to be expected in the first month from release v0.1.0.

- Website: [https://entropicalabs.com/](https://entropicalabs.com/)
- API reference: [https://loom-docs.entropicalabs.com/](https://loom-docs.entropicalabs.com/)
- Bug reports: [https://github.com/entropicalabs/el-loom/issues](https://github.com/entropicalabs/el-loom/issues)
- Support: loom-design-support@entropicalabs.com

## Introductory Example

Please visit [https://loom-docs.entropicalabs.com/loom/memory_experiment/memory_experiment/](https://loom-docs.entropicalabs.com/loom/memory_experiment/memory_experiment/)

## `loom`: Installation Guide on MacOS/Linux (Poetry)

Welcome to `loom`! This guide walks you through the setup process using Poetry to ensure a smooth installation.

### **1. Clone the Repository**

Start by cloning the loom repository from GitHub:

```sh
git clone https://github.com/entropicalabs/loom.git
cd loom
```

If you already have the repository, ensure it is up to date:

```sh
git pull origin main
```

### **2. Verify Python Installation**

`loom` supports Python versions 3.11 to 3.12. To check if you have the correct version installed, run:

```sh
python --version
```

If Python is missing or outdated, download and install the latest supported version from [python.org](https://www.python.org/downloads/).

### **3. Install Poetry (If Not Installed)**

To check if Poetry is already installed, run:

```sh
poetry --version
```

If Poetry is not available, install it using the following method:

```sh
curl -sSL https://install.python-poetry.org | python3 -
```

### **4. Add Poetry to Your PATH**

After installing Poetry, ensure that it is accessible from the command line.

```sh
export PATH="$HOME/.local/bin:$PATH"
```

To make this change permanent, add it to your shell configuration:

```sh
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc  # For Bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc   # For Zsh (default on Mac)
source ~/.bashrc  # or `source ~/.zshrc`
```

Restart your terminal and verify the installation:

```sh
poetry --version
```

### **5. Install `loom` (Standard Mode)**

To install `loom` and its dependencies in the default mode, run:

```sh
poetry install
```

This ensures all required dependencies are installed within a **Poetry-managed virtual environment**.

### **6. Install `loom` in Editable Mode (For Development)**

If you're planning to contribute to `loom`, install it in **editable mode** instead:

```sh
poetry install --with dev
```

This includes additional dependencies such as `pytest` for testing.

If you are planning to build the documentation using Sphinx, or running any of the Jupyter notebooks,
you will need to install those respective dependencies as well.

```sh
poetry install --with docs
poetry install --with notebooks
```

Some tests include notebooks and require to define a kernel within the environment.
In order to create an ipykernel, use: 

```sh
poetry run python -m ipykernel install --user --name env
```

You can set the kernel name to your convenience using the `--name "KERNEL_NAME"` flag (default is `"env"`) and its displayed name as `--display-name "DISPLAY_NAME"`.

### **7. Locate the Virtual Environment Path**

To check where Poetry has created the virtual environment, use:

```sh
poetry env info --path
```

**Example Output**:

```
/Users/username/Library/Caches/pypoetry/virtualenvs/loom-abc123-py3.10
```

### **8. Activate the Virtual Environment**

Before running any commands, activate the virtual environment.

```sh
eval $(poetry env activate)
```

### **9. Alternative: Store `.venv` in the Project Directory**

If you prefer to keep the virtual environment inside the project directory, configure Poetry accordingly:

```sh
poetry config virtualenvs.in-project true
```

**After making this change, reinstall `loom`:**

```sh
poetry install
```

Now, the virtual environment will be located at:

```
.venv/
```

Activate it as follows:

```sh
source .venv/bin/activate
```

### **10. Verify the Installation with `pytest`**

Run the test suite to ensure everything is set up correctly:

```sh
poetry run pytest
```

If all tests pass, `loom` is successfully installed and you're all set to use `loom`! If you encounter any issues, refer to the Poetry Documentation or open an issue on GitHub.

## `loom`: Installation Guide on Windows PowerShell (Poetry)

### **1. Clone the Repository**

Start by cloning the loom repository from GitHub:

```sh
git clone https://github.com/entropicalabs/loom.git
cd loom
```

If you already have the repository, ensure it is up to date:

```sh
git pull origin main
```

### **2. Verify Python Installation**

`loom` supports Python versions 3.9 to 3.12. To check if you have the correct version installed, run:

```sh
python --version
```

If Python is missing or outdated, download and install the latest supported version from [python.org](https://www.python.org/downloads/).

### **3. Install Poetry (If Not Installed)**

To check if Poetry is already installed, run:

```sh
poetry --version
```

If Poetry is not available, install it using the following method:

```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
```

### **4. Add Poetry to Your PATH**

After installing Poetry, ensure that it is accessible from the command line.

To add Poetry to PATH permanently:

```powershell
$env:Path += ";$env:APPDATA\Python\Scripts"
```

Restart your terminal and verify the installation:

```sh
poetry --version
```

### **5. Install `loom` (Standard Mode)**

To install `loom` and its dependencies in the default mode, run:

```sh
poetry install
```

This ensures all required dependencies are installed within a **Poetry-managed virtual environment**.

### **6. Install `loom` in Editable Mode (For Development)**

If you're planning to contribute to `loom`, install it in **editable mode** instead:

```sh
poetry install --with dev
```

This includes additional dependencies such as `pytest` for testing.

Some tests include notebooks and require to define a kernel within the environment.

In order to create an ipykernel, use:

```sh
poetry run python -m ipykernel install --user --name env
```

You can set the kernel name to your convenience using the `--name "KERNEL_NAME"` flag (deafult is `"env"`) and its displayed name as `--display-name "DISPLAY_NAME"`.

### **7. Locate the Virtual Environment Path**

To check where Poetry has created the virtual environment, use:

```sh
poetry env info --path
```

**Example Output**:

```
C:\Users\username\AppData\Local\pypoetry\Cache\virtualenvs\loom-abc123-py3.10
```

### **8. Activate the Virtual Environment**

Before running any commands, activate the virtual environment.

```powershell
Invoke-Expression (poetry env activate)
```

### **9. Alternative: Store `.venv` in the Project Directory**

If you prefer to keep the virtual environment inside the project directory, configure Poetry accordingly:

```sh
poetry config virtualenvs.in-project true
```

**After making this change, reinstall `loom`:**

```sh
poetry install
```

Now, the virtual environment will be located at:

```
.venv/
```

Activate it as follows:

```sh
.venv\Scripts\activate
```

### **10. Verify the Installation with `pytest`**

Run the test suite to ensure everything is set up correctly:

```sh
poetry run pytest
```

If all tests pass, `loom` is successfully installed and you're all set to use `loom`! If you encounter any issues, refer to the Poetry Documentation or open an issue on GitHub.
