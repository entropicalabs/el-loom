Getting Started
===============

Welcome to the Loom documentation! This page will help you get started quickly.

Installation
------------

To install Loom, first clone the Loom repository from GitHub:

.. code-block:: bash

   git clone https://github.com/entropicalabs/loom-for-<your_repo_here>.git

Loom uses Poetry for dependency management and packaging. Poetry's full installation 
instructions can be found from their official documentation at 
https://python-poetry.org/docs/main/#installing-with-the-official-installer.
It is recommended that you install Poetry in a dedicated virtual environment. 

We will provide a quick summary of the steps needed to get Poetry up and running. First, 
install Poetry using the official installer, and ensure that it is accessible from the 
command line:

.. code-block:: bash

   curl -sSL https://install.python-poetry.org | python3 -
   export PATH="$HOME/.local/bin:$PATH"

To make poetry permanently accessible from the command line, add it to your shell 
configuration:

.. code-block:: bash

   echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc  # For Bash
   echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc   # For Zsh (default on Mac)
   source ~/.bashrc  # or `source ~/.zshrc`

Next, install Loom using Poetry:


.. code-block:: bash

   poetry install

This ensures all required dependencies are installed within a Poetry-managed virtual 
environment. 

.. Note::
   *Optional:* If you prefer to keep the virtual environment inside the project directory, 
   you can configure Poetry accordingly before installation:

   .. code-block:: bash

      poetry config virtualenvs.in-project true

   *Optional:* If you plan on contributing to Loom, running the documentation website, or running the Jupyter notebooks, you can add in the following flags to install additional dependencies: 

   .. code-block:: bash

      poetry install --with dev        # For development
      poetry install --with docs       # For documentation
      poetry install --with notebooks  # For Jupyter Notebooks


After installing, you can activate the virtual environment with the following command:

.. code-block:: bash

   eval $(poetry env activate)


Finally, verify the installation with Pytest.

.. code-block:: bash

   poetry run pytest

If all tests pass, Loom is successfully installed and you're all set to use 
Loom! If you encounter any issues, refer to the Poetry documentation or open an 
issue on GitHub.

Next Steps
----------

- :doc:`Discover Loom Basics <../basics/index>`
- :doc:`Explore Examples <../examples/index>`
- :doc:`Understand Loom Architecture <../eka_qec/index>`
- :doc:`Supported Backends <../backends/index>`
- :doc:`API Reference <../../apidoc/custom/index>`

