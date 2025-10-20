# Loom Documentation

This folder contains tools and content to build the Loom documentation.

NOTE: The documentation is accessible by opening docs/build/index.html 
within any web browser. This README contains the instructions for building the 
documentation and locally hosting the website on your device.

This project uses [Sphinx](https://www.sphinx-doc.org/en/master/) to build documentation.
The dependencies needed for building the documentation can be installed as:

```sh
poetry install --with docs
poetry install --with notebooks
```

In order to build the HTML documentation, first generate the api documentation (from within the `docs` folder):

```sh
poetry run make apidoc
```

Then build the HTML:

```sh
poetry run make html
```

In order to have a dynamic view on the HTML documentation run:

```sh
poetry run sphinx-autobuild source build/html
```

And then open the link displayed in the CLI (which should be a server running on `localhost`)
