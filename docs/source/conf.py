# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "loom"
copyright = "2025, Entropica Labs Pte Ltd"
author = "Entropica Labs Pte Ltd"
# release = "0.1.0" # TODO uncomment later

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "myst_nb",
]

# Paths for templates and static files
templates_path = ["_templates"]
exclude_patterns = []

# -- Options for class documentation -----------------------------------------
# Both the class’ and the __init__ method’s docstring are concatenated and inserted.
autoclass_content = "both"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_theme = "sphinx_rtd_theme"
html_logo = "_static/entropica_logo.svg"
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 5,
}
html_static_path = ["_static"]

suppress_warnings = ["toc.not_included"]

# -- Options for MyST-NB notebook execution ----------------------------------
nb_execution_mode = "cache"

# -- Options for MyST-parser -------------------------------------------------
myst_enable_extensions = ["dollarmath", "amsmath"]  # For inline and LaTeX rendering

# -- Options for MathJax3 ----------------------------------------------------
mathjax3_config = {
    "tex2jax": {
        "inlineMath": [["$", "$"], ["\\(", "\\)"]],
        "processEscapes": True,
        "ignoreClass": "document",
        "processClass": "math|output_area",
    }
}  # Required for plotly and dollarmath to work together

autodoc_default_options = {
    "exclude-members": "model_post_init",
}
