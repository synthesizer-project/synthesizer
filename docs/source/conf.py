# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'synthesizer'
copyright = '2023, Chris Lovell, Stephen Wilkins, Aswin Vijayan, Will Roper'
author = 'Chris Lovell, Stephen Wilkins, Aswin Vijayan, Will Roper'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

import os
import sys
sys.path.insert(0, os.path.abspath('../../synthesizer'))  # Source code dir relative to this file
sys.path.insert(0, os.path.abspath('../../'))  # Source code dir relative to this file
sys.path.insert(0, os.path.abspath("."))

extensions = [
	"nbsphinx",
    'sphinx.ext.autodoc',  # Core library for html generation from docstrings
    'sphinx.ext.autosummary',  # Create neat summary tables
]

autosummary_generate = True  # Turn on sphinx.ext.autosummary

templates_path = ['templates']
exclude_patterns = []

master_doc = 'index'


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'sphinx_rtd_theme'
html_theme = 'furo'
html_static_path = ['_static']
