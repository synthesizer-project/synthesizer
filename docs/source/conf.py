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
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',  # Core library for html generation from docstrings
    'sphinx.ext.autosummary',  # Create neat summary tables
    'sphinx.ext.viewcode',  # Add a link to the Python source code for classes, functions etc.
    'sphinx_autodoc_typehints', # Automatically document param types (less noise in class signature)
    'IPython.sphinxext.ipython_console_highlighting',
    'sphinx_gallery.gen_gallery',
]

sphinx_gallery_conf = {
     'examples_dirs': '../../examples',   # path to your example scripts
     'gallery_dirs': 'auto_examples',  # path to where to save gallery generated output
     'nested_sections': True,

     # directory where function/class granular galleries are stored
     'backreferences_dir'  : 'auto_examples/backreferences',

     # Modules for which function/class level galleries are created. In
     # this case sphinx_gallery and numpy in a tuple of strings.
     'doc_module'          : ('synthesizer'),
 
#     # Regexes to match objects to exclude from implicit backreferences.
#     # The default option is an empty set, i.e. exclude nothing.
#     # To exclude everything, use: '.*'
#     # 'exclude_implicit_doc': {r'pyplot\.show'},
}

autosummary_generate = True  # Turn on sphinx.ext.autosummary
html_show_sourcelink = False  # Remove 'view source code' from top of page (for html, not python)
autodoc_inherit_docstrings = True  # If no docstring, inherit from base class
set_type_checking_flag = True  # Enable 'expensive' imports for sphinx_autodoc_typehints

nbsphinx_allow_errors = True  # Continue through Jupyter errors

templates_path = ['templates']

# exclude_patterns = []
# exclude .py and .ipynb files in auto_examples generated by sphinx-gallery
# this is to prevent sphinx from complaining about duplicate source files
# exclude_patterns = [
#     "auto_examples/*.ipynb",
#     "auto_examples/*.py",
# ]

master_doc = 'index'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = []

html_logo = 'synth.png'

# html_theme = 'sphinx_rtd_theme'
html_theme = 'furo'
html_theme_options = {
    "sidebar_hide_name": True,
}
