# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
import sphinx_gallery

import os


sys.path.insert(0, os.path.abspath('../../'))
#sys.path.insert(0, os.path.abspath('../../pipeline'))
#sys.path.insert(0, os.path.abspath('../../pipeline/code/dintegrate'))
#sys.path.insert(0, os.path.abspath('../../pipeline/code/graph_walk'))
#sys.path.insert(0, os.path.abspath('../../pipeline/code/dnn'))
#sys.path.insert(0, os.path.abspath('../../transbrain'))

project = 'TransBrain'
copyright = '2025, TransBrain'
author = 'TransBrain'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_theme_options = {
    'collapse_navigation': False,
    'navigation_depth': 4,
    'titles_only': False,
}

extensions = [
    'sphinx.ext.todo',
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages', 
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    "nbsphinx",
    'sphinx_autodoc_typehints'
]

autodoc_default_options = {
    'members': True,         
    'special-members': '__init__',
    'private-members': True,  
    'show-inheritance': True
}



nbsphinx_execute = 'never'
autodoc_member_order = 'bysource'  
autodoc_default_flags = ['members']  

html_logo = "_static/figures/logo_final.png"
pygments_style = 'sphinx'

from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).parent.parent / 'tutorials' / 'notebooks'
nbsphinx_prolog = f"""
{{% set notebook_path = '{NOTEBOOKS_DIR}' %}}
"""
