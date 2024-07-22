# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

project = "Progress-Gym"
copyright = "2024, Tianyi Qiu, Yang Zhang, Xuchuan Huang, Xinze Li"
author = "Tianyi Qiu, Yang Zhang, Xuchuan Huang, Xinze Li"
release = "1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc"]

autodoc_default_options = {
    'members': True,
    'special-members': '__init__',  # 只显示构造函数
    'private-members': False,       # 不显示私有成员
}

templates_path = ["_templates"]
exclude_patterns = []

language = "cn"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
