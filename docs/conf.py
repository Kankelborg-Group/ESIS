# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.axes
import astropy.units
import astropy.time
import pandas


sys.path.insert(0, os.path.abspath('../'))

# -- Project information -----------------------------------------------------

project = 'ESIS'
copyright = '2020, Roy T. Smart, Charles C. Kankelborg, Jacob D. Parker, Nelson C. Goldsworth'
author = 'Roy T. Smart, Charles C. Kankelborg, Jacob D. Parker, Nelson C. Goldsworth'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    # "sphinx.ext.autodoc.typehints",
    'sphinx_autodoc_typehints',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.mathjax',
    'matplotlib.sphinxext.plot_directive',
    'IPython.sphinxext.ipython_console_highlighting',
    'IPython.sphinxext.ipython_directive',
    'jupyter_sphinx'
]
autosummary_generate = True  # Turn on sphinx.ext.autosummary
# autosummary_imported_members = True
# autoclass_content = 'both'
autodoc_typehints = "description"

# autosummary_filename_map = {
#     'kgpy.optics.Surface': 'kgpy.optics.Surface_cls',
# }

typehints_fully_qualified = True

graphviz_output_format = 'svg'
inheritance_graph_attrs = dict(rankdir='TB')

plot_include_source = False
plot_html_show_source_link = False
plot_formats = ['png']


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# This pattern also affects html_static_path and html_extra_path.
# directories to ignore when looking for source files.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

html_theme = 'bootstrap-astropy'
html_theme_options = {
    'logotext1': 'ESIS',  # white,  semi-bold
    'logotext2': '',  # blue, light
    'logotext3': ':docs',  # white,  light
    'astropy_project_menubar': False
}
html_sidebars = {
   '**': ['globaltoc.html'],
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# https://github.com/readthedocs/readthedocs.org/issues/2569
master_doc = 'index'

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference', None),
    'matplotlib': ('https://matplotlib.org', None),
    'astropy': ('https://docs.astropy.org/en/stable/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'kgpy': ('https://kgpy.readthedocs.io/en/latest/', None),
}

plt.Axes.__module__ = matplotlib.axes.__name__
astropy.units.Quantity.__module__ = astropy.units.__name__
astropy.time.Time.__module__ = astropy.time.__name__
pandas.DataFrame.__module__ = pandas.__name__
