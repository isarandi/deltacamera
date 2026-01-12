"""Sphinx configuration for deltacamera documentation."""

import importlib
import os
import re
import sys

import setuptools_scm
import toml

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

# Read project info from pyproject.toml
pyproject_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pyproject.toml"))

with open(pyproject_path) as f:
    data = toml.load(f)

project_info = data["project"]
project_slug = project_info["name"].replace(" ", "-").lower()
tool_urls = project_info.get("urls", {})

repo_url = tool_urls.get("Repository", "")
author_url = tool_urls.get("Author", "")

# Extract GitHub username from repo URL
github_match = re.match(r"https://github\.com/([^/]+)/?", repo_url)
github_username = github_match[1] if github_match else ""

project = project_info["name"]
release = setuptools_scm.get_version("..")
version = ".".join(release.split(".")[:2])
main_module_name = project_slug.replace("-", "_")

# -- Project information -----------------------------------------------------

author = project_info["authors"][0]["name"]
copyright = "2024"

# -- General configuration ---------------------------------------------------

add_module_names = False
python_use_unqualified_type_names = True

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinxcontrib.bibtex",
    "autoapi.extension",
    "sphinx_markdown_builder",
    "sphinx.ext.inheritance_diagram",
]

bibtex_bibfiles = ["abbrev_long.bib", "references.bib"]
bibtex_footbibliography_header = ".. rubric:: References"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- HTML output -------------------------------------------------------------

html_title = project
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "show_toc_level": 3,
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "primary_sidebar_end": [],
    "icon_links": [
        {
            "name": "GitHub",
            "url": repo_url,
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        }
    ],
    "navigation_with_keys": False,
}
html_static_path = ["_static"]
html_css_files = ["styles/my_theme.css"]

html_context = {
    "author_url": author_url,
    "author": author,
}

# -- AutoAPI configuration ---------------------------------------------------

autoapi_root = "api"
autoapi_member_order = "bysource"
autodoc_typehints = "description"
autoapi_own_page_level = "class"
autoapi_type = "python"

autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "undoc-members": False,
    "exclude-members": "__init__, __weakref__, __repr__, __str__",
}

autoapi_options = ["members", "show-inheritance", "special-members", "show-module-summary"]
autoapi_add_toctree_entry = True
autoapi_dirs = ["../src"]
autoapi_template_dir = "_templates/autoapi"

autodoc_member_order = "bysource"
autoclass_content = "class"

autosummary_generate = True
autosummary_imported_members = False

toc_object_entries_show_parents = "hide"
python_display_short_literal_types = True


# -- Skip undocumented members -----------------------------------------------


def autodoc_skip_member(app, what, name, obj, skip, options):
    """Skip members (functions, classes, modules) without docstrings."""
    # Check if the object has a docstring
    if not getattr(obj, "docstring", None):
        return True
    elif what in ("class", "function", "attribute"):
        # Check if the module of the class has a docstring
        module_name = ".".join(name.split(".")[:-1])
        try:
            module = importlib.import_module(module_name)
            return not getattr(module, "__doc__", None)
        except ModuleNotFoundError:
            return None
    return skip


def setup(app):
    """Sphinx setup hook."""
    app.connect("autoapi-skip-member", autodoc_skip_member)
    app.connect("autodoc-skip-member", autodoc_skip_member)
