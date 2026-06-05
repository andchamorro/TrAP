# Sphinx configuration for TrAP documentation.
# Build: cd docs && make html
# Serve: python -m http.server -d _build/html

import sys
from pathlib import Path

# Make the trap package importable without installing
sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Project metadata
# ---------------------------------------------------------------------------
project = "TrAP"
author = "Andres D. Chamorro-Parejo"
copyright = "2024, Andres D. Chamorro-Parejo"
release = "0.0.1"

# ---------------------------------------------------------------------------
# Extensions
# ---------------------------------------------------------------------------
extensions = [
    # Core autodoc
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    # Google / NumPy docstring support
    "sphinx.ext.napoleon",
    # Hyperlinks to source code on GitHub
    "sphinx.ext.viewcode",
    # Cross-links to NumPy, PyTorch, HuggingFace Transformers docs
    "sphinx.ext.intersphinx",
    # Inline type-annotation rendering
    "sphinx_autodoc_typehints",
    # .md source files (used for guides/)
    "myst_parser",
    # Copy button on code blocks
    "sphinx_copybutton",
]

# ---------------------------------------------------------------------------
# autodoc
# ---------------------------------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "special-members": "__init__",
    "exclude-members": "__weakref__, __dict__, __module__",
}
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
autodoc_member_order = "bysource"

autosummary_generate = True
autosummary_imported_members = False

# ---------------------------------------------------------------------------
# Napoleon (Google-style docstrings)
# ---------------------------------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True

# ---------------------------------------------------------------------------
# MyST (Markdown)
# ---------------------------------------------------------------------------
myst_enable_extensions = [
    "colon_fence",   # :::note syntax
    "deflist",
    "tasklist",
    "fieldlist",
]
myst_heading_anchors = 3

# ---------------------------------------------------------------------------
# Intersphinx — external doc cross-links
# ---------------------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "transformers": ("https://huggingface.co/docs/transformers/main/en", None),
    "datasets": ("https://huggingface.co/docs/datasets/main/en", None),
}

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
html_theme = "furo"
html_title = "TrAP"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "source_repository": "https://github.com/andchamorro/TrAP",
    "source_branch": "main",
    "source_directory": "docs/",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/andchamorro/TrAP",
            "html": "",
            "class": "fa-brands fa-github fa-2x",
        },
    ],
}

# ---------------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------------
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
