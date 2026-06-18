"""Compatibility entrypoint for Streamlit Cloud.

The preferred entrypoint is `streamlit_app.py`. This file exists so an app
configured with `app.py` still renders the same complete HW8 landing page.
"""

from streamlit_app import *  # noqa: F401,F403
