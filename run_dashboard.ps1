$ErrorActionPreference = "Stop"

Push-Location $PSScriptRoot
try {
    python -m streamlit run app/streamlit_app.py --server.port 8501
}
finally {
    Pop-Location
}
