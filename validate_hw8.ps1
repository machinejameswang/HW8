$ErrorActionPreference = "Stop"

Push-Location $PSScriptRoot
try {
    python -m compileall src app animations
    python -m pytest -q
}
finally {
    Pop-Location
}
