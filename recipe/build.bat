@echo off

REM TrAP conda package build script for Windows
REM This script is called during the conda build process on Windows

echo Building TrAP for Windows...

REM Install the package using pip
"%PYTHON%" -m pip install . ^
    --no-deps ^
    --no-build-isolation ^
    --ignore-installed ^
    --no-cache-dir ^
    -vvv

if errorlevel 1 exit 1

REM Verify installation
echo Verifying trap installation...
"%PYTHON%" -c "import trap; print('TrAP imported successfully')"
if errorlevel 1 exit 1

echo Build completed successfully!
