@echo off

set ENV_NAME=thesis_cpu
set ENV_FILE=environment_cpu.yml
set REQ_FILE=requirements_cpu.txt

echo Checking conda installation...
where conda >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo ERROR: conda not found.
    exit /b 1
)

call conda activate %ENV_NAME% >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Creating environment...
    call conda env create -f %ENV_FILE%
)

call conda activate %ENV_NAME%

echo Installing requirements...
pip install -r %REQ_FILE%

echo Running program...
python __main__.py

echo Done.
pause
