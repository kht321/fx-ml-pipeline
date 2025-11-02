@echo off
REM FX ML Pipeline - Automatic .env Setup for Windows
REM This script creates a .env file with your current project path

echo ========================================
echo FX ML Pipeline - Environment Setup
echo ========================================
echo.

REM Get the current directory (project root)
set CURRENT_DIR=%cd%
REM Convert backslashes to forward slashes for Docker
set DOCKER_PATH=%CURRENT_DIR:\=/%

echo Detected project root: %DOCKER_PATH%
echo.

REM Check if .env already exists
if exist .env (
    echo WARNING: .env file already exists!
    set /p OVERWRITE="Do you want to overwrite it? (y/n): "
    if /i not "%OVERWRITE%"=="y" (
        echo Setup cancelled.
        exit /b 0
    )
)

REM Create .env file
echo Creating .env file...
(
echo # FX ML Pipeline Environment Variables
echo # Auto-generated on %date% at %time%
echo.
echo # Project root - use forward slashes for Docker compatibility
echo FX_ML_PIPELINE_ROOT=%DOCKER_PATH%
echo.
echo # Derived paths
echo HOST_DATA_DIR=${FX_ML_PIPELINE_ROOT}/data_clean
echo HOST_MODELS_DIR=${FX_ML_PIPELINE_ROOT}/data_clean/models
echo.
echo # OANDA API Credentials ^(replace with your actual credentials^)
echo OANDA_ACCOUNT_ID=your_account_id_here
echo OANDA_TOKEN=your_api_token_here
) > .env

echo.
echo ========================================
echo SUCCESS! .env file created
echo ========================================
echo.
echo Project root set to: %DOCKER_PATH%
echo.
echo Next steps:
echo 1. Update OANDA credentials in .env if needed
echo 2. Run: docker-compose up -d
echo.
echo For more information, see SETUP_ENVIRONMENT.md
echo.

pause
