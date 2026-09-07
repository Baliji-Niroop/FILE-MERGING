@echo off
setlocal enabledelayedexpansion

echo ==================================================
echo          Exam Reference Merger  v2.0
echo ==================================================
echo.

set INPUT_DIR=input
set OUTPUT_DIR=output

:: Create input folder if it does not exist
if not exist "%INPUT_DIR%" (
    echo [INFO] Creating "%INPUT_DIR%\" folder...
    mkdir "%INPUT_DIR%"
    echo.
    echo  Drop your PDF, PPTX, and DOCX files into the "%INPUT_DIR%\" folder,
    echo  then double-click run_merger.bat again to merge them.
    echo.
    goto end
)

:: Check if input folder is empty
dir /b /a "%INPUT_DIR%" | findstr . >nul
if errorlevel 1 (
    echo [WARNING] The "%INPUT_DIR%\" folder is empty!
    echo  Please copy your PDF, PPTX, and DOCX files into "%INPUT_DIR%\" first.
    echo.
    goto end
)

:: Create output folder
if not exist "%OUTPUT_DIR%" (
    mkdir "%OUTPUT_DIR%"
)

echo [RUNNING] Merging exam materials in "%INPUT_DIR%\"...
echo           Output will be written to "%OUTPUT_DIR%\master_reference.md"
echo.

python exam_merger/main.py "%INPUT_DIR%" ^
    --output "%OUTPUT_DIR%\master_reference.md" ^
    --verbose ^
    --similarity-threshold 0.65 ^
    --max-workers 4

if errorlevel 1 (
    echo.
    echo [ERROR] Merger finished with errors. Check the output above.
) else (
    echo.
    echo ==================================================
    echo [SUCCESS] Your merged file is ready:
    echo   %OUTPUT_DIR%\master_reference.md
    echo ==================================================
)

echo.

:end
pause
