@echo off
setlocal enabledelayedexpansion

:: LaTeX PDF Build Script
:: Usage: build.bat [filename]   (default: report)

set "TEXFILE=%~1"
if "%TEXFILE%"=="" set "TEXFILE=report"

:: Strip .tex extension if provided
set "TEXFILE=%TEXFILE:.tex=%"

cd /d "%~dp0"

if not exist "%TEXFILE%.tex" (
    echo [ERROR] %TEXFILE%.tex not found in %cd%
    exit /b 1
)

echo ========================================
echo  Building %TEXFILE%.tex
echo ========================================

:: Pass 1
echo [1/2] First pass...
xelatex -interaction=nonstopmode "%TEXFILE%.tex" >nul 2>&1
if errorlevel 1 (
    echo [WARN] First pass had errors, check %TEXFILE%.log
)

:: Pass 2 (resolve cross-references)
echo [2/2] Second pass (cross-references)...
xelatex -interaction=nonstopmode "%TEXFILE%.tex" >nul 2>&1
if errorlevel 1 (
    echo [WARN] Second pass had errors, check %TEXFILE%.log
)

if exist "%TEXFILE%.pdf" (
    echo ========================================
    echo  Done! Output: %TEXFILE%.pdf
    echo ========================================
) else (
    echo [ERROR] PDF generation failed. See %TEXFILE%.log
    exit /b 1
)

endlocal
