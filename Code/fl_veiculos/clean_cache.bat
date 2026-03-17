@echo off
REM ############################################################################
REM Script de Limpeza de Cache Python (Windows)
REM Parte do TODO 1.6 - FASE 1 CRÍTICA
REM ############################################################################

echo ==========================================
echo Limpando cache Python (.pyc e __pycache__)
echo ==========================================
echo.

echo Removendo diretorios __pycache__...
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"

echo Removendo arquivos .pyc...
del /s /q *.pyc 2>nul

echo Removendo arquivos .pyo...
del /s /q *.pyo 2>nul

echo.
echo ==========================================
echo Cache limpo com sucesso!
echo ==========================================
echo.
echo Proximo passo:
echo   python main.py --algorithm all --strategy both
echo.
pause
