@REM Loads the conda environment for this repo

@REM Open miniconda as admin
@echo off
%windir%\System32\WindowsPowerShell\v1.0\powershell.exe -ExecutionPolicy ByPass -NoExit -Command "& 'C:\ProgramData\miniconda3\shell\condabin\conda-hook.ps1' ; conda activate 'C:\ProgramData\miniconda3' ; conda activate conda-torch"
