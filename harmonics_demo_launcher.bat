@echo off
rem This is a launcher for Windows. (On Linux you'll probably manage by just typing 
rem "python3 harmonics_demo.py" on the command line, possibly after activating an environment.)
rem 
rem The path needs to be adjusted based on where your Anaconda or Miniconda is installed
rem The environment name (here "base") needs to be changed to one of your Anaconda/Miniconda environments
CALL C:\Programfiles\Miniconda3\Scripts\activate.bat base
CALL python harmonics_demo.py
