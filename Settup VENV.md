# Setting Up the Bomberman Virtual Environment

## Install Python 3.6.7
3.6 is the latest version of Python which is currently compatible with
tensorflow, the leading AI library, so we will stay with standard and use
it for our work.

Download and install Python from the following link
https://www.python.org/downloads/release/python-367/

You do NOT need to include path variables if not desired!

## Creating the VENV
cd C:\ ...Bomberman\ 

To install the correct version into the venv, use the direct path to the
python install from the last step to make the python call to. This allows
us to circumvent overlapping environment variable and path issues.

C:\ ...\Python3\python3.exe -m venv bomberman_env

## Activating the VENV (Linux)
source bomberman_env/bin/activate

## Activating VENV (Windows)
bomberman_env\Scripts\activate

## Installing Dependencies
pip install -r requirements.txt

## Saving New Dependencies
pip freeze > requirements.txt

## Deactivating the VENV
deactivate

## Using VENV in PyCharm
File->Settings->Project Interpreter

Click on Gear Symbol in the top right next to Project Interpreter

Select Add from context menu

Make sure Virtualenv Environment is selected on the left side

Check the box next to Exsisting Environment

Select the interpreter path "C:\ ...\Bomberman\bomberman_env\Scripts\python.exe"

Click OK

Click Apply
