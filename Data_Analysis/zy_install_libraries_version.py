# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 10:42:23 2023
with help from ChatGPT
python = 3.12.10

1. Open your shell terminal with aa acount
2. cd /path/to/zy_install_libraries_version.py
3. python3 zy_install_libraries_version.py


@author: Zengyou Ye
"""

import importlib.util
import subprocess
import sys, os

def install_libraries(library_dict):
    for library_name, library_version in library_dict.items():
        spec = importlib.util.find_spec(library_name)
        if spec is None:
            print(f"The library '{library_name}' is not installed. Attempting to install...")

            try:
                # Try installing with Conda as an administrator
                if library_version:
                    conda_install_command = f"conda install -y {library_name}={library_version}" #base is environment
                else:
                    conda_install_command = f"conda install -y {library_name}"
                
                conda_install_result = subprocess.call(conda_install_command, shell=True)

                if conda_install_result != 0:
                    # If Conda installation fails, try pip as an administrator
                    if library_version:
                        pip_install_command = f"pip install {library_name}=={library_version}"
                    else:
                        pip_install_command = f"pip install {library_name}"
                        
                    pip_install_result = subprocess.call(pip_install_command, shell=True)

                    if pip_install_result != 0:
                        print(f"Failed to install '{library_name}' version {library_version} using Conda and pip. Please install it manually as an administrator.")
                        sys.exit(1)
                else:
                    print(f"'{library_name}' version {library_version} installed successfully with Conda as an administrator.")
            except Exception as e:
                print(f"Error occurred while installing '{library_name}': {str(e)}")
                sys.exit(1)
        else:
            print(f"The library '{library_name}' is already installed.")

def conda_env_exists(name):
    try:
        result = subprocess.run(
            ["conda", "env", "list"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return any(line.split()[0] == name for line in result.stdout.splitlines() if line.strip())
    except Exception as e:
        print(f"Error checking environment: {e}")
        return False

def create_conda_env(name, python_version="3.12.10"):
    try:
        subprocess.run(["conda", "create", "-n", name, f"python={python_version}", "-y"], check=True)
        print(f"Environment '{name}' created successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to create environment '{name}': {e}")
        
def activate_conda_env(env_name):
    """
    Activates the specified conda environment.
    """
    try:
        conda_bat = os.path.expandvars(r"%UserProfile%\anaconda30\Scripts\activate.bat")
        command = f'call "{conda_bat}" {env_name} && echo Activated {env_name}'
        subprocess.run(command, shell=True)
    except:
        conda_bat = os.path.expandvars(r"%UserProfile%\anaconda3\Scripts\activate.bat")
        command = f'call "{conda_bat}" {env_name} && echo Activated {env_name}'
        subprocess.run(command, shell=True)    

# Dictionary of libraries and their versions to install
# libraries_to_install = {
#     'opencv': '4.6.0', #4.11.0
#     'pandas': '2.2.3',
#     'numpy':'1.26.4', #2.2.5
#     'PyExcelerate': '0.12.0',
#     'collections': '',
#     'pillow':'9.4.0' #11.1.0
# }
if __name__ == "__main__":
    env_name = "Auto_TST"
    
    if conda_env_exists(env_name):
        print(f"Environment '{env_name}' already exists.")
    else:
        print(f"Environment '{env_name}' not found. Creating...")
        create_conda_env(env_name)
        
    # activate_conda_env(env_name)
    
    libraries_to_install = {
        'ipykernel':'6.29.5',
        # 'opencv': '',
        'openpyxl':'3.1.5',
        'pandas': '2.2.3',
        'numpy':'2.2.5',
        'PyExcelerate': '0.12.0',
        'pillow':'11.1.0',
        'spyder-kernels':'3.0.3',
        # 'spyder-base':'6.0.5',
        'matplotlib':'',
        'scipy':'',
        'nptdms':''
    }# 04/22/2025
    #python: 3.12.10
    # Call the function to check and install the libraries
    install_libraries(libraries_to_install)
