# run_all_experiments.py

import os
import subprocess
import sys

def run_script(script_path):
    """
    Executes a Python script as a subprocess.

    Parameters:
    - script_path (str): Path to the Python script to execute.

    Returns:
    - bool: True if the script executed successfully, False otherwise.
    """
    if not os.path.exists(script_path):
        print(f"[ERROR] Script {script_path} does not exist. Skipping.")
        return False

    print(f"\n[INFO] Starting execution of {script_path} ...")
    try:
        # Execute the script using the same Python interpreter
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,               # Raises CalledProcessError if the script exits with a non-zero status
            capture_output=True,      # Captures stdout and stderr
            text=True                 # Returns output as strings instead of bytes
        )
        print(f"[SUCCESS] Execution of {script_path} completed successfully.")
        print(f"[OUTPUT]\n{result.stdout}")
        if result.stderr:
            print(f"[WARNING] {script_path} produced the following warnings/errors:\n{result.stderr}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[FAILURE] Execution of {script_path} failed with return code {e.returncode}.")
        print(f"[OUTPUT]\n{e.stdout}")
        print(f"[ERRORS]\n{e.stderr}")
        return False
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred while executing {script_path}: {e}")
        return False

def main():
    """
    Main function to run all experimental scripts sequentially.
    """
    # List of dataset-specific scripts to execute
    scripts = [
        'bank32NH.py',
        'concrete.py',
        'delta_elevators.py',
        'house_16.py',
        'housing.py',
        'insurance.py',
        'movies.py'
    ]

    # Directory where this master script resides
    master_dir = os.path.dirname(os.path.abspath(__file__))

    # Iterate over each script and execute it
    for script in scripts:
        script_path = os.path.join(master_dir, script)
        run_script(script_path)

    print("\n[INFO] All experiments have been processed.")

if __name__ == '__main__':
    main()
