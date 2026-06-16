import os
import runpy
import sys


PROJECT_DIR = os.path.join(os.path.dirname(__file__), "project")

os.chdir(PROJECT_DIR)
sys.path.insert(0, PROJECT_DIR)

runpy.run_path(os.path.join(PROJECT_DIR, "app.py"), run_name="__main__")
