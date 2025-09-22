import tkinter as tk
import sys
import os

# Add project root to path to solve import issues
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Victor.gui.main import VictorGUI

def main():
    """
    The main entry point for the Victor GUI.
    """
    root = tk.Tk()
    app = VictorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
