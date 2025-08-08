import os
import json
import logging
from tkinter import Tk
from tkinter.filedialog import askopenfilenames

LAST_PATH_FILE = os.path.join('results', 'last_path.json')

def load_last_path(default_path):
    if os.path.exists(LAST_PATH_FILE):
        try:
            with open(LAST_PATH_FILE, 'r') as f:
                data = json.load(f)
                last_path = data.get('last_path', default_path)
                if os.path.exists(last_path):
                    return last_path
        except Exception as e:
            logging.warning(f"Kann letztes Verzeichnis nicht lesen: {e}")
    return default_path

def save_last_path(path):
    try:
        os.makedirs('results', exist_ok=True)
        with open(LAST_PATH_FILE, 'w') as f:
            json.dump({'last_path': path}, f)
    except Exception as e:
        logging.warning(f"Kann letztes Verzeichnis nicht speichern: {e}")

def select_files(default_last_path):
    try:
        root = Tk()
        root.withdraw()
        initial_dir = load_last_path(default_last_path)
        file_paths = askopenfilenames(
            title="Wähle Parquet-Dateien aus",
            initialdir=initial_dir,
            filetypes=[("Parquet files", "*.parquet")]
        )
        root.destroy()

        if file_paths:
            last_dir = os.path.dirname(file_paths[0])
            save_last_path(last_dir)
        return list(file_paths)

    except Exception as e:
        logging.error(f"Fehler im Dateidialog: {str(e)}")
        return []
