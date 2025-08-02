"""
Modul: utils.py

Allgemeine Helfer für das Batch-/Analyse-Framework:
- Dateiauswahl, Parquet-Import, ParamGrid-Builder, Fortschrittsbalken (tqdm),
- Zeit-/Index-Handling, Logging, Parallelisierung.

Funktionen:
- list_parquet_files(data_path)
- select_parquet_files(files, default_files=None)
- load_parquet_file(filename)
- build_param_grid(GRID, FAST_ANALYSIS)
- progress_bar(iterator, total=None, desc=None)
- ensure_datetime_index(df)
- unique_run_dir(base_dir)
- logger(msg, logfile=None)

Author: ChatGPT Research, 2025
"""

import os
import pandas as pd
from itertools import product
from tqdm import tqdm
import datetime
import uuid

def list_parquet_files(data_path):
    files = [f for f in os.listdir(data_path) if f.endswith('.parquet')]
    return files

def select_parquet_files(files, default_files=None):
    print("Verfügbare Parquet-Dateien:")
    for idx, fname in enumerate(files):
        print(f"{idx+1:2d}. {fname}")
    sel = input(f"Mehrere Dateien im Batch? (z.B. 1,2,5 oder Enter für Defaults ({', '.join(default_files)})): ") if default_files else ""
    if sel.strip():
        selection = [int(i.strip())-1 for i in sel.split(',') if i.strip().isdigit()]
        parquet_files = [files[i] for i in selection if 0 <= i < len(files)]
    elif default_files:
        parquet_files = list(default_files)
    else:
        parquet_files = [files[0]]
    return parquet_files

def load_parquet_file(filename, data_path=""):
    path = os.path.join(data_path, filename) if data_path else filename
    df = pd.read_parquet(path)
    df = ensure_datetime_index(df)
    return df

def ensure_datetime_index(df):
    # Macht den Index zu einem DatetimeIndex, ggf. aus Spalte
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
            df.set_index('timestamp', inplace=True)
        else:
            df.index = pd.to_datetime(df.index, errors='coerce')
    return df

def build_param_grid(GRID, FAST_ANALYSIS):
    if FAST_ANALYSIS:
        grid = [tuple([vals[0] for vals in GRID.values()])]
        print("FAST-Flag aktiviert: Nur 1 Paramset pro Asset!")
    else:
        grid = list(product(*GRID.values()))
    param_names = list(GRID.keys())
    return grid, param_names

def progress_bar(iterator, total=None, desc=None):
    return tqdm(iterator, total=total, desc=desc, ncols=80)

def unique_run_dir(base_dir="./runs"):
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{now}_{uuid.uuid4().hex[:6]}"
    out_dir = os.path.join(base_dir, run_id)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def logger(msg, logfile=None):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out = f"[{ts}] {msg}"
    print(out)
    if logfile:
        with open(logfile, "a") as f:
            f.write(out + "\n")

# Optionaler Self-Test
if __name__ == "__main__":
    print("utils.py – Self-Test – bitte als Helper importieren!")
