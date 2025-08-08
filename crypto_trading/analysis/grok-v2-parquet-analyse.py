import pandas as pd
import os
from datetime import datetime
import glob

def analyze_parquet_structure(file_path):
    """
    Analysiert die Struktur einer Parquet-Datei und gibt die Ergebnisse in der Konsole und als Datei aus.
    """
    print(f"\nAnalyse der Datei: {file_path}")
    
    # Parquet-Datei einlesen
    try:
        df = pd.read_parquet(file_path, engine='pyarrow')
    except Exception as e:
        print(f"Fehler beim Einlesen der Datei: {e}")
        return None
    
    # Grundlegende Informationen
    print("\n=== Grundlegende Informationen ===")
    print(f"Anzahl der Zeilen: {len(df)}")
    print(f"Anzahl der Spalten: {len(df.columns)}")
    
    # Spalten und Datentypen
    print("\n=== Spalten und Datentypen ===")
    column_info = pd.DataFrame({
        'Spalte': df.columns,
        'Datentyp': [str(dtype) for dtype in df.dtypes]
    })
    print(column_info.to_string(index=False))
    
    # Fehlende Werte
    print("\n=== Fehlende Werte ===")
    missing_values = pd.DataFrame({
        'Spalte': df.columns,
        'Anzahl fehlender Werte': df.isna().sum(),
        'Prozent fehlender Werte': (df.isna().sum() / len(df) * 100).round(2)
    })
    print(missing_values.to_string(index=False))
    
    # Zeitliche Analyse (falls timestamp-Spalte vorhanden)
    timestamp_cols = df.select_dtypes(include=['datetime64[ns]']).columns
    if timestamp_cols.any():
        print("\n=== Zeitliche Analyse ===")
        for col in timestamp_cols:
            print(f"Zeitspalte: {col}")
            print(f"  Startdatum: {df[col].min()}")
            print(f"  Enddatum: {df[col].max()}")
            print(f"  Zeitspanne: {df[col].max() - df[col].min()}")
            
            # Zeitauflösung prüfen
            time_diffs = df[col].diff().dropna()
            if not time_diffs.empty:
                median_diff = time_diffs.median()
                print(f"  Mediandifferenz zwischen Zeitstempeln: {median_diff}")
    
    # Statistische Analyse für numerische Spalten
    print("\n=== Statistische Analyse (numerische Spalten) ===")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if numeric_cols.any():
        stats = df[numeric_cols].describe().T.round(4)
        print(stats[['count', 'mean', 'std', 'min', '50%', 'max']].to_string())
    else:
        print("Keine numerischen Spalten gefunden.")
    
    return {
        'columns': column_info,
        'missing_values': missing_values,
        'stats': stats if numeric_cols.any() else None
    }

def save_analysis_results(analysis_results, file_paths):
    """
    Speichert die Analyseergebnisse als CSV und Excel.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Spalteninformationen
    all_columns = []
    all_missing = []
    all_stats = []
    
    for file_path, result in zip(file_paths, analysis_results):
        if result is None:
            continue
        result['columns']['Datei'] = os.path.basename(file_path)
        result['missing_values']['Datei'] = os.path.basename(file_path)
        all_columns.append(result['columns'])
        all_missing.append(result['missing_values'])
        if result['stats'] is not None:
            result['stats']['Datei'] = os.path.basename(file_path)
            all_stats.append(result['stats'])
    
    # Ergebnisse zusammenfassen
    if all_columns:
        columns_df = pd.concat(all_columns, ignore_index=True)
        missing_df = pd.concat(all_missing, ignore_index=True)
        
        # Als CSV speichern
        columns_df.to_csv(f"parquet_columns_{timestamp}.csv", index=False)
        missing_df.to_csv(f"parquet_missing_values_{timestamp}.csv", index=False)
        print(f"\nSpalteninformationen gespeichert als: parquet_columns_{timestamp}.csv")
        print(f"Fehlende Werte gespeichert als: parquet_missing_values_{timestamp}.csv")
        
        # Als Excel speichern
        with pd.ExcelWriter(f"parquet_analysis_{timestamp}.xlsx", engine='openpyxl') as writer:
            columns_df.to_excel(writer, sheet_name='Spalten', index=False)
            missing_df.to_excel(writer, sheet_name='Fehlende_Werte', index=False)
            if all_stats:
                stats_df = pd.concat(all_stats, ignore_index=True)
                stats_df.to_excel(writer, sheet_name='Statistiken', index=True)
        print(f"Analyseergebnisse als Excel gespeichert: parquet_analysis_{timestamp}.xlsx")
    else:
        print("Keine Ergebnisse zum Speichern.")

def main():
    # Dateipfad-Eingabe
    print("Gib die Pfade der Parquet-Dateien ein (mehrere Pfade durch Komma trennen, oder '*' für alle Dateien im Verzeichnis):")
    file_input = input().strip()
    
    if '*' in file_input:
        file_paths = glob.glob(file_input)
    else:
        file_paths = [path.strip() for path in file_input.split(',')]
    
    analysis_results = []
    
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Datei nicht gefunden: {file_path}")
            continue
        
        result = analyze_parquet_structure(file_path)
        if result:
            analysis_results.append(result)
    
    # Ergebnisse speichern
    save_analysis_results(analysis_results, file_paths)

if __name__ == "__main__":
    main()