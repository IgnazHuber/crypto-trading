import pandas as pd
import json
import os
import datetime

# ... weitere Importe, z.B. relative Imports deines Packages
from .data_loader import select_files
from .backtest import run_backtests_parallel
from .export import export_comprehensive_results


def main_complete():
    os.makedirs('results', exist_ok=True)
    last_path = os.path.join(os.getcwd(), 'data', 'raw')

    # Letztes Verzeichnis laden
    try:
        with open(os.path.join('results', 'last_path.json'), 'r') as f:
            last_path = json.load(f).get('last_path', last_path)
    except Exception:
        pass

    file_paths = select_files(last_path)
    if not file_paths:
        print("❌ Keine Dateien ausgewählt")
        return

    initial_capital = 10000
    all_trades, all_equity, all_metrics = run_backtests_parallel(file_paths, initial_capital=initial_capital)
    if not all_trades:
        print("❌ Keine Trades generiert")
        return

    total_initial = initial_capital * len(file_paths)
    total_final = sum(m['final_capital'] if 'final_capital' in m else initial_capital + m.get('total_profit', 0) for m in all_metrics)
    total_profit = total_final - total_initial
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    results = export_comprehensive_results(
        global_trades=pd.concat(all_trades, ignore_index=True),
        global_equity=pd.concat(all_equity, ignore_index=True),
        all_metrics=all_metrics,
        timestamp=timestamp
    )

    print("\n" + "=" * 80)
    print("💰 **FINALE PORTFOLIO-ERGEBNISSE**")
    print("=" * 80)
    print(f"📊 Anzahl Assets: {len(file_paths)}")
    print(f"💰 Startkapital pro Asset: {initial_capital:,.2f}€")
    print(f"💰 Gesamt-Startkapital: {total_initial:,.2f}€")
    print(f"💰 Gesamt-Endkapital: {total_final:,.2f}€")
    print(f"💰 Gesamt-Gewinn/Verlust: {total_profit:,.2f}€")
    print(f"📈 Gesamt-Rendite: {(total_profit / total_initial) * 100:.2f}%")
    print("\n📈 **Asset-Details:**")
    for metrics in all_metrics:
        print(f"   {metrics.get('symbol','')} : "
              f"Start → Ende: {initial_capital:,.2f}€ → {metrics.get('final_capital',0):,.2f}€ "
              f"({((metrics.get('final_capital',0) - initial_capital) / initial_capital)*100:.2f}%) "
              f"- Trades: {metrics.get('total_trades',0)}")
    print("\n📁 **Exportierte Dateien:")
    print(f"   📄 CSV: {results['csv']}")
    print(f"   📊 Excel: {results['excel']}")
    print(f"   🌐 HTML Dashboard: {results['html']}")
    print(f"   🌐 Marker Chart: {results['html_trades']}")
    print("=" * 80)

if __name__ == "__main__":
    main_complete()
