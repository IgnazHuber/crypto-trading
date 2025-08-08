import pandas as pd
from .visualization import (
    create_advanced_dashboard,
    create_trade_marker_chart,
    create_detailed_dashboard
)
from .metrics import trade_cluster_summary

def create_performance_heatmap(trades_df: pd.DataFrame):
    """Erstellt eine Heatmap für Performance (Profit) nach Wochentag und Stunde."""
    if trades_df.empty:
        return pd.DataFrame()
    trades_df['weekday_name'] = trades_df['entry_time'].dt.day_name()
    heatmap = trades_df.pivot_table(
        values='profit',
        index='weekday_name',
        columns='hour',
        aggfunc='sum'
    ).fillna(0)
    return heatmap

def export_comprehensive_results(global_trades: pd.DataFrame, global_equity: pd.DataFrame, all_metrics: list, timestamp: str):
    csv_file = f"results/complete_trades_{timestamp}.csv"
    if not global_trades.empty:
        global_trades.to_csv(csv_file, index=False)

    excel_file = f"results/complete_analysis_{timestamp}.xlsx"
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        # Einzeltrades
        global_trades.to_excel(writer, sheet_name='Einzeltrades', index=False)

        # Portfolio Equity
        equity_curve_df = global_equity[['timestamp', 'symbol', 'capital']].rename(columns={'capital': 'equity'})
        equity_curve_df.to_excel(writer, sheet_name='Equity_Verlauf', index=False)

        # Performance Heatmap
        heatmap = create_performance_heatmap(global_trades)
        if not heatmap.empty:
            heatmap.to_excel(writer, sheet_name='Performance_Heatmap')

        # Zeit-Analysen (Stunde & Wochentag)
        if {'hour', 'weekday'}.issubset(global_trades.columns):
            time_analysis = global_trades.groupby(['hour', 'weekday']).agg({
                'profit': ['sum', 'count', 'mean'],
                'return_pct': 'mean'
            })
            time_analysis.columns = ['_'.join(col) for col in time_analysis.columns]
            time_analysis.to_excel(writer, sheet_name='Zeit_Analysen')

        # Symbol Analyse
        symbol_analysis = global_trades.groupby('symbol').agg({
            'profit': ['sum', 'count', 'mean', 'std'],
            'return_pct': ['mean', 'std'],
            'duration_hours': 'mean',
            'risk_reward_ratio': 'mean',
            'regime': lambda x: x.mode()[0] if len(x) > 0 else 'N/A'
        })
        symbol_analysis.columns = ['_'.join(col) for col in symbol_analysis.columns]
        symbol_analysis.to_excel(writer, sheet_name='Symbol_Performance')

        # Regime Analyse
        regime_analysis = global_trades.groupby('regime').agg({
            'profit': ['sum', 'count', 'mean'], 'return_pct': 'mean', 'risk_reward_ratio': 'mean'
        })
        regime_analysis.columns = ['Profit_Sum', 'Trade_Count', 'Profit_Mean', 'Return_Mean', 'RR_Mean']
        regime_analysis.to_excel(writer, sheet_name='Regime_Analyse')

        # Pyramid Analyse
        pyramid_analysis = global_trades.groupby('pyramid_levels').agg({
            'profit': ['sum', 'count', 'mean'], 'return_pct': 'mean'
        })
        pyramid_analysis.columns = ['Profit_Sum', 'Trade_Count', 'Profit_Mean', 'Return_Mean']
        pyramid_analysis.to_excel(writer, sheet_name='Pyramid_Analyse')

        # Exit Gründe
        exit_analysis = global_trades.groupby('exit_reason').agg({
            'profit': ['sum', 'count', 'mean'], 'duration_hours': 'mean'
        })
        exit_analysis.columns = ['Profit_Sum', 'Trade_Count', 'Profit_Mean', 'Duration_Mean']
        exit_analysis.to_excel(writer, sheet_name='Exit_Analysen')

        # Cluster Summary exportieren (Zusatz - aus metrics)
        cluster_summary = trade_cluster_summary(global_trades)
        if not cluster_summary.empty:
            cluster_summary.to_excel(writer, sheet_name='Cluster_Analyse', index=False)

        # Zusammenfassung (alle Metriken)
        summary_df = pd.DataFrame(all_metrics)
        summary_df.to_excel(writer, sheet_name='Zusammenfassung', index=False)

    # Erzeuge HTML-Dashboards
    html_basic = create_advanced_dashboard(global_trades, global_equity, all_metrics, timestamp)
    html_detailed = create_detailed_dashboard(global_trades, global_equity, all_metrics, timestamp)
    html_markers = create_trade_marker_chart(global_trades, global_equity, timestamp)

    return {
        'csv': csv_file,
        'excel': excel_file,
        'html_basic_dashboard': html_basic,
        'html_detailed_dashboard': html_detailed,
        'html_marker_chart': html_markers
    }
