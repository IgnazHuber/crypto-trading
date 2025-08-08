# mod_markdown.py
import os

def export_to_markdown(summary_df, all_trades_dict, path="results/summary.md"):
    lines = ["# Backtest Summary", ""]
    lines.append("## Zusammenfassung")
    lines.append(summary_df.to_markdown(index=False))
    lines.append("")
    for name, trades in all_trades_dict.items():
        lines.append(f"## Trades – {name}")
        lines.append(trades.to_markdown(index=False))
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[Markdown/TXT] Export abgeschlossen: {path}")

def export_params_to_markdown(params, param_origin=None, path="results/used_params.md"):
    """Speichert die Parameter in eine Markdown-Datei, inkl. Herkunft (default/optimized)."""
    lines = ["# Verwendete Parameter", ""]
    for k, v in params.items():
        if param_origin and k in param_origin:
            src = "(default)" if param_origin[k] == "default" else "(optimized)"
            lines.append(f"- **{k}**: {v} {src}")
        else:
            lines.append(f"- **{k}**: {v}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[INFO] Parameter-Markdown gespeichert: {path}")
