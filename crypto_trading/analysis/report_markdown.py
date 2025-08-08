def export_markdown_params(params_dict, path):
    """
    Exportiert die verwendeten Parameter als Markdown-Datei.
    Format:
    # Used Parameters
    ## Kontext
    - Parameter: Wert
    """
    lines = []
    lines.append("# Used Parameters\n")

    for context, params in params_dict.items():
        lines.append(f"## {context}\n")
        if isinstance(params, dict):
            for k, v in params.items():
                lines.append(f"- **{k}**: {v}")
        else:
            lines.append(f"- Keine Parameterdaten verfügbar: {params}")
        lines.append("")  # Leerzeile

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[Markdown] Exportiert: {path}")
