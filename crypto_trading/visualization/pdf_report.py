# crypto_trading/backtesting/report_pdf.py
from fpdf import FPDF
import os

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Crypto Backtest Report', ln=True, align='C')
        self.ln(10)

    def add_summary(self, summary_df):
        self.set_font("Arial", size=10)
        self.cell(0, 10, "Performance-Übersicht", ln=True)
        self.ln(5)
        for _, row in summary_df.iterrows():
            self.cell(0, 8, 
                f"{row['symbol']} {row['interval']} "
                f"CAGR: {row['CAGR[%]']:.2f}% | "
                f"Sharpe: {row['Sharpe Ratio']:.2f} | "
                f"MaxDD: {row['Max Drawdown [%]']:.2f}%",
                ln=True)

    def add_trade_table(self, trades_df, symbol, interval):
        self.add_page()
        self.set_font("Arial", 'B', 12)
        self.cell(0, 10, f"Trades {symbol} @ {interval}", ln=True)
        self.set_font("Arial", size=8)
        col_width = self.w / 5.5
        self.ln(5)
        # Header
        for col in trades_df.columns:
            self.cell(col_width, 8, str(col), border=1)
        self.ln(8)
        # Rows
        for _, row in trades_df.iterrows():
            for col in trades_df.columns:
                self.cell(col_width, 8, str(row[col]), border=1)
            self.ln(8)

    def add_chart(self, chart_path):
        self.add_page()
        self.set_font("Arial", 'B', 12)
        self.cell(0, 10, "Chart", ln=True)
        self.ln(5)
        self.image(chart_path, x=15, w=180)

def create_pdf(summary_df, trades_dict, charts_dict, output_path):
    pdf = PDFReport()
    pdf.add_page()
    pdf.add_summary(summary_df)
    for key, trades_df in trades_dict.items():
        symbol, interval = key
        pdf.add_trade_table(trades_df, symbol, interval)
        if key in charts_dict:
            pdf.add_chart(charts_dict[key])
    pdf.output(output_path)
    print(f"PDF gespeichert: {output_path}")
