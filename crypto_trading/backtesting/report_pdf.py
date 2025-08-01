# crypto_trading/backtesting/report_pdf.py
from fpdf import FPDF
import os

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Crypto Backtest Report', ln=True, align='C')
        self.ln(10)

def create_pdf(summary_df, output_path):
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_font("Arial", size=10)
    for _, row in summary_df.iterrows():
        pdf.cell(0, 10, f"{row['symbol']} {row['interval']} - CAGR: {row['CAGR[%]']:.2f}%  Sharpe: {row['Sharpe Ratio']:.2f}", ln=True)
    pdf.output(output_path)
    print(f"PDF gespeichert: {output_path}")
