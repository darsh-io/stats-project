import yfinance as yf
import pandas as pd

invpercom = 20000
stocks = {
    "Tech": ["AAPL","MSFT","GOOGL","NVDA","ORCL"],
    "Healthcare": ["JNJ","LLY","PFE","MRK","ABT"],
    "Tourism": ["MAR","HLT","BKNG","EXPE","CCL"],
    "Finance": ["JPM","BAC","GS","V","MA"],
    "Energy": ["XOM","CVX","COP","SHEL","TTE"],
    "Utilities": ["NEE","DUK","SO","D","AEP"]
}
portfolio = pd.DataFrame()
for sector, tickers in stocks.items():
    sector_df = pd.DataFrame()
    for ticker in tickers:
        data = yf.download(
            ticker,
            start="2005-01-04",
            auto_adjust=True,
            progress=False
        )
        if len(data) == 0:
            continue
        purchase_price = data["Close"].iloc[0]
        shares = invpercom / purchase_price
        sector_df[ticker] = data["Close"] * shares
    portfolio[sector] = sector_df.sum(axis=1)

portfolio["Total Portfolio"] = portfolio.sum(axis=1)
portfolio.to_csv("DATA.csv")
#from https://github.com/NZX37/statsdata/blob/main/main.py
