import yfinance as yf

from datetime import datetime, timedelta


def download_recent_stock_data(
    ticker, days_back=360, interval="1d", output_path="data/recent/"
):
    """
    Downloads the most recent stock data for a given ticker symbol.

    Parameters:
    ticker (str): The stock ticker symbol.
    days_back (int): Number of days back from today to start collecting data.
    interval (str): Data interval. Default is '1d' for daily.
    output_path (str): Path to save the downloaded data.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    print(f"Downloading recent stock data for {ticker}")

    data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    filename = f"{ticker}_recent.csv"
    data.to_csv(output_path + filename)
    print(f"Data saved to {output_path + filename}")


if __name__ == "__main__":
    download_recent_stock_data("AAPL")
