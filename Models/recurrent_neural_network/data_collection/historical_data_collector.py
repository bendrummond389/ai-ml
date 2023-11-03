import yfinance as yf


def download_stock_data(
    ticker, start_date, end_date, interval="1d", output_path="data/historical/"
):
    """
    Downloads historical stock data for a given ticker symbol.

    Parameters:
    ticker (str): The stock ticker symbol.
    start_date (str): Start date in format 'YYYY-MM-DD'.
    end_date (str): End date in format 'YYYY-MM-DD'.
    interval (str): Data interval. Default is '1d' for daily.
    output_path (str): Path to save the downloaded data.
    """
    print(f"Downloading historical stock data for {ticker}")
    data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    filename = f"{ticker}_{start_date}_to_{end_date}.csv"
    data.to_csv(output_path + filename)
    print(f"Data saved to {output_path + filename}")


if __name__ == "__main__":
    # Example usage
    download_stock_data("AAPL", "2010-01-01", "2023-01-01")
