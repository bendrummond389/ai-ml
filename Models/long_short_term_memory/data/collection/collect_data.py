import yfinance as yf
import os


def download_and_save_data(file_path, ticker, start_date, end_date, interval="1d"):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    stock_data = yf.download(ticker, start=start_date, end=end_date, interval=interval)

    stock_data.to_csv(file_path)


def download_aapl_data():
    training_file_path = "data/storage/AAPL_training_data.csv"
    validation_file_path = "data/storage/AAPL_validation_data.csv"

    download_and_save_data(training_file_path, "AAPL", "2013-01-01", "2022-01-01")
    download_and_save_data(validation_file_path, "AAPL", "2023-01-01", "2023-09-01")


if __name__ == "__main__":
    download_aapl_data()
