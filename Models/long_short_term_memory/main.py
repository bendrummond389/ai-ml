from data.collection.collect_data import download_aapl_data
from data.processing.process_data import process_data

if __name__ == "__main__":
    training = process_data(file_path="data/storage/AAPL_training_data.csv")