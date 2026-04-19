import src.utils as utils
from src.data_loader import load_data
from src.train import run_training_loop

"""
The entry point of the program
"""
def get_data():
    """ Ask user for stock choice, load data, start training loop """
    stocks = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    option = utils.get_selection("Select a stock", stocks)
    return load_data(option, 50)

def main():
    data = get_data()
    models = ["RNN", "GRU", "LSTM"]
    option = utils.get_selection("Select a model", models)
    run_training_loop(option, data)

if __name__ == "__main__":
    main()