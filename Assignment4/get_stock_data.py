import src.data_loader as dl
import src.visuals as vis
"""
This file is used to quiry yahoo for stock data and save it in data/filename.csv
This way we dont have to quiery yahoo everytime we want to run the program
"""
stocks = ["AAPL", "MSFT", "GOOGL", "AMZN"]
# stocks = ["TSLA", "NVDA", "JPM", "WMT"]
dl.save_data(stocks)
stock_data = dl.load_save_data(stocks)
vis.plot_stocks(stock_data)
vis.log_stocks(stock_data)