from src.data_loader import load_data
"""
The entry point
"""
def get_stock_choice():
    """
    Ask user for stock choice and load data
    """
    stocks = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    print("Select a stock")
    for i, s in enumerate(stocks, start = 1): print(f"{i}) {s}")
    option = stocks[int(input("")) - 1]
    data = load_data(option)
    return data

def main():
    data = get_stock_choice()
    

if __name__ == "__main__":
    main()