from src.data_loader import load_data
from src.model import Net


def main():
    X_train, y_train, X_test, y_test = load_data()
    model = Net(X_train.shape[1], [128, 64], dropout=0.5)
    print(model)


if __name__ == "__main__":
    main()