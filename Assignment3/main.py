import src.API as api
from src.data_loader import load_data


def main():
    data = load_data()
    task_num = int(input("Which Task?\n 3) Baseline\n4) k fold\n5) Dropout\n "))

    match task_num:
        case 3:
            use_test = False if int(input("are you tunning?\n1) Yes\n2) No")) == 1 else True
            if not use_test: 
                x = int(input("how many rounds of tunning?"))
                for i in range(x):
                    config = api.get_baseline_config()
                    api.run_baseline(data, config)
            else:
                config = api.get_baseline_config()
                api.run_baseline(data, config, use_test=True)

        case 4:
            config = api.get_kfold_config()
            api.run_kfold(data, config)

        case 5:
            config = api.get_dropout_config()
            api.run_dropout(data, config)


if __name__ == "__main__":
    main()