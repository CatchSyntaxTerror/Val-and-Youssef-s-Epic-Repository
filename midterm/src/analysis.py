import src.models as mods
import src.graphing as gr
import src.data_loader as dl

"""
Run project analyses and generate results.
"""

def get_params(ker, test = False):
    """
    Gets C, gamma and degree from user.
    """
    if test: str = "Input Final "
    else: str = "Input "
    match ker:
        case "linear": 
            C = float(input(f"{str}C: "))
            gamma = -1.0
            degree = -1
        case "rbf":
            C = float(input(f"{str}C: "))
            gamma = float(input(f"{str}gamma: "))
            degree = -1
        case "poly":
            C = float(input(f"{str}C: "))
            gamma = float(input(f"{str}gamma: "))
            degree = int(input(f"{str}degree: "))
    return C, gamma, degree


def get_comps():
    n = int(input("enter number of comps for PCA: "))
    return n

def run_test(model_func, X_train, y_train, X_test, y_test, kernel):
    """
    run a 
    """
    is_pca = model_func == mods.build_pca_model
    folds = mods.get_folds(X_train, y_train)
    num_comps = get_comps() if is_pca else -1
    err_v, err_t, time = 0, 0, 0
    
    for x_re, y_tr, x_valid, y_valid in folds:
        C, gamma, degree = get_params(kernel, err_v, err_t, time)
        print(f"error_v: {err_v}, error_t: {err_t}, time: {time}")
        model = model_func(num_comps, kernel, C, gamma, degree)
        err_v, err_t, time = mods.run_model(model, x_re, y_tr, x_valid, y_valid)
        gr.record_test(err_v, err_t, time, C, gamma, degree, kernel, num_comps, is_pca)

    C, gamma, degree = get_params(kernel, err_v, err_t, time)
    model = model_func(num_comps, kernel, C, gamma, degree)
    err_v, err_t, time = mods.run_model(model, X_train, y_train, X_test, y_test)
    gr.record_final(err_v, time, C, gamma, degree, kernel, num_comps, is_pca)
    print(f"Final: error_test: {err_v}, time: {time}")

def tune(tech:str):
    ker = input("Enter Kernel: ")
    X_train, y_train, X_test, y_test = dl.load_mnist();
    if tech == "pca": run_test(mods.build_pca_model, X_train, y_train, X_test, y_test, ker)
    else: run_test(mods.build_lda_model, X_train, y_train, X_test, y_test, ker)
