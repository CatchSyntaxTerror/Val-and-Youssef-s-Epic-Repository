import src.models as mods
import src.graphing as gr
import src.data_loader as dl
import src.bagging as bagger
import time as time

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

def run_dim_timing(tech, X_train, y_train, X_test, y_test):
    is_pca = tech == "pca"
    C = 0.05
    num_comps = get_comps() if is_pca else -1

    start = time.time()
    model_func = mods.build_pca_model if is_pca  else mods.build_lda_model
    model = model_func(num_comps, "linear", C, -1, -1)
    err_test, err_train, _ = mods.run_model(model, X_train, y_train, X_test, y_test)
    end = time.time() - start
    print(f"Training Error: {err_train:.3f}, Test Error: {err_test:.3f}, Time: {end:.3f}")

def run_kernel(model_func, X_train, y_train, X_test, y_test, kernel):
    is_pca = model_func == mods.build_pca_model
    num_comps = get_comps() if is_pca else -1
    
    C, gamma, degree = get_params(kernel, True)
    model = model_func(num_comps, kernel, C, gamma, degree)
    err_v, err_t, time = mods.run_model(model, X_train, y_train, X_test, y_test)
    print(f"Final: error_test: {err_v}, time: {time}")

def run_tunning(model_func, X_train, y_train, X_test, y_test, kernel):
    """
    run a 
    """
    is_pca = model_func == mods.build_pca_model
    folds = mods.get_folds(X_train, y_train)
    num_comps = get_comps() if is_pca else -1
    err_v, err_t, time = 0, 0, 0
    print(f"error_v: {err_v}, error_t: {err_t}, time: {time}")
    
    for x_re, y_tr, x_valid, y_valid in folds:
        C, gamma, degree = get_params(kernel, False)
        model = model_func(num_comps, kernel, C, gamma, degree)
        err_v, err_t, time = mods.run_model(model, x_re, y_tr, x_valid, y_valid)
        print(f"error_v: {err_v}, error_t: {err_t}, time: {time}")
        gr.record_test(err_v, err_t, time, C, gamma, degree, kernel, num_comps, is_pca)

    C, gamma, degree = get_params(kernel, True)
    model = model_func(num_comps, kernel, C, gamma, degree)
    err_v, err_t, time = mods.run_model(model, X_train, y_train, X_test, y_test)
    gr.record_final(err_v, time, C, gamma, degree, kernel, num_comps, is_pca)
    print(f"Final: error_test: {err_v}, time: {time}")

def get_input(tech):
    """
    bagging input
    """
    num_comps = 0
    if tech == "pca": num_comps = get_comps()
    ker = input("Enter Kernal: ").lower()
    C, gamma, degree = get_params(ker, False)
    return ker, C, gamma, degree, num_comps

def timeDimRed(tech:str):
    """
    external method for main. Calls run_dim_timing
    """
    ker = "linear"
    X_train, y_train, X_test, y_test = dl.load_fashion_mnist();
    if tech == "pca": run_dim_timing(tech, X_train, y_train,X_test,y_test)
    else: run_dim_timing(tech, X_train, y_train,X_test,y_test)
    
def kernel(tech:str):
    """
    external method for main. Calls run_tuning
    """
    ker = input("Enter Kernel: ")
    X_train, y_train, X_test, y_test = dl.load_fashion_mnist();
    if tech == "pca": run_kernel(mods.build_pca_model, X_train, y_train, X_test, y_test, ker)
    else: run_kernel(mods.build_lda_model, X_train, y_train, X_test, y_test, ker)

def tune(tech:str):
    """
    external method for main. Calls run_tuning
    """
    ker = input("Enter Kernel: ")
    X_train, y_train, X_test, y_test = dl.load_mnist();
    if tech == "pca": run_tunning(mods.build_pca_model, X_train, y_train, X_test, y_test, ker)
    else: run_tunning(mods.build_lda_model, X_train, y_train, X_test, y_test, ker)

def bag(tech):
    """
    external method for bagging. Calls run bagging
    """
    num_bags = int(input("Enter number of bags: "))
    X_train, y_train, X_test, y_test = dl.load_mnist()
    time, error, ker, num_comps = bagger.run_bagging(tech, X_train, y_train, X_test, y_test, num_bags)
    gr.record_bagging(tech, num_bags, error, time, ker, num_comps)