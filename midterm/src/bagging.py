import src.models as mdl
import numpy as np
import statistics as st
import src.data_loader as dl

# Implement bootstrap aggregating with multiple SVC models.
def lda_bootstrap(X_train, y_train, X_test, y_test, ker, C, gamma=0, degree=0) :
    error_v, error_t, total_time = bootstrap(X_train, y_train, X_test, y_test, 0,  ker, C, gamma, degree, "lda")

def pca_bootstrap(X_train, y_train, X_test, y_test, num_comps,  ker, C, gamma=0, degree=0) :
    error_v, error_t, total_time = bootstrap(X_train, y_train, X_test, y_test, num_comps,  ker, C, gamma, degree, "pca")

def bootstrap(X_train, y_train, X_test, y_test, num_comps,  ker, C, gamma=0, degree=0, model="lda"):
    y_pred_v_arr = np.zeros((8,len(y_train)))
    y_pred_t_arr = np.zeros((8,len(y_train)))
    train_time_arr = [0]*8
    X_train_arr = np.split(X_train,8)
    y_train_arr = np.split(y_train,8)
    train_vote = np.zeros((8,len(y_train)))
    test_vote =  []
    for i in range(8):
        if model == "pca" :
            y_pred_v_arr[i:], _, train_time_arr[i:] = mdl.pca_model(
                X_train_arr[i], y_train_arr[i], X_test, y_test, num_comps,  ker, C, gamma, degree)
        else :
            y_pred_v_arr[i:], _, train_time_arr[i:] = mdl.lda_model(
                X_train_arr[i], y_train_arr[i], X_test, y_test, ker, C, gamma, degree)
    print(y_pred_t_arr)
    pred_t = st.mode(train_vote)
    pred_v = st.mode(test_vote)
    total_time = sum(train_time_arr)
    error_v = np.count_nonzero(pred_v != y_test) / len(y_test)
    error_t = np.count_nonzero(pred_t != y_train) / len(y_train)
    return error_v, error_t, total_time

def compare_bagging():
    X_train, y_train, X_test, y_test = dl.load_mnist(); 
    dim_red_method = input("LDA or PCA: ").lower()
    ker = input("Enter Kernel: ")
    C, gamma, degree = mdl.get_params(ker)
    if dim_red_method == "pca" :
        num_comps = int(input("Number components: "))
        single_err_v, single_err_t, single_time = mdl.pca_pipeline(X_train,y_train,X_test,y_test,num_comps,ker,C,gamma,degree)
        print(f"single error train: {single_err_t}, single error test: {single_err_v}, single run-time: {single_time}")
        bagging_err_v, bagging_err_t, bagging_time = pca_bootstrap(X_train,y_train,X_test,y_test,num_comps,ker,C,gamma,degree)
        print(f"bagging error train: {bagging_err_t}, bagging error test: {bagging_err_v}, bagging run-time: {bagging_time}")
    else :
        # single_err_v, single_err_t, single_time = mdl.lda_pipeline(X_train,y_train,X_test,y_test,ker,C,gamma,degree)
        # print(f"single error train: {single_err_t}, single error test: {single_err_v}, single run-time: {single_time}")
        bagging_err_v, bagging_err_t, bagging_time = lda_bootstrap(X_train,y_train,X_test,y_test,ker,C,gamma,degree)
        print(f"bagging error train: {bagging_err_t}, bagging error test: {bagging_err_v}, bagging run-time: {bagging_time}")

compare_bagging()