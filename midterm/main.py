import src.analysis as ana
import src.data_loader as dl

"""
Main program. Everything can be runn from here
"""
def main():

    str = input("What would you like to run?\nA) Tunning\nB) Bagging\n").lower()
    tech = input("LDA or PCA: ").lower()
    if str == "a": ana.tune(tech)
    if str == "b": ana.bag()

if __name__ == main():
    main()