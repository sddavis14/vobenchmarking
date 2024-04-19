import matplotlib.pyplot as plt
import numpy as np

def evaluate():
    data = np.genfromtxt('../results/predicted.csv', delimiter=',')
    print(data)

if __name__ == '__main__':
    evaluate()
