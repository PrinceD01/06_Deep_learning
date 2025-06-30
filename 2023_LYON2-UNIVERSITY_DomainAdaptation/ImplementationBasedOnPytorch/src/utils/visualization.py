import matplotlib.pyplot as plt
from matplotlib import pyplot
import pylab

def visualize_loss(train_hist, plot_iter=10):
    x = range(len(train_hist['Total_loss']))
    x = [i * plot_iter for i in x]
    
    plt.plot(x, train_hist['Total_loss'], label='total loss')
    plt.plot(x, train_hist['Class_loss'], label='class loss')
    plt.plot(x, train_hist['MMD_loss'], label='mmd loss')
    
    plt.xlabel('Step')
    plt.ylabel('Loss')
    pyplot.legend(loc='best')
    plt.grid(True)
    pylab.show()