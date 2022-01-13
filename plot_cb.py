"""
This callback is a copy-paste with minor adjustments of the one by
Daniel, which can be found here:
https://medium.com/geekculture/how-to-plot-model-loss-while-training-in-tensorflow-9fa1a1875a5
all credits to him.
"""

import numpy as np 
from tensorflow import keras
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")

from IPython.display import clear_output

class PlotLearning(keras.callbacks.Callback):
    """
    Callback to plot the learning curves of the model during training.
    """
    def on_train_begin(self, logs={}):
        self.metrics = {}
        for metric in logs:
            self.metrics[metric] = []
            

    def on_epoch_end(self, epoch, logs={}):
        # Storing metrics
        for metric in logs:
            if metric in self.metrics:
                self.metrics[metric].append(logs.get(metric))
            else:
                self.metrics[metric] = [logs.get(metric)]
        
        # Plotting
        metrics = [x for x in logs if 'val' not in x[0:3]]
        
        n_cols = len(metrics)//2
        if len(metrics)%2 != 0:
            n_cols += 1
        f, axs = plt.subplots(2, n_cols, figsize=(n_cols*3, 2*3))
        clear_output(wait=True)
        j = 0
        for k, metric in enumerate(metrics):
            if k >= n_cols:
                j = 1
                i = k - n_cols
            else:
                j = 0
                i = k
            
            axs[j][i].plot(range(1, epoch + 2), 
                        self.metrics[metric], 
                        label=metric)
            if 'sigma' in metric:
                axs[j][i].set_yscale('log')
            if 'val_' + metric in logs:
                axs[j][i].plot(range(1, epoch + 2), 
                            self.metrics['val_' + metric], 
                            label='val_' + metric)
                
            axs[j][i].legend()
            #axs[j][i].grid()

        plt.tight_layout()
        plt.show()