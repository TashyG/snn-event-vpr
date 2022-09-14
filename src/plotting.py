import numpy as np
import matplotlib.pyplot as plt
import seaborn


def plot_confusion_matrix(data, labels, annotations, output_filename, vmin, vmax):
    """Plot confusion matrix using heatmap.
 
    Args:
        data (list of list): List of lists with confusion matrix data.
        labels (list): Labels which will be plotted across x and y axis.
        output_filename (str): Path to output file.
 
    """
    
    seaborn.set(color_codes=True)
    #plt.figure(1, figsize=(int(num_classes*0.6), int(num_classes*0.6*(2/3)))) # 2x
    plt.figure(1, figsize=(12, 8)) # 2x
 
    plt.title("Confusion Matrix")
 
    seaborn.set(font_scale=1.4)

    ax = seaborn.heatmap(data, annot=annotations, annot_kws={'fontsize': 'xx-small'}, fmt='', cbar_kws={'label': 'Scale'},
                  linewidths=0.1, linecolor='gray')
    #vmin=vmin, vmax=vmax,
    #vmin=0, vmax=0.3,
    labels = [label.get_text() for label in ax.get_yticklabels()]
    
    ax.set_yticklabels(labels, rotation=0)
 
    ax.set(ylabel="Predicted Label", xlabel="True Label")
 
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    plt.close()
