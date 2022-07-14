# By Alexandre Hirsch
import numpy as np
import matplotlib.pyplot as plt


def show(I, mask=False, title='', color=1):
    I = np.array(I)
    I = np.transpose(I,(1,2,0)) if I.shape[0] in {3,4} else I
    if I.shape[2] == 4:
        if mask: #replace with np.where?
            I[I[:,:,3]==1] = color
        I = I[:,:,:3]
    plt.axis('off')
    if title:
        plt.title(title)
    plt.imshow(I)
    plt.show()