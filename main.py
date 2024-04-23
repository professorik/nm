import utils
import waves
import numpy as np
import time


if __name__ == '__main__':
    start = time.time()
    model = waves.Model(beta=np.pi / 2, d=0.01, L=0.05, _lambda=0.0005, K=120, N=50)
    table = model.solve(w=4)
    print(time.time() - start)

    utils.heatmap_waves(table, w=4*0.05, h=0.2)
    #utils.plt.legend()
    utils.plt.show()
