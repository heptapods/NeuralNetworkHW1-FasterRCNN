import numpy as np
import json
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm

if __name__ =="__main__":
    loss_file = "loss_and_mAP.json"
    with open(loss_file, "r") as f:
        train_process = json.loads(f.read())
    train_loss = train_process["train_loss"]
    test_loss = train_process["test_loss"]
    iter = np.arange(len(train_loss)) * 10
    plt.plot(iter, np.log(train_loss))
    plt.plot(iter, np.log(test_loss), '-.')
    plt.legend(['train_loss', 'test_loss'])
    plt.title("loss vs iteration")
    plt.xlabel('iteration')
    plt.ylabel('log(loss)')
    plt.show()

    mAP = train_process["mAP"]
    iter = np.arange(len(mAP))
    plt.plot(iter, mAP)
    plt.title("mAP vs epochs")
    plt.xlabel('epochs')
    plt.ylabel('mAP')
    plt.show()



