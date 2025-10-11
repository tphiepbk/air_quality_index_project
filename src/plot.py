# Author: tphiepbk

import matplotlib.pyplot as plt

# ==========================================================================================

def plot_1_data(data, datalabel="label", xlabel="xlabel", ylabel="ylabel", figsize=(13, 5)):
    size = len(data)
    scheme = [x for x in range(size)]
    plt.figure(figsize=figsize)
    plt.plot(scheme, data, marker='.', label=datalabel)
    plt.ylabel(ylabel, size=15)
    plt.xlabel(xlabel, size=15)
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.show()

# ==========================================================================================

def plot_2_data(data1, data2, datalabel1="label1", datalabel2="label2", xlabel="xlabel", ylabel="ylabel", figsize=(13, 5)):
    assert len(data1) == len(data2), "data1 and data2 should be the same length"
    size = len(data1)
    scheme=[x for x in range(size)]
    plt.figure(figsize=figsize)
    plt.plot(scheme, data1, c='b', label=datalabel1)
    plt.plot(scheme, data2, c='r', label=datalabel2)
    plt.ylabel(ylabel, size=15)
    plt.xlabel(xlabel, size=15)
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.show()

# ==========================================================================================

def plot_3_data(data1, data2, data3, datalabel1="label1", datalabel2="label2", datalabel3="label3", xlabel="xlabel", ylabel="ylabel", figsize=(13, 5)):
    assert len(data1) == len(data2) == len(data3), "data1 and data2 should be the same length"
    size = len(data1)
    scheme=[x for x in range(size)]
    plt.figure(figsize=figsize)
    plt.plot(scheme, data1, c='b', label=datalabel1)
    plt.plot(scheme, data2, c='r', label=datalabel2)
    plt.plot(scheme, data3, c='g', label=datalabel3)
    plt.ylabel(ylabel, size=15)
    plt.xlabel(xlabel, size=15)
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.show()

# ==========================================================================================

def plot_4_data(data1, data2, data3, data4, datalabel1="label1", datalabel2="label2", datalabel3="label3", datalabel4="label4", xlabel="xlabel", ylabel="ylabel", figsize=(13, 5)):
    assert len(data1) == len(data2) == len(data3) == len(data4), "all data should be the same length"
    size = len(data1)
    scheme=[x for x in range(size)]
    plt.figure(figsize=figsize)
    plt.plot(scheme, data1, c='b', label=datalabel1)
    plt.plot(scheme, data2, c='r', label=datalabel2)
    plt.plot(scheme, data3, c='g', label=datalabel3)
    plt.plot(scheme, data4, c='m', label=datalabel4)
    plt.ylabel(ylabel, size=15)
    plt.xlabel(xlabel, size=15)
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.show()

# ==========================================================================================

def plot_5_data(data1, data2, data3, data4, data5,
                datalabel1="label1", datalabel2="label2", datalabel3="label3", datalabel4="label4", datalabel5="label5",
                xlabel="xlabel", ylabel="ylabel",
                figsize=(13, 5)):
    assert len(data1) == len(data2) == len(data3) == len(data4) == len(data5), "all data should be the same length"
    size = len(data1)
    scheme=[x for x in range(size)]
    plt.figure(figsize=figsize)
    plt.plot(scheme, data1, c='b', label=datalabel1)
    plt.plot(scheme, data2, c='r', label=datalabel2)
    plt.plot(scheme, data3, c='g', label=datalabel3)
    plt.plot(scheme, data4, c='m', label=datalabel4)
    plt.plot(scheme, data5, c='y', label=datalabel5)
    plt.ylabel(ylabel, size=15)
    plt.xlabel(xlabel, size=15)
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.show()

# ==========================================================================================

def plot_learning_curves(history, figsize=(12, 5)):
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=figsize)

    plt.plot(epochs, train_loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# ==========================================================================================

# Visualize the prediction
def plot_prediction(inv_y_pred, inv_y_test, n_future):
    print(f"inv_y_pred.shape = {inv_y_pred.shape}\ninv_y_test.shape = {inv_y_test.shape}")
    for day in range(n_future):
        plot_2_data(data1=inv_y_pred[:, day],
                    data2=inv_y_test[:, day],
                    datalabel1="Prediction",
                    datalabel2="Actual",
                    xlabel="Time step",
                    ylabel="PM2.5")
