# Author: tphiepbk

import matplotlib.pyplot as plt

# ==========================================================================================

def plot_1_data(data, datalabel="label", xlabel="xlabel", ylabel="ylabel", figsize=(20, 8)):
  size = len(data)
  scheme = [x for x in range(size)]
  plt.figure(figsize=figsize)
  plt.plot(scheme, data, marker='.', label=datalabel)
  plt.ylabel(ylabel, size=15)
  plt.xlabel(xlabel, size=15)
  plt.legend(fontsize=15)
  plt.show()

# ==========================================================================================

def plot_2_data(data1, data2, datalabel1="label1", datalabel2="label2", xlabel="xlabel", ylabel="ylabel", figsize=(20, 8)):
  assert len(data1) == len(data2), "data1 and data2 should be the same length"
  size = len(data1)
  scheme=[x for x in range(size)]
  plt.figure(figsize=figsize)
  plt.plot(scheme, data1, c='b', label=datalabel1)
  plt.plot(scheme, data2, c='r', label=datalabel2)
  plt.ylabel(ylabel, size=15)
  plt.xlabel(xlabel, size=15)
  plt.legend(fontsize=15)
  plt.show()