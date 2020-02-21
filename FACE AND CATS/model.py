import pandas as pd
hdf = pd.HDFStore('keras_model.h5',mode='r')
print(hdf.keys())