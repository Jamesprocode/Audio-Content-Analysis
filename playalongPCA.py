import numpy as np

import pandas as pd

data = np.array([
    [2.5, 2.325,  1.8],
    [0.5, 0.325,  -0.2],
    [2.2, 2.120, -0.7],
    [1.9, 1.065,  2.1],
    [3.1, 2.735,  -0.1],
    [2.3, 2.005,  1.5]
])

# first pass to pandas and plot as-is using seaborn
df = pd.DataFrame(data, columns=['MFCC1', 'Flux', 'ZCR'])
print(df)