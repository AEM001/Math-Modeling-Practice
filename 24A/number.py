import numpy as np

def number(arr, decimals=6):
    if isinstance(arr, (list, np.ndarray)):
        for i in range(len(arr)):
            if isinstance(arr[i], (list, np.ndarray)):
                for j in range(len(arr[i])):
                    arr[i][j] = round(arr[i][j], decimals)
            else:
                arr[i] = round(arr[i], decimals)
    return arr