import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

if __name__ == '__main__':
    benchmark_df = pd.read_csv('../local/run_1.csv')
    benchmark_df_mean = benchmark_df.groupby(['gpu_count', 'threads_per_gpu']).mean()
    #print(benchmark_df)
    print(benchmark_df_mean)
    print(np.array(benchmark_df_mean.values)[:, -2])
    print(np.array(benchmark_df_mean.values)[:, -1])

    plt.bar([1, 2, 3, 4, 5, 6], np.array(benchmark_df_mean.values)[:, -2], 0.2)
    plt.show()

    plt.bar([1, 2, 3, 4, 5, 6], np.array(benchmark_df_mean.values)[:, -1] / 8, 0.2)
    plt.show()