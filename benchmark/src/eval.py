import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from os import walk


def read_benchmark(path: str, prefix='l') -> pd.DataFrame:
    df = None
    for _, _, filenames in walk(path):
        for f in filenames:
            if df:
                df = pd.concat([df, pd.read_csv('{}/{}'.format(path, f))])
            else:
                df = pd.read_csv('{}/{}'.format(path, f))
    df['prefix'] = prefix
    return df

def plot_throughput_gb(df: pd.DataFrame) -> None:
    pass

if __name__ == '__main__':
    # read data
    benchmark_testbench_df = read_benchmark('../testbench', 'Testbench')
    benchmark_local_df = read_benchmark('../local', 'Local')
    benchmark_df = pd.concat([benchmark_testbench_df, benchmark_local_df])

    # group df
    group_columns = ['prefix', 'gpu_count', 'blocks_per_gpu', 'threads_per_gpu']
    benchmark_df['group'] = benchmark_df[group_columns].astype(str).agg(','.join, axis=1)
    benchmark_df_mean = benchmark_df.groupby('group', as_index=False).mean(numeric_only=False)
    benchmark_df_mean['prefix'] = benchmark_df_mean['group'].apply(lambda x: x.split(',')[0])

    #plt.bar(list(range(benchmark_df_mean_np.shape[0])), np.array(benchmark_df_mean.values)[:, -2])
    #plt.show()

    print(benchmark_df_mean)
    print(benchmark_df_mean.columns)
    benchmark_df_mean = benchmark_df_mean.sort_values(by=['throughput_no_mem_gb'])
    sns.set_theme(style="ticks")
    sns.catplot(x='throughput_no_mem_gb', y='group', color='#455CFF', data=benchmark_df_mean, kind='bar', saturation=.7, height=50)

    #plt.bar(list(range(benchmark_df_mean_np.shape[0])), np.array(benchmark_df_mean.values)[:, -1])
    plt.show()