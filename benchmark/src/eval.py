import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
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


def plot_filter_throughput_gb(df: pd.DataFrame) -> None:
    df = df.copy()
    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(20, 15))
    for gpus in [1, 2]:
        df_gpu = df[benchmark_df_mean['gpu_count'] == gpus]
        throughput = df_gpu.pivot(index='threads_per_gpu', columns='blocks_per_gpu', values='throughput_gb')
        sns.heatmap(throughput, ax=axs[(gpus-1)][0])
        throughput = df_gpu.pivot(index='threads_per_gpu', columns='blocks_per_gpu', values='elements_per_thread')
        sns.heatmap(throughput, ax=axs[(gpus-1)][1], norm=LogNorm(0.1, 40000))
        axs[(gpus-1)][0].set_title('Throughput GB/s Gpus={}'.format(gpus))
        axs[(gpus-1)][0].set(xlabel='blocks', ylabel='threads')
        axs[(gpus-1)][0].set_xticklabels(axs[(gpus-1)][0].get_xticklabels(), rotation=45, ha='right')
        axs[(gpus-1)][1].set_title('Elements/Thread GB/s Gpus={}'.format(gpus))
        axs[(gpus-1)][1].set(xlabel='blocks', ylabel='threads')
        axs[(gpus-1)][1].set_xticklabels(axs[(gpus-1)][1].get_xticklabels(), rotation=45, ha='right')
        fig.suptitle('Datasize={}GB'.format((df_gpu['element_count'].max()*df_gpu['element_size'].max()/pow(10, 9))))
    fig.tight_layout()
    fig.savefig('../images/filter_throughput.pdf')
    #plt.show()

if __name__ == '__main__':
    # read data
    benchmark_testbench_df = read_benchmark('../testbench', 'Testbench')
    benchmark_local_df = read_benchmark('../local', 'Local')
    #benchmark_df = pd.concat([benchmark_testbench_df, benchmark_local_df])
    benchmark_df = benchmark_testbench_df

    # group df
    group_columns = ['prefix', 'gpu_count', 'blocks_per_gpu', 'threads_per_gpu']
    benchmark_df['group'] = benchmark_df[group_columns].astype(str).agg(','.join, axis=1)
    benchmark_df_mean = benchmark_df.groupby('group', as_index=False).mean(numeric_only=False)
    benchmark_df_mean['prefix'] = benchmark_df_mean['group'].apply(lambda x: x.split(',')[0])

    #plt.bar(list(range(benchmark_df_mean_np.shape[0])), np.array(benchmark_df_mean.values)[:, -2])
    #plt.show()

    print(benchmark_df_mean)
    print(benchmark_df_mean.columns)
    benchmark_df_mean = benchmark_df_mean.sort_values(by=['throughput_gb'])
    sns.set_theme(style="ticks")
    #sns.catplot(x='throughput_gb', y='group', color='#455CFF', data=benchmark_df_mean, kind='bar', saturation=.7, height=50)

    plot_filter_throughput_gb(benchmark_df_mean)