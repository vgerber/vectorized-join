
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import gc

def draw_bar(path, output_name):

    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=1.8)
    df_csv = pd.read_csv(path, sep=';')

    print(df_csv)

    #benenne csv spalte um
    df_grouped = df_csv.rename({'vectorsize': 'vector size MiB', 'b': 'Y'}, axis='columns')
    #umrechnen von vector size auf mega byte
    df_grouped['vector size MiB'] = df_grouped['vector size MiB'].divide(128 * 1024)
    #nimm nur besten throughput wert je vectorsize
    df_grouped = df_grouped.groupby(['vector size MiB']).agg({'throughput [GiB / s ]': 'max'})
    df_grouped = df_grouped.reset_index()

    print(df_grouped)


    #erstelle plot
    sns_plot = sns.barplot(data=df_grouped, x='vector size MiB', y='throughput [GiB / s ]', ci=None,color="teal", saturation=.8)


    #speichern als png und pdf (vector graphik für latex)
    sns_plot.figure.savefig(output_name+".png")
    plt.savefig(output_name+".pdf", bbox_inches='tight')



if __name__ == '__main__':

    #größe und dpi festlegen
    fig = plt.figure(figsize=(14, 9), dpi=300)

    sns.set_style("whitegrid")

    #farbpalette
    sns.color_palette("viridis", as_cmap=True)

    #funktion zum bar chart aufrufen mit csv pfad + output name
    draw_bar(
        path='atomic_add.txt',
        output_name='atomics_jetson'

         )



    plt.show()
    gc.collect()


