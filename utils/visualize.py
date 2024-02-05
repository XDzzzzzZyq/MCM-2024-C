import matplotlib.pyplot as plt
import pandas as pd


def draw_range(new_set: list[pd.Timestamp], axis: plt.axis = None):
    for i in range(len(new_set)):
        if i % 2 == 0 and i + 1 < len(new_set):
            (plt if axis is None else axis).axvspan(new_set[i], new_set[i + 1], color='orange', alpha=0.3)


def set_label(title, xlabel, ylabel, axis: plt.axis = None):
    if axis is None:
        plt.title(title, fontname='Times New Roman', fontsize=25)
        plt.xlabel(xlabel, fontname='Times New Roman', fontsize=20)
        plt.ylabel(ylabel, fontname='Times New Roman', fontsize=20)
    else:
        axis.set_title(title, fontname='Times New Roman', fontsize=20)
        axis.set_xlabel(xlabel, fontname='Times New Roman', fontsize=20)
        axis.set_ylabel(ylabel, fontname='Times New Roman', fontsize=20)

def set_caption(r:bool = False, dot=False, orig=True, axis = None):

    if r is True:
        from matplotlib.patches import Rectangle
        from matplotlib.lines import Line2D
        legend_entry = [Rectangle((0, 0), 1, 1, fc='orange', edgecolor='none', label = "different set"),
                        Line2D([0], [0], marker='o', color='w', label='P1 game win', markerfacecolor='orange', markersize=10),
                        Line2D([1], [0], marker='o', color='w', label='P2 game win', markerfacecolor='#5080A0', markersize=10)]
        if dot is False:
            legend_entry = [legend_entry[0]]

        l = (plt if axis is None else axis).legend(handles=legend_entry, fontsize=15, loc='upper right')
        if orig:(plt if axis is None else axis).legend(fontsize=15, loc='upper left')

        (plt.gca() if axis is None else axis).add_artist(l)
    else:
        if orig:plt.legend(fontsize=15)


def set_xaxis():
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M:%S'))


def save(name, bbox='tight'):
    plt.savefig(f'figs/{name}.eps', format='eps', bbox_inches=bbox)
