import matplotlib.pyplot as plt
import pandas as pd

def draw_range(new_set: list[pd.Timestamp]):
    for i in range(len(new_set)):
        if i%2==0 and i+1<len(new_set):
            plt.axvspan(new_set[i], new_set[i+1], color='orange', alpha=0.3)


def set_label(title, xlabel, ylabel):
    plt.title(title, fontsize=20)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)


def set_xaxis():
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M:%S'))

def save(name, bbox='tight'):
    plt.savefig(f'figs/{name}.eps', format='eps', bbox_inches=bbox)