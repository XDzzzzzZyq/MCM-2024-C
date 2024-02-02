import matplotlib.pyplot as plt

def set_label(title, xlabel, ylabel):
    plt.title(title, fontsize=20)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)

def save(name):
    plt.savefig(f'figs/{name}.eps', format='eps', bbox_inches='tight')