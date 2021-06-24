import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import axisartist
from mpl_toolkits.axes_grid1 import host_subplot


# plot fit
def plot_tps(dir_out, tf_seqs, file_name=""):
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(9,6))
    for ax in axes.flat:
        ax.set(xlabel='sequences', ylabel='TPs')
    for itm in tf_seqs.items():
        i = 0 if itm[0] < 3 else 1
        j = int(itm[0])
        # the histogram of the data
        axes[i % 2][j % 3].set_title("order= " + str(itm[0]))
        axes[i % 2][j % 3].set_xticks(np.arange(start=0, stop=len(itm[1][0]), step=1))
        axes[i % 2][j % 3].grid(True)
        axes[i % 2][j % 3].set_yticks(np.arange(start=0, stop=1.1, step=0.1))
        axes[i % 2][j % 3].label_outer()
        for ln in itm[1]:
            if len(ln) > 0:
                axes[i % 2][j % 3].axis([0,len(ln), 0.0, 1.1])
                vl = [round(x,2) if x != "-" else 0 for x in ln]
                axes[i % 2][j % 3].plot(range(0,len(ln)),vl)
    # fig.tight_layout()
    # plt.show()
    fn = ""
    if file_name:
        fn = "_" + file_name
    fig.savefig(dir_out + "tps" + fn, bbox_inches="tight")
    plt.clf()
    plt.close()


# plot fit
def plot_fits(dir_out, ngen, fits, novs, title):
    plt.figure()
    plt.title(title)
    plt.xlabel('ngen')
    plt.plot(range(0, ngen), fits, label="fitness")
    plt.plot(range(0, ngen), novs, label="novelty")
    plt.legend()
    # plt.tight_layout()
    plt.savefig(dir_out + "fits", bbox_inches="tight")
    plt.clf()
    plt.close()


# plot
def plot_data(dir_out, ngen, fits, novs, narchs, title):
    host = host_subplot(111, axes_class=axisartist.Axes)
    host.set_title(title)
    par1 = host.twinx()
    new_fixed_axis = par1.get_grid_helper().new_fixed_axis
    par1.axis["right"] = new_fixed_axis(loc="right", axes=par1, offset=(20, 0))

    par1.axis["right"].toggle(all=True)

    host.set_xlim(0, ngen)
    host.set_ylim(0, max(fits))
    par1.set_xlim(0, ngen)
    par1.set_ylim(0, max(narchs) + 1)

    host.set_xlabel("ngen")
    host.set_ylabel("values")
    par1.set_ylabel("narchiv")

    host.plot(range(0, ngen), fits, label="fitness")
    host.plot(range(0, ngen), novs, label="novelty")
    p1, = par1.plot(range(0, ngen), narchs, '--', label="narch")

    host.legend()

    par1.axis["right"].label.set_color(p1.get_color())

    plt.savefig(dir_out + "fig2", bbox_inches="tight")
    plt.clf()
    plt.close()


# plot pareto like
def plot_pareto(dir_out, pop, bests, title):
    plt.figure()
    plt.title(title)
    plt.xlabel('novelty')
    plt.ylabel('fitness')
    plt.scatter(pop["novs"], pop["fits"], label="population", c='gray')
    plt.scatter(bests["novs"], bests["fits"], label="selected", c="red")
    plt.legend()
    # plt.tight_layout()
    plt.savefig(dir_out + "pareto", bbox_inches="tight")
    plt.clf()
    plt.close()

