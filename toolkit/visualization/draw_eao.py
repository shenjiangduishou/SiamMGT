import matplotlib.pyplot as plt
import numpy as np
import pickle

from matplotlib import rc
from .draw_utils import COLOR1, MARKER_STYLE

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

def draw_eao(result):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')
    angles = np.linspace(0, 2*np.pi, 8, endpoint=True)
    for r in np.linspace(0.2, 1.0, 5):
        ax.plot(np.linspace(0, 2*np.pi, 100),
                np.ones(100)*r,
                color='lightgray',
                linestyle='-',
                linewidth=0.7,
                alpha=0.6)

    attr2value = []
    for i, (tracker_name, ret) in enumerate(result.items()):
        value = list(ret.values())
        attr2value.append(value)
        value.append(value[0])
    attr2value = np.array(attr2value)
    max_value = np.max(attr2value, axis=0)
    min_value = np.min(attr2value, axis=0)
    for i, (tracker_name, ret) in enumerate(result.items()):
        value = list(ret.values())
        value.append(value[0])
        value = np.array(value)
        value *= (1 / max_value)
        plt.plot(angles, value, linestyle='-', color=COLOR1[i], marker=MARKER_STYLE[i],
                label=tracker_name, linewidth=1.5, markersize=12,markerfacecolor='none',
                 markeredgewidth=1.5)
    attrs = ["Overall", "Camera motion",
             "Illumination change","Motion Change",
             "Size change","Occlusion",
             "Unassigned"]
    attr_value = []
    for attr, maxv, minv in zip(attrs, max_value, min_value):
        attr_value.append(attr + "\n({:.3f},{:.3f})".format(minv, maxv))
    ax.set_thetagrids(angles[:-1] * 180/np.pi, attr_value,fontsize=12,
                      verticalalignment='center',horizontalalignment='center')#雷达边上的显示
    ax.spines['polar'].set_visible(False)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5,-0.05), frameon=False, ncol=5,fontsize=14)
    ax.grid()
    ax.set_ylim(0, 1.18)
    ax.set_yticks([])
    # pos = ax.get_position()
    # new_pos = [pos.x0, pos.y0 + 0.1, pos.width, pos.height]  # y0 增加 0.1
    # ax.set_position(new_pos)
    plt.tight_layout(pad=1.0)
    plt.show()

if __name__ == '__main__':
    result = pickle.load(open("../../result.pkl", 'rb'))
    draw_eao(result)
