import numpy as np
import matplotlib.pyplot as plt

from toolkit.visualization.draw_utils import COLOR, LINE_STYLE2


def get_result(success_ret,name,videos, attr):
    """pretty print result
    Args:
        result: returned dict from function eval
    """
    # sort tracker
    tracker_auc = {}
    success = []
    for tracker_name in success_ret.keys():
        value = [v for k, v in success_ret[tracker_name].items() if k in videos]
        tracker_auc[tracker_name] = np.mean(value)
    return tracker_auc,attr#返回一个字典其中键是跟踪器名字，值是成功率，arrt是对应挑战

def radar_map(dataset,success_ret):
    plt.style.use('ggplot')
    tracker_names = {}
    chanllenge = []
    for i in success_ret.keys():
        tracker_names[i] = []
    for attr, videos in dataset.attr.items():
        tracker_auc,attr = get_result(success_ret, name=dataset.name, videos=videos, attr=attr)
        chanllenge.append(attr)
        for j in tracker_auc.keys():
            tracker_names[j].append(tracker_auc[j])

    angles = np.linspace(0, 2 * np.pi, len(chanllenge), endpoint=False)
    fig = plt.figure(figsize=(6, 4.8))
    ax = fig.add_subplot(111, polar=True)
    for n,k in enumerate(tracker_names):
        ax.plot(np.concatenate((angles, [angles[0]])), np.concatenate((tracker_names[k], [tracker_names[k][0]])), LINE_STYLE2[n], linewidth=2,
            label=k, color=COLOR[n])
    # logo = [str(x) + '\n' + '(' + str(tracker_names[i][m]) + ')' for m, x in enumerate(chanllenge)]
    logo = [f"{x}\n({tracker_names[i][m]:.2g})" for m, x in enumerate(chanllenge)]
    ax.set_thetagrids(angles * 180 / np.pi, logo)
    ax.set_ylim(0, 0.75)
    ax.set_yticks([0, 0.2, 0.4, 0.6])
    plt.title('Success Rate', fontsize=15,x=-0.15,y=1.05)
    plt.legend(loc='lower center', prop={'size': 10}, ncol=4, bbox_to_anchor=[0.5, -0.3])  # 控制图标的居中和往下的距离
    ax.grid(True)
    # 调整子图位置，将底部边距增大，顶部边距减小，实现整体向上移动
    fig.subplots_adjust(bottom=0.2, top=0.9)
    plt.show()


if __name__ == '__main__':
    plt.style.use('ggplot')
    foundation = [0.562, 0.393, 0.329, 0.318, 0.370, 0.282, 0.407, 0.222, 0.312, 0.406, 0.387, 0.370, 0.389, 0.352,
                  0.353, 0.639, 0.403, 0.418, 0.387, 0.412]
    protrack = [0.580, 0.396, 0.342, 0.386, 0.395, 0.334, 0.444, 0.267, 0.321, 0.428, 0.388, 0.363, 0.416, 0.358, 0.386,
                0.458, 0.414, 0.425, 0.391, 0.420]
    ViPT = [0.684, 0.503, 0.461, 0.438, 0.459, 0.412, 0.542, 0.350, 0.426, 0.557, 0.518, 0.465, 0.500, 0.460, 0.465,
            0.650, 0.514, 0.525, 0.495, 0.525]
    feature = ['NO', 'PO', 'TO', 'HO', 'MB', 'LI', 'HI', 'AIV', 'LR', 'DEF', 'BC', 'SA', 'CM', 'TC', 'FL', 'OV', 'FM',
               'SV', 'ARC', 'ALL']

    angles = np.linspace(0, 2 * np.pi, len(feature), endpoint=False)
    fig = plt.figure(figsize=(6, 4.8))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(np.concatenate((angles, [angles[0]])), np.concatenate((foundation, [foundation[0]])), 's-', linewidth=2,
            label='Foundation', color='orange')
    ax.plot(np.concatenate((angles, [angles[0]])), np.concatenate((protrack, [protrack[0]])), '^-', linewidth=2,
            label='ProTrack', color='blue')
    ax.plot(np.concatenate((angles, [angles[0]])), np.concatenate((ViPT, [ViPT[0]])), 'o-', linewidth=2, label='ViPT',
            color='red')
    logo = [str(x) + '\n' + '(' + str(ViPT[i]) + ')' for i, x in enumerate(feature)]
    ax.set_thetagrids(angles * 180 / np.pi, logo)
    ax.set_ylim(0, 0.75)
    ax.set_yticks([0, 0.2, 0.4, 0.6])
    plt.title('Success Rate', fontsize=15,x=-0.15,y=1.05)
    plt.legend(loc='lower center', prop={'size': 10}, ncol=4, bbox_to_anchor=[0.5, -0.2])  # 控制图标的居中和往下的距离
    ax.grid(True)
    # 调整子图位置，将底部边距增大，顶部边距减小，实现整体向上移动
    fig.subplots_adjust(bottom=0.2, top=0.9)
    plt.show()