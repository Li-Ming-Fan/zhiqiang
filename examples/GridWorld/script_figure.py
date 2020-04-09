
from zhiqiang.utils.log_parser import parse_log_for_eval_rewards


# logs
log_files = {}
log_files["VanilaDQN"] = "./data_root/log/log_VanilaDQN_GridWorld_2020-04-06-23-39-04.txt"

curve_color = {}
curve_color["VanilaDQN"] = "b"

# parse logs
result_dict = {}
for item in log_files.keys():
    list_result = parse_log_for_eval_rewards(log_files[item])
    result_dict[item] = list_result
#
y_label = "Rewards"
#

# list_x
list_x = [d * 5 for d in range(20)]



##
# draw
import numpy as np
import matplotlib.pyplot as plt

figsize_1 = (8, 5)
#
fig = plt.figure(figsize=figsize_1)
#
for item in result_dict.keys():
    color = curve_color[item]
    list_result = result_dict[item]
    list_y = [list_result[idx] for idx in range(len(list_x))]
    plt.plot(list_x, list_y, label=item, color=color, linewidth=2)
    #
'''
color：指定曲线的颜色。颜色可以表示为
1，英文单词或略写；
2，如‘#FF0000’格式的3个16进制数；
3，如（1.0,0.0,0.0）格式的3元组。
'''

##
plt.xlabel("Numer of Boost")
plt.ylabel(y_label)
# plt.title("Learning Curriculum")

# plt.xticks(list_x)
plt.legend()
plt.grid()

plt.show()

#
# fig.savefig("./data_root/rewards.pdf", format='pdf', transparent=True, pad_inches = 0, bbox_inches="tight")
#

