
from zhiqiang.utils.log_parser import parse_log_for_eval_rewards

from collections import OrderedDict

# logs
log_files = OrderedDict()
log_files["VanilaDQN"] = "./data_root/zzz_reserved/log_GridWorld_VanilaDQN_2020_04_10_20_23_25.txt"
log_files["DoubleDQN"] = "./data_root/zzz_reserved/log_GridWorld_DoubleDQN_2020_04_10_21_02_09.txt"
log_files["MStepDQN"] = "./data_root/zzz_reserved/log_GridWorld_MStepDQN_2020_04_10_21_49_53.txt"
log_files["PriorityDQN"] = "./data_root/zzz_reserved/log_GridWorld_PriorityDQN_2020_04_10_22_28_07.txt"
log_files["MStepPolicy"] = "./data_root/zzz_reserved/log_GridWorld_MStepPolicy_2020_04_11_03_03_49.txt"
log_files["EntropyACQ"] = "./data_root/zzz_reserved/log_GridWorld_EntropyACQ_2020_04_11_00_03_25.txt"
log_files["EntropyACV"] = "./data_root/zzz_reserved/log_GridWorld_EntropyACV_2020_04_11_00_58_01.txt"

curve_color = {}
curve_color["VanilaDQN"] = "r"
curve_color["DoubleDQN"] = "b"
curve_color["MStepDQN"] = "k"
curve_color["PriorityDQN"] = "g"
curve_color["MStepPolicy"] = "y"
curve_color["EntropyACQ"] = "c"
curve_color["EntropyACV"] = "m"

# result_dicts, parse logs
result_dict = {}
for item in log_files.keys():
    list_result = parse_log_for_eval_rewards(log_files[item])
    result_dict[item] = list_result
#
y_label = "Rewards"
#

# list_x
list_x = [d * 1 for d in range(len(result_dict["VanilaDQN"]))]

##
# draw
import numpy as np
import matplotlib.pyplot as plt

figsize_1 = (8, 5)
#
fig = plt.figure(figsize=figsize_1)
#
for item in log_files.keys():
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

