

def parse_log_for_eval_rewards(log_file):
    """
    """
    fp = open(log_file, "r", encoding="utf-8")
    lines = fp.readlines()
    fp.close()

    ##
    anchor_str = "INFO - max_aver_rewards, aver_rewards:"
    #
    def check_and_extract_reward(line_str):
        found = False
        reward = 0
        if anchor_str in line_str:
            str_arr = line_str.split(anchor_str)
            str_data = str_arr[-1]
            str_arr = str_data.split(",")
            #
            reward = float(str_arr[-1].strip())
            found = True
        #
        return found, reward
        #
    
    ##
    list_result = []
    for line in lines:
        found, result = check_and_extract_reward(line)
        if found:
            list_result.append(result)
        #
    #
    return list_result
    #
   
