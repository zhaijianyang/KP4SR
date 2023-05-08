
all_tasks = {}


# =====================================================
# Task Subgroup 2 -- Sequential -- 11 Prompts
# =====================================================

task_subgroup_2 = {}

template = {}

template['source'] = "User_{} has a purchase history of {} and is expected to purchase <extra_id_0> next."
template['target'] = "<extra_id_0> {} <extra_id_1>"
template['task'] = "sequential"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'purchase_history']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "2-1"

task_subgroup_2["2-1"] = template


template = {}

template['source'] = "Based on {}'s previous purchases of {} , it is likely that they will purchase <extra_id_0> next."
template['target'] = "<extra_id_0> {} <extra_id_1>"
template['task'] = "sequential"
template['source_argc'] = 2
template['source_argv'] = ['user_desc', 'purchase_history']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "2-2"

task_subgroup_2["2-2"] = template


template = {}

template['source'] = "The items that User_{} has previously purchased include {} , and their next purchase is expected to be <extra_id_0>."
template['target'] = "<extra_id_0> {} <extra_id_1>"
template['task'] = "sequential"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'purchase_history']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "2-3"

task_subgroup_2["2-3"] = template


template = {}


template['source'] = "User_{} has a pattern of purchasing {} , and their next purchase is predicted to be <extra_id_0>."
template['target'] = "<extra_id_0> {} <extra_id_1>"
template['task'] = "sequential"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'purchase_history']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "2-4"

task_subgroup_2["2-4"] = template


template = {}

template['source'] = "According to {}'s purchase history of {} , the next item they are likely to purchase is <extra_id_0>."
template['target'] = "<extra_id_0> {} <extra_id_1>"
template['task'] = "sequential"
template['source_argc'] = 2
template['source_argv'] = ['user_desc', 'purchase_history']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "2-5"

task_subgroup_2["2-5"] = template


template = {}

template['source'] = "User_{} has previously bought {} , and <extra_id_0> is expected to be their next purchase."
template['target'] = "<extra_id_0> {} <extra_id_1>"
template['task'] = "sequential"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'purchase_history']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "2-6"

task_subgroup_2["2-6"] = template


template = {}

template['source'] = "Based on {}'s previous purchases, including {} , their next purchase is anticipated to be <extra_id_0>."
template['target'] = "<extra_id_0> {} <extra_id_1>"
template['task'] = "sequential"
template['source_argc'] = 3
template['source_argv'] = ['user_desc', 'purchase_history']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "2-7"

task_subgroup_2["2-7"] = template


template = {}

template['source'] = "{}'s purchase history includes {} , and <extra_id_0> is the next item on their list."
template['target'] = "<extra_id_0> {} <extra_id_1>"
template['task'] = "sequential"
template['source_argc'] = 3
template['source_argv'] = ['user_desc', 'purchase_history']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "2-8"

task_subgroup_2["2-8"] = template


template = {}

template['source'] = "User_{} has a record of buying {} , and it is expected that they will purchase <extra_id_0> next."
template['target'] = "<extra_id_0> {} <extra_id_1>"
template['task'] = "sequential"
template['source_argc'] = 3
template['source_argv'] = ['user_id', 'purchase_history']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "2-9"

task_subgroup_2["2-9"] = template


template = {}

template['source'] = "User_{} has shown a tendency to purchase {} , and their next purchase is likely to be <extra_id_0>."
template['target'] = "<extra_id_0> {} <extra_id_1>"
template['task'] = "sequential"
template['source_argc'] = 3
template['source_argv'] = ['user_id', 'purchase_history']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "2-10"

task_subgroup_2["2-10"] = template

template = {}

template['source'] = "User_{} purchased {} . Now the user wants to purchase <extra_id_0>."
template['target'] = "<extra_id_0> {} <extra_id_1>"
template['task'] = "sequential"
template['source_argc'] = 3
template['source_argv'] = ['user_id', 'purchase_history']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "2-10"

task_subgroup_2["2-11"] = template

all_tasks['sequential'] =  task_subgroup_2