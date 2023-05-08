
all_tasks = {}

# =====================================================
# Task Subgroup 2 -- Sequential -- 13 Prompts
# =====================================================

task_subgroup_2 = {}

template = {}

template['source'] = "User_{} has listened to {} in the past, and now they might listen to <extra_id_0>."
template['target'] = "<extra_id_0> {} <extra_id_1>"
template['task'] = "sequential"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'history']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "2-1"

task_subgroup_2["2-1"] = template


template = {}

template['source'] = "{}'s listening history includes {} , and they may be interested in listening to <extra_id_0>."
template['target'] = "<extra_id_0> {} <extra_id_1>"
template['task'] = "sequential"
template['source_argc'] = 2
template['source_argv'] = ['user_desc', 'history']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "2-2"

task_subgroup_2["2-2"] = template


template = {}

template['source'] = "After listening to {} , user_{} may consider tuning in to <extra_id_0>."
template['target'] = "<extra_id_0> {} <extra_id_1>"
template['task'] = "sequential"
template['source_argc'] = 2
template['source_argv'] = ['history', 'user_id']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "2-3"

task_subgroup_2["2-3"] = template


template = {}

template['source'] = "Having previously listened to {} , user_{} could now add <extra_id_0> to their listening repertoire."
template['target'] = "<extra_id_0> {} <extra_id_1>"
template['task'] = "sequential"
template['source_argc'] = 2
template['source_argv'] = ['history', 'user_id']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "2-4"

task_subgroup_2["2-4"] = template


template = {}

template['source'] = "{}'s previous listening choices were {} , and they could now be inclined to listen to <extra_id_0>."
template['target'] = "<extra_id_0> {} <extra_id_1>"
template['task'] = "sequential"
template['source_argc'] = 2
template['source_argv'] = ['user_desc', 'history']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "2-5"

task_subgroup_2["2-5"] = template


template = {}

template['source'] = "With a listening history of {} , user_{} may find <extra_id_0> to be an appealing option."
template['target'] = "<extra_id_0> {} <extra_id_1>"
template['task'] = "sequential"
template['source_argc'] = 2
template['source_argv'] = ['history', 'user_id']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "2-6"

task_subgroup_2["2-6"] = template


template = {}

template['source'] = "{} are among the shows that user_{} has listened to before, and <extra_id_0> could be their next pick."
template['target'] = "<extra_id_0> {} <extra_id_1>"
template['task'] = "sequential"
template['source_argc'] = 3
template['source_argv'] = ['history', 'user_id']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "2-7"

task_subgroup_2["2-7"] = template


template = {}

template['source'] = "After tuning in to {} , user_{} may be interested in checking out <extra_id_0>."
template['target'] = "<extra_id_0> {} <extra_id_1>"
template['task'] = "sequential"
template['source_argc'] = 3
template['source_argv'] = ['history', 'user_id']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "2-8"

task_subgroup_2["2-8"] = template


template = {}

template['source'] = "User_{} has a history of listening to {} , and now they may want to add <extra_id_0> to the mix."
template['target'] = "<extra_id_0> {} <extra_id_1>"
template['task'] = "sequential"
template['source_argc'] = 3
template['source_argv'] = ['user_id', 'history']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "2-9"

task_subgroup_2["2-9"] = template


template = {}

template['source'] = "Based on {}'s previous listening history which includes {} , <extra_id_0> might be a suitable choice for them now."
template['target'] = "<extra_id_0> {} <extra_id_1>"
template['task'] = "sequential"
template['source_argc'] = 3
template['source_argv'] = ['user_desc', 'history']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "2-10"

task_subgroup_2["2-10"] = template

template = {}

template['source'] = "User_{} listened {} . Now the user wants to listen <extra_id_0>."
template['target'] = "<extra_id_0> {} <extra_id_1>"
template['task'] = "sequential"
template['source_argc'] = 3
template['source_argv'] = ['user_id', 'history']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "2-10"

task_subgroup_2["2-11"] = template


all_tasks['sequential'] =  task_subgroup_2

