import os
import tools.tools as t
dataset_name = 'img_align_celeba'
dataset_dir = '../../data/%s'%dataset_name
eval_dataset_name = 'img_align_celeba_eval'
eval_dataset_dir = '../../data/%s'%eval_dataset_name
os.makedirs(eval_dataset_dir, exist_ok=True)
eval_list = t.readlines('../../data/Eval/list_eval_partition.txt')
for line in eval_list:
    file, is_eval = line.split()
    is_eval = int(is_eval)
    print(line)
    if is_eval:
        os.rename(os.path.join(dataset_dir, file),os.path.join(eval_dataset_dir, file))
