import os
from multiprocessing import Process, Lock
import re

# ant_code = 'python3 airl_state_scripts/ant_irl.py --visible_gpus {0} && \
#     python3 airl_state_scripts/ant_transfer_disabled.py --visible_gpus {0}'

# maze_code = 'python3 airl_state_scripts/maze_right_irl.py --visible_gpus {0} && \
#     python3 airl_state_scripts/maze_left_transfer.py --visible_gpus {0}'

# system_code = [ant_code, maze_code]

system_code = ['python3 airl_state_scripts/maze_right_irl.py --visible_gpus {0} --exp_folder state_action_no_score', \
               'python3 airl_state_scripts/maze_right_irl.py --visible_gpus {0} --state_only --exp_folder state_only_no_score', \
               'python3 airl_state_scripts/maze_right_irl.py --visible_gpus {0} --state_only --score_discrim --exp_folder state_only_score', \
               'python3 airl_state_scripts/maze_right_irl.py --visible_gpus {0} --score_discrim --exp_folder state_action_score']

# for idx, single_code in enumerate(system_code):
#     os.system(single_code.format('1'))

def run_funct(lock, code, gpu):
    lock.acquire()
    os.system(code.format(gpu))
    lock.release()

processes = []
locks = [Lock() for _ in range(2)]

for idx, single_code in enumerate(system_code):
    p = Process(target=run_funct, args=(locks[idx % len(locks)], single_code, idx % len(locks)))
    p.start()
    processes.append(p)    

for p in processes:
    p.join()