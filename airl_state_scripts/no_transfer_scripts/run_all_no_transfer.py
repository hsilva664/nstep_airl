import os
from multiprocessing import Process, Lock
import re

# swimmer_code = 'python3 airl_state_scripts/no_transfer_scripts/swimmer_data_collect.py --visible_gpus {0} && \
#     python3 airl_state_scripts/no_transfer_scripts/swimmer_irl.py --visible_gpus {0} && \
#     python3 airl_state_scripts/no_transfer_scripts/swimmer_irl_state_action.py --visible_gpus {0} && \
#     python3 airl_state_scripts/no_transfer_scripts/swimmer_irl_no_disent.py --visible_gpus {0}'

# half_cheetah_code = 'python3 airl_state_scripts/no_transfer_scripts/half_cheetah_data_collect.py --visible_gpus {0} && \
#     python3 airl_state_scripts/no_transfer_scripts/half_cheetah_irl.py --visible_gpus {0} && \
#     python3 airl_state_scripts/no_transfer_scripts/half_cheetah_irl_state_action.py --visible_gpus {0} && \
#     python3 airl_state_scripts/no_transfer_scripts/half_cheetah_irl_no_disent.py --visible_gpus {0}'

# pendulum_code = 'python3 airl_state_scripts/no_transfer_scripts/pendulum_data_collect.py --visible_gpus {0} && \
#     python3 airl_state_scripts/no_transfer_scripts/pendulum_irl.py --visible_gpus {0} && \
#     python3 airl_state_scripts/no_transfer_scripts/pendulum_irl_state_action.py --visible_gpus {0} && \
#     python3 airl_state_scripts/no_transfer_scripts/pendulum_irl_no_disent.py --visible_gpus {0}'


# system_code = ['python3 airl_state_scripts/no_transfer_scripts/half_cheetah_irl.py --visible_gpus {0} --state_only --exp_folder state_only_no_score',
#               'python3 airl_state_scripts/no_transfer_scripts/half_cheetah_irl.py --visible_gpus {0} --state_only --score_discrim --exp_folder state_only_score',
#               'python3 airl_state_scripts/no_transfer_scripts/half_cheetah_irl.py --visible_gpus {0} --exp_folder state_action_no_score',
#               'python3 airl_state_scripts/no_transfer_scripts/half_cheetah_irl.py --visible_gpus {0} --score_discrim --exp_folder state_action_score',
#               'python3 airl_gae_scripts/half_cheetah_irl_bootstrap.py --visible_gpus {0} --n_val 10 --n_rew 10 --score_discrim --exp_folder nrew_10_nval_10_state_action_score',
#               'python3 airl_gae_scripts/half_cheetah_irl_bootstrap.py --visible_gpus {0} --n_val 10 --n_rew 10 --state_only --score_method sample_rewards --exp_folder nrew_10_nval_10_state_only_sample_rewards',
#               'python3 airl_gae_scripts/half_cheetah_irl_bootstrap.py --visible_gpus {0} --n_val 10 --n_rew 10 --state_only --score_method average_rewards --exp_folder nrew_10_nval_10_state_only_avg_rewards',
#               'python3 airl_gae_scripts/half_cheetah_irl_bootstrap.py --visible_gpus {0} --n_val 10 --n_rew 10 --state_only --score_method teacher_student --exp_folder nrew_10_nval_10_state_only_teacher_student',
#               'python3 airl_gae_scripts/half_cheetah_irl_bootstrap.py --visible_gpus {0} --n_val 10 --n_rew 10 --score_method sample_rewards --exp_folder nrew_10_nval_10_state_act_sample_rewards',
#               'python3 airl_gae_scripts/half_cheetah_irl_bootstrap.py --visible_gpus {0} --n_val 10 --n_rew 10 --score_method average_rewards --exp_folder nrew_10_nval_10_state_act_avg_rewards',
#               'python3 airl_gae_scripts/half_cheetah_irl_bootstrap.py --visible_gpus {0} --n_val 10 --n_rew 10 --score_method teacher_student --exp_folder nrew_10_nval_10_state_act_teacher_student']

system_code = ['python3 airl_gae_scripts/half_cheetah_irl_bootstrap.py --visible_gpus {0} --max_nstep 10 --n_val 10 --n_rew 10 --score_discrim --exp_folder nrew_10_nval_10_state_action_score',
              'python3 airl_gae_scripts/half_cheetah_irl_bootstrap.py --visible_gpus {0} --max_nstep 10 --n_val 10 --n_rew 10 --state_only --score_method sample_rewards --exp_folder nrew_10_nval_10_state_only_sample_rewards',
              'python3 airl_gae_scripts/half_cheetah_irl_bootstrap.py --visible_gpus {0} --max_nstep 10 --n_val 10 --n_rew 10 --state_only --score_method average_rewards --exp_folder nrew_10_nval_10_state_only_avg_rewards',
              'python3 airl_gae_scripts/half_cheetah_irl_bootstrap.py --visible_gpus {0} --max_nstep 10 --n_val 10 --n_rew 10 --state_only --score_method teacher_student --exp_folder nrew_10_nval_10_state_only_teacher_student',
              'python3 airl_gae_scripts/half_cheetah_irl_bootstrap.py --visible_gpus {0} --max_nstep 10 --n_val 10 --n_rew 10 --score_method sample_rewards --exp_folder nrew_10_nval_10_state_act_sample_rewards',
              'python3 airl_gae_scripts/half_cheetah_irl_bootstrap.py --visible_gpus {0} --max_nstep 10 --n_val 10 --n_rew 10 --score_method average_rewards --exp_folder nrew_10_nval_10_state_act_avg_rewards',
              'python3 airl_gae_scripts/half_cheetah_irl_bootstrap.py --visible_gpus {0} --max_nstep 10 --n_val 10 --n_rew 10 --score_method teacher_student --exp_folder nrew_10_nval_10_state_act_teacher_student']


# system_code = [swimmer_code, half_cheetah_code, pendulum_code]

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