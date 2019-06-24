import os

os.system('python3 airl_gae_scripts/maze_right_irl_bootstrap_temp.py --visible_gpus 1 --state_only --n_val 5 --n_rew 5 --max_nstep 10 --score_method sample_rewards --exp_folder mstep_10_nval_5_nrew_5_sample_rew')
os.system('python3 airl_gae_scripts/maze_right_irl_bootstrap_temp.py --visible_gpus 1 --state_only --n_val 5 --n_rew 5 --max_nstep 10 --score_method average_rewards --exp_folder mstep_10_nval_5_nrew_5_avg_rew')
os.system('python3 airl_gae_scripts/maze_right_irl_bootstrap_temp.py --visible_gpus 1 --state_only --n_val 5 --n_rew 5 --max_nstep 10 --score_method teacher_student --exp_folder mstep_10_nval_5_nrew_5_teacher_student')