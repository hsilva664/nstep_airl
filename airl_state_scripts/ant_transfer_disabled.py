import tensorflow as tf
import os

from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv


from inverse_rl.algos.irl_trpo import IRLTRPO
from inverse_rl.envs.env_utils import CustomGymEnv
from inverse_rl.models.airl_state import *
from inverse_rl.models.tf_util import load_prior_params
from inverse_rl.utils.log_utils import rllab_logdir, load_latest_experts
from inverse_rl.utils.hyper_sweep import run_sweep_parallel, run_sweep_serial


DATA_DIR = 'data/ant_state_irl'
def main(exp_name, params_folder=None, visible_gpus='0',discount=0.99):
    env = TfEnv(CustomGymEnv('DisabledAnt-v0', record_video=False, record_log=False))

    gpu_options = tf.GPUOptions(allow_growth=True,visible_device_list=visible_gpus)
    tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1, gpu_options=gpu_options)

    irl_itr = 100  # earlier IRL iterations overfit less; 100 seems to work well.
    params_file = os.path.join(DATA_DIR, '%s/itr_%d.pkl' % (params_folder, irl_itr))
    prior_params = load_prior_params(params_file, config = tf_config)

    irl_model = AIRL(discount=discount, env=env, expert_trajs=None, state_only=True)
    policy = GaussianMLPPolicy(name='policy', env_spec=env.spec, hidden_sizes=(32, 32))
    algo = IRLTRPO(
        init_irl_params=prior_params,
        env=env,
        policy=policy,
        irl_model=irl_model,
        n_itr=1000,
        batch_size=10000,
        max_path_length=500,
        discount=discount,
        store_paths=False,
        train_irl=False,
        irl_model_wt=1.0,
        entropy_weight=0.1,
        zero_environment_reward=True,
        baseline=LinearFeatureBaseline(env_spec=env.spec),
        log_params_folder=params_folder,
        log_experiment_name=exp_name,
    )
    with rllab_logdir(algo=algo, dirname='data/ant_transfer/%s'%exp_name):
        with tf.Session(config=tf_config) as sess:
            algo.train(sess)

if __name__ == "__main__":
    import os
    params_folders = os.listdir(DATA_DIR)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--visible_gpus', type=str, default='0')
    args = parser.parse_args()        

    params_dict = {
        'params_folder': params_folders,
        'visible_gpus': [args.visible_gpus]
    }
    run_sweep_parallel(main, params_dict, repeat=1)

