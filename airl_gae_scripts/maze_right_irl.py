import tensorflow as tf

from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv

from inverse_rl.envs.env_utils import CustomGymEnv
from inverse_rl.algos.irl_trpo import IRLTRPO
from inverse_rl.models.airl_nstep import *
from inverse_rl.utils.log_utils import rllab_logdir, load_latest_experts, load_latest_experts_multiple_runs
from inverse_rl.utils.hyper_sweep import run_sweep_parallel, run_sweep_serial

def main(exp_name=None, fusion=False, visible_gpus='0', discount=0.99, debug=False):
    env = TfEnv(CustomGymEnv('PointMazeRight-v0', record_video=False, record_log=False))

    gpu_options = tf.GPUOptions(allow_growth=True,visible_device_list=visible_gpus)
    tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1, gpu_options=gpu_options)

    # load ~2 iterations worth of data from each forward RL experiment as demos
    experts = load_latest_experts_multiple_runs('data/maze_right_data_collect', n=2, visible_gpus=visible_gpus)

    nstep = 10
    irl_model = AIRL_Nstep(discount=discount, env=env, expert_trajs=experts, state_only=False, fusion=fusion, max_itrs=10, number_obs = nstep, score_discrim=True)

    policy = GaussianMLPPolicy(name='policy', env_spec=env.spec, hidden_sizes=(32, 32))
    algo = IRLTRPO(
        env=env,
        policy=policy,
        irl_model=irl_model,
        n_itr=1000,
        batch_size=10000,
        max_path_length=500,
        discount=discount,
        store_paths=True,
        irl_model_wt=1.0,
        entropy_weight=0.1,
        zero_environment_reward=True,
        baseline=LinearFeatureBaseline(env_spec=env.spec),
    )
    with rllab_logdir(algo=algo, dirname='data/maze_right_state_nstep_%d_irl/%s' % (nstep, exp_name) ):
        with tf.Session(config=tf_config) as sess:
            algo.train(sess)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--visible_gpus', type=str, default='0')    
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    params_dict = {
        'fusion': [True],
        'visible_gpus': [args.visible_gpus],
        'debug' : [args.debug]
    }
    if args.debug == True:
        main(fusion=True, debug=args.debug, visible_gpus=args.visible_gpus)
    else:
        run_sweep_parallel(main, params_dict, repeat=1) 
