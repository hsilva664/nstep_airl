import tensorflow as tf

from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv

from inverse_rl.envs.env_utils import CustomGymEnv
from inverse_rl.algos.irl_trpo import IRLTRPO
from inverse_rl.models.airl_bootstrap import *
from inverse_rl.utils.log_utils import rllab_logdir, load_latest_experts, load_latest_experts_multiple_runs
from inverse_rl.utils.hyper_sweep import run_sweep_parallel, run_sweep_serial
from tensorflow.python import debug as tf_debug


def main(exp_name=None, fusion=False, visible_gpus='0', discount=0.99, debug=False, n_val=1, n_rew=1, \
         max_nstep=1, exp_folder=None, state_only=False, score_discrim=True, score_method=None):
    env = TfEnv(CustomGymEnv('PointMazeRight-v0', record_video=False, record_log=False))

    gpu_options = tf.GPUOptions(allow_growth=True,visible_device_list=visible_gpus)
    tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1, gpu_options=gpu_options)


    # load ~2 iterations worth of data from each forward RL experiment as demos
    experts = load_latest_experts_multiple_runs('data/maze_right_data_collect', n=2, visible_gpus=visible_gpus)

    sess = tf.Session(config=tf_config)

    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)

    max_path_length=500
    irl_model = AIRL_Bootstrap(discount=discount, env=env, expert_trajs=experts, state_only=state_only, fusion=fusion, max_itrs=10, score_discrim=score_discrim, debug = debug, \
                               max_nstep = max_nstep, n_value_funct = n_val, n_rew_funct = n_rew, score_method=score_method)

    policy = GaussianMLPPolicy(name='policy', env_spec=env.spec, hidden_sizes=(32, 32))
    algo = IRLTRPO(
        env=env,
        policy=policy,
        irl_model=irl_model,
        n_itr=1000,
        batch_size=10000,
        max_path_length=max_path_length,
        discount=discount,
        store_paths=True,
        irl_model_wt=1.0,
        entropy_weight=0.1,
        zero_environment_reward=True,
        baseline=LinearFeatureBaseline(env_spec=env.spec),
    )

    # temp_folder = '/media/data/temp_exp_nstep/maze_right_state_bootstrap_%d_irl/%s'
    dirname = 'data/maze_right_state_bootstrap_%d_irl/%s/%s'%(max_nstep, exp_folder, exp_name) if exp_folder is not None else 'data/maze_right_state_bootstrap_%d_irl/%s'%(max_nstep, exp_name)
    with rllab_logdir(algo=algo, dirname=dirname ):
        sess.__enter__()
        algo.train(sess)
        sess.close()  

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--visible_gpus', type=str, default='0')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--n_val', type=int, default=1)
    parser.add_argument('--n_rew', type=int, default=1)
    parser.add_argument('--max_nstep', type=int, default=1)
    parser.add_argument('--exp_folder', type=str) 
    parser.add_argument('--score_method', type=str, default="average_rewards")
    parser.add_argument('--state_only', action="store_true")
    parser.add_argument('--score_discrim', action="store_true")    

    args = parser.parse_args()
    params_dict = {
        'fusion': [True],
        'exp_folder': [args.exp_folder],
        'visible_gpus': [args.visible_gpus],
        'debug' : [args.debug],
        'n_val' : [args.n_val],
        'n_rew' : [args.n_rew],
        'max_nstep' : [args.max_nstep],
        'state_only': [args.state_only],
        'score_discrim': [args.score_discrim],
        'score_method': [args.score_method]        
    }
    if args.debug == True:
        main(fusion=True, debug=args.debug, visible_gpus=args.visible_gpus, \
             n_val=args.n_val, n_rew=args.n_rew, max_nstep=args.max_nstep, exp_folder=args.exp_folder, \
             state_only=args.state_only, score_discrim=args.score_discrim, score_method=args.score_method)
    else:
        run_sweep_parallel(main, params_dict, repeat=1)    
