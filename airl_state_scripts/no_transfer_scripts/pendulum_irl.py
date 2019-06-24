import tensorflow as tf

from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv


from inverse_rl.algos.irl_trpo import IRLTRPO
from inverse_rl.models.imitation_learning import AIRLStateAction
from inverse_rl.models.airl_state import *
from inverse_rl.utils.log_utils import rllab_logdir, load_latest_experts

from inverse_rl.utils.hyper_sweep import run_sweep_parallel, run_sweep_serial


def main(exp_name=None, fusion=False, visible_gpus='0', discount=0.99):
    env = TfEnv(GymEnv('Pendulum-v0', record_video=False, record_log=False))

    gpu_options = tf.GPUOptions(allow_growth=True,visible_device_list=args.visible_gpus)
    tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1, gpu_options=gpu_options)

    experts = load_latest_experts('data/pendulum', n=5, visible_gpus=visible_gpus)

    irl_model = AIRL(discount=discount, env=env, expert_trajs=experts, state_only=True, fusion=args.fusion, max_itrs=10)
    policy = GaussianMLPPolicy(name='policy', env_spec=env.spec, hidden_sizes=(32, 32))
    algo = IRLTRPO(
        env=env,
        policy=policy,
        irl_model=irl_model,
        n_itr=200,
        batch_size=1000,
        max_path_length=100,
        discount=discount,
        store_paths=True,
        discrim_train_itrs=50,
        irl_model_wt=1.0,
        entropy_weight=0.1, # this should be 1.0 but 0.1 seems to work better
        zero_environment_reward=True,
        baseline=LinearFeatureBaseline(env_spec=env.spec)
    )

    with rllab_logdir(algo=algo, dirname='data/pendulum_airl_state_only'):
        with tf.Session(config=tf_config) as sess:
            algo.train(sess)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--visible_gpus', type=str, default='0')
    parser.add_argument('--fusion', action='store_false')
    args = parser.parse_args()
    params_dict = {
        'fusion': [args.fusion],
        'visible_gpus': [args.visible_gpus]
    }
    run_sweep_parallel(main, params_dict, repeat=2)