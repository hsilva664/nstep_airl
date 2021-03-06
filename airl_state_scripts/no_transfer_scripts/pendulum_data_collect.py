import tensorflow as tf

from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv

from inverse_rl.utils.log_utils import rllab_logdir
from inverse_rl.utils.hyper_sweep import run_sweep_parallel, run_sweep_serial


def main(exp_name, ent_wt=0.1, visible_gpus='0',discount=0.99):

    gpu_options = tf.GPUOptions(allow_growth=True,visible_device_list=visible_gpus)
    tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1, gpu_options=gpu_options)

    env = TfEnv(GymEnv('Pendulum-v0', record_video=False, record_log=False))
    policy = GaussianMLPPolicy(name='policy', env_spec=env.spec, hidden_sizes=(32, 32))

    with tf.Session(config=tf_config) as sess:    
        algo = TRPO(
            env=env,
            policy=policy,
            n_itr=200,
            batch_size=1000,
            max_path_length=100,
            discount=discount,
            store_paths=True,
            baseline=LinearFeatureBaseline(env_spec=env.spec),
            entropy_weight=ent_wt,
            sess=sess,
            exp_name=exp_name,            
        )

        with rllab_logdir(algo=algo, dirname='data/pendulum'):
            algo.train(sess)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--visible_gpus', type=str, default='0')
    args = parser.parse_args()  
    params_dict = {
        'ent_wt': [0.1],
        'visible_gpus': [args.visible_gpus]
    }    
    run_sweep_parallel(main, params_dict, repeat=2)
