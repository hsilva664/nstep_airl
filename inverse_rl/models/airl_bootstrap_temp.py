import tensorflow as tf
import numpy as np
from sandbox.rocky.tf.spaces.box import Box

from inverse_rl.models.fusion_manager import RamFusionDistr
from inverse_rl.models.imitation_learning import SingleTimestepIRL
from inverse_rl.models.architectures import relu_net
from inverse_rl.utils import TrainingIterator

from inverse_rl.utils.math_utils import gauss_log_pdf, categorical_log_pdf
from tensorflow.python import debug as tf_debug

DIST_GAUSSIAN = 'gaussian'
DIST_CATEGORICAL = 'categorical'

class AIRL_Bootstrap_temp(SingleTimestepIRL):
    """ 
    Args:
        fusion (bool): Use trajectories from old iterations to train.
        state_only (bool): Fix the learned reward to only depend on state.
        score_discrim (bool): Use log D - log 1-D as reward (if true you should not need to use an entropy bonus)
        max_itrs (int): Number of training iterations to run per fit step.
    """
    def __init__(self, env,
                 expert_trajs=None,
                 reward_arch=relu_net,
                 reward_arch_args=None,
                 value_fn_arch=relu_net,
                 score_discrim=False,
                 sess = None,
                 discount=1.0,
                 max_nstep = 10,
                 n_value_funct = 1,
                 n_rew_funct = 1,
                 state_only=False,
                 max_itrs=100,
                 fusion=False,
                 debug=False,
                 score_method = None,
                 name='airl'):
        super(AIRL_Bootstrap_temp, self).__init__()
        env_spec = env.spec
        if reward_arch_args is None:
            reward_arch_args = {}

        if fusion:
            self.fusion = RamFusionDistr(100, subsample_ratio=0.5)
        else:
            self.fusion = None
        self.dO = env_spec.observation_space.flat_dim
        self.dU = env_spec.action_space.flat_dim
        assert isinstance(env.action_space, Box)
        self.score_discrim = score_discrim
        self.gamma = discount
        assert value_fn_arch is not None
        self.set_demos(expert_trajs)
        self.state_only=state_only
        self.max_itrs=max_itrs
        self.max_nstep = max_nstep
        self.n_value_funct = n_value_funct
        self.n_rew_funct = n_rew_funct

        self.reward_arch = reward_arch
        self.reward_arch_args = reward_arch_args
        self.value_fn_arch = value_fn_arch

        self.score_method = score_method

        self.debug = debug

        # build energy model
        with tf.variable_scope(name) as _vs:
            # Should be batch_size x T x dO/dU
            self.obs_t = tf.placeholder(tf.float32, [None, None, self.dO], name='obs')
            self.nobs_t = tf.placeholder(tf.float32, [None, self.dO], name='nobs')
            self.act_t = tf.placeholder(tf.float32, [None, None,self.dU], name='act')
            self.nact_t = tf.placeholder(tf.float32, [None, self.dU], name='nact')
            self.labels = tf.placeholder(tf.float32, [None, 1], name='labels')
            self.lprobs = tf.placeholder(tf.float32, [None, 1], name='log_probs')
            self.lr = tf.placeholder(tf.float32, (), name='lr')

            number_obs = tf.shape(self.obs_t)[1]

            with tf.variable_scope('discrim') as dvs:
                rew_input = self.obs_t
                if not self.state_only:
                    rew_input = tf.concat([self.obs_t, self.act_t], axis=2)

                self.reward = [None for i in range(self.n_rew_funct)]
                self.value_fn = [None for i in range(self.n_value_funct)]
                fitted_value_fn_n = [None for i in range(self.n_value_funct)]
                self.qfn = [ [None for i in range(self.n_value_funct)] for j in range(self.n_rew_funct)]
                self.discrim_output = [ [None for i in range(self.n_value_funct)] for j in range(self.n_rew_funct)]
                self.loss = [ [None for i in range(self.n_value_funct)] for j in range(self.n_rew_funct)]
                self.step = [ [None for i in range(self.n_value_funct)] for j in range(self.n_rew_funct)]

                log_q_tau = self.lprobs

                for i in range(self.n_rew_funct):
                    with tf.variable_scope('reward_%d'%(i), reuse=tf.AUTO_REUSE):                    
                        self.reward[i] = self.reward_arch(tf.reshape(rew_input, [-1, rew_input.shape[2] ]), dout=1, **self.reward_arch_args)
                        self.reward[i] = tf.reshape(self.reward[i], [-1, number_obs, 1])
                        #energy_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)                    
                for j in range(self.n_value_funct):
                    # value function shaping
                    with tf.variable_scope('vfn_%d'%(j), reuse=tf.AUTO_REUSE):
                        fitted_value_fn_n[j] = self.value_fn_arch(self.nobs_t, dout=1)
                    with tf.variable_scope('vfn_%d'%(j), reuse=tf.AUTO_REUSE):
                        self.value_fn[j] = self.value_fn_arch(self.obs_t[:,0,:], dout=1)

                self.avg_reward = tf.reduce_mean( tf.stack(self.reward), axis=0)

                gamma_coefs = tf.concat([tf.ones([1], dtype = tf.float32) ,self.gamma * tf.ones([number_obs - 1], dtype = tf.float32)], axis=0)
                gamma_coefs = tf.cumprod(gamma_coefs)
                gamma_coefs = tf.expand_dims(gamma_coefs, axis=1)

                for i in range(self.n_rew_funct):
                    for j in range(self.n_value_funct):
                        # Define log p_tau(a|s) = r + gamma * V(s') - V(s) 
                        self.qfn[i][j] = tf.reduce_sum(self.reward[i]*gamma_coefs, axis=1) + tf.math.pow( tf.constant(self.gamma), tf.to_float(number_obs) )*fitted_value_fn_n[j]
                        log_p_tau = self.qfn[i][j] - self.value_fn[j]                   

                        log_pq = tf.reduce_logsumexp([log_p_tau, log_q_tau], axis=0)
                        self.discrim_output[i][j] = tf.exp(log_p_tau-log_pq)
                        cent_loss = -tf.reduce_mean(self.labels*(log_p_tau-log_pq) + (1-self.labels)*(log_q_tau-log_pq))

                        self.loss[i][j] = cent_loss

                        self.step[i][j] = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss[i][j])

                self._combine_predictions()
            self._make_param_ops(_vs)


    def _combine_predictions(self):
            t0_reward = [None for i in range(self.n_rew_funct)]
            t0_value = [None for i in range(self.n_value_funct)]

            reward = [None for i in range(self.n_rew_funct)]
            fitted_value_fn_n = [None for i in range(self.n_value_funct)]

            log_p_tau = [ [None for i in range(self.n_rew_funct * self.n_value_funct)] for j in range(self.max_nstep)]


            rew_input = self.obs_t
            if not self.state_only:
                rew_input = tf.concat([self.obs_t, self.act_t], axis=2)                
            for i in range(self.n_rew_funct):
                with tf.variable_scope('reward_%d'%(i), reuse=tf.AUTO_REUSE):     
                    reward[i] = self.reward_arch(tf.reshape(rew_input, [-1, rew_input.shape[2] ]), dout=1, **self.reward_arch_args)
                    reward[i] = tf.reshape(reward[i], [-1, self.max_nstep, 1])
                    t0_reward[i] = reward[i][:,0] 

            #Master-student score method
            with tf.variable_scope('student_reward'):
                self.student_reward = self.reward_arch(rew_input[:,0], dout=1, **self.reward_arch_args)                
                
            v_input = tf.concat( [self.obs_t[:,1:,:], tf.expand_dims(self.nobs_t, axis=1) ], axis = 1)                 
            for j in range(self.n_value_funct):
                # value function shaping
                with tf.variable_scope('vfn_%d'%(j), reuse=tf.AUTO_REUSE):
                    fitted_value_fn_n[j] = self.value_fn_arch(tf.reshape(v_input, [-1, v_input.shape[2] ]), dout=1)
                    fitted_value_fn_n[j] = tf.reshape(fitted_value_fn_n[j], [-1, self.max_nstep, 1])                    
                with tf.variable_scope('vfn_%d'%(j), reuse=tf.AUTO_REUSE):
                    t0_value[j] = self.value_fn_arch(self.obs_t[:,0,:], dout=1)

            #Master-student score method
            with tf.variable_scope('student_value', reuse=tf.AUTO_REUSE):
                self.student_value_n = self.value_fn_arch(v_input[:,0], dout=1)                   
            with tf.variable_scope('student_value', reuse=tf.AUTO_REUSE):
                self.student_value = self.value_fn_arch(self.obs_t[:,0,:], dout=1)                    

            gamma_coefs = np.ones([self.max_nstep], dtype = np.float32)
            gamma_coefs[1:] *=  self.gamma
            gamma_coefs = np.cumprod(gamma_coefs)
            gamma_coefs = np.expand_dims(gamma_coefs, axis=1)

            log_q_tau = self.lprobs

            for i in range(self.n_rew_funct):
                for j in range(self.n_value_funct):
                    for single_nsteps in range(1,self.max_nstep+1):
                        # Define log p_tau(a|s) = r + gamma * V(s') - V(s) 
                        qfn = tf.reduce_sum(reward[i][:,:single_nsteps]*gamma_coefs[:single_nsteps], axis=1) + (self.gamma**single_nsteps)*(fitted_value_fn_n[j][:,(single_nsteps-1) ])
                        log_p_tau[single_nsteps - 1][i*self.n_value_funct + j] = qfn - t0_value[j]

            #Master-student score method
            student_qfn = self.student_reward + self.gamma*self.student_value_n
            student_log_p_tau = student_qfn - self.student_value

            mean_list = [None for i in range(self.max_nstep)]
            variance_list = [None for i in range(self.max_nstep)]

            for i in range(self.max_nstep):
                mean_list[i] = tf.reduce_mean( tf.stack(log_p_tau[i]), axis=0 )
                variance_list[i] = tf.math.reduce_variance( tf.stack(log_p_tau[i]), axis=0 )

            self.weights = tf.concat(variance_list, axis=1)
            self.weights = tf.nn.softmax(1./(self.weights + 1e-8), axis=1)

            # self.weights = tf.constant( ([0.0]*(self.max_nstep-1)) + [1.0], dtype=tf.float32 )

            log_p_tau = tf.reduce_sum(self.weights * tf.concat(mean_list, axis=1), axis=1, keepdims = True)
        
            log_pq = tf.reduce_logsumexp([log_p_tau, log_q_tau], axis=0)                
            self.ensemble_discrim_output = tf.exp(log_p_tau-log_pq)

            #Master-student score method
            student_log_pq = tf.reduce_logsumexp([student_log_p_tau, log_q_tau], axis=0)  
            self.student_discrim_output = tf.exp(student_log_p_tau-student_log_pq)   
            self.student_loss = -tf.reduce_mean(tf.stop_gradient(self.ensemble_discrim_output)*(student_log_p_tau-student_log_pq) + \
                                 (1-tf.stop_gradient(self.ensemble_discrim_output))*(log_q_tau-student_log_pq))
            self.student_step = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.student_loss)
            self.student_absolute_loss = -tf.reduce_mean(self.labels*(student_log_p_tau-student_log_pq) + (1-self.labels)*(log_q_tau-student_log_pq))     

            self.ensemble_loss = -tf.reduce_mean(self.labels*(log_p_tau-log_pq) + (1-self.labels)*(log_q_tau-log_pq))     

            self.t0_reward = tf.reduce_mean(tf.concat(t0_reward, axis=1), axis=1, keepdims=True)
            self.t0_value = tf.reduce_mean(tf.concat(t0_value, axis=1), axis=1, keepdims=True)

    def _reorganize_states(self, paths, number_obs = 1, pad_val=0.0):
        for path in paths:
            if 'observations_next' in path:
                continue
            nobs = path['observations'][number_obs:]
            nact = path['actions'][number_obs:]
            nobs = np.r_[nobs, pad_val*np.ones(shape=[number_obs, self.dO ], dtype=np.float32 )]
            nact = np.r_[nact, pad_val*np.ones(shape=[number_obs, self.dU ], dtype=np.float32 )]
            path['observations_next'] = nobs
            path['actions_next'] = nact

            multiple_obs = np.ones( (path['observations'].shape[0], number_obs, self.dO), dtype=np.float32 )
            multiple_act = np.ones( (path['actions'].shape[0], number_obs, self.dU), dtype=np.float32 )

            for idx in range(path['observations'].shape[0]):
                if idx+number_obs < path['observations'].shape[0]:
                    final_idx = idx+number_obs
                    obs = path['observations'][idx:final_idx]
                    act = path['actions'][idx:final_idx]
                else:
                    final_idx = path['observations'].shape[0]
                    delta_idx = number_obs - (final_idx - idx)
                    obs = np.r_[ path['observations'][idx:final_idx] , np.ones(shape=[delta_idx, self.dO ], dtype=np.float32) ]
                    act = np.r_[ path['actions'][idx:final_idx] , np.ones(shape=[delta_idx, self.dU ], dtype=np.float32) ]
                multiple_obs[idx,:,:] = obs
                multiple_act[idx,:,:] = act

            path['multi_observations'] = multiple_obs
            path['multi_actions'] = multiple_act

        return paths

    def fit(self, paths, policy=None, batch_size=32, logger=None, lr=1e-3,**kwargs):

        if self.fusion is not None:
            old_paths = self.fusion.sample_paths(n=len(paths))
            self.fusion.add_paths(paths)
            paths = paths+old_paths

        # eval samples under current policy
        self._compute_path_probs(paths, insert=True)

        # eval expert log probs under current policy
        self.eval_expert_probs(self.expert_trajs, policy, insert=True)

        self._reorganize_states(paths, number_obs=self.max_nstep)
        self._reorganize_states(self.expert_trajs, number_obs=self.max_nstep)
        obs, obs_next, acts, acts_next, path_probs = \
            self.extract_paths(paths,
                               keys=('multi_observations', 'observations_next', 'multi_actions', 'actions_next', 'a_logprobs'))
        expert_obs, expert_obs_next, expert_acts, expert_acts_next, expert_probs = \
            self.extract_paths(self.expert_trajs,
                               keys=('multi_observations', 'observations_next', 'multi_actions', 'actions_next', 'a_logprobs'))


        all_obs = np.concatenate([obs, expert_obs], axis=0)
        all_nobs = np.concatenate([obs_next, expert_obs_next], axis=0)
        all_acts = np.concatenate([acts, expert_acts], axis=0)
        all_nacts = np.concatenate([acts_next, expert_acts_next], axis=0)    
        all_probs = np.concatenate([path_probs, expert_probs], axis=0)    
        all_labels = np.zeros((all_obs.shape[0], 1))
        all_labels[obs.shape[0]:] = 1.0

        # Train discriminator
        for it in TrainingIterator(self.max_itrs, heartbeat=5):

            if self.n_rew_funct < self.n_value_funct:
                delta = self.n_value_funct - self.n_rew_funct
                temp = np.arange(self.n_rew_funct)
                np.random.shuffle(temp)
                rew_idxs = np.r_[temp, np.random.randint(self.n_rew_funct, size=delta)]
                val_idxs = np.arange(self.n_value_funct)
            else:
                delta = self.n_rew_funct - self.n_value_funct
                temp = np.arange(self.n_value_funct)
                np.random.shuffle(temp)
                val_idxs = np.r_[temp, np.random.randint(self.n_value_funct, size=delta)]
                rew_idxs = np.arange(self.n_rew_funct)

            for idx in range(val_idxs.shape[0]):
                i = rew_idxs[idx]
                j = val_idxs[idx]
                for single_nstep in range(1,self.max_nstep+1):
                    nobs_batch, obs_batch, nact_batch, act_batch, lprobs_batch = \
                        self.sample_batch(obs_next, obs, acts_next, acts, path_probs, batch_size=batch_size)

                    nexpert_obs_batch, expert_obs_batch, nexpert_act_batch, expert_act_batch, expert_lprobs_batch = \
                        self.sample_batch(expert_obs_next, expert_obs, expert_acts_next, expert_acts, expert_probs, batch_size=batch_size)

                    # Build feed dict
                    labels = np.zeros((batch_size*2, 1))
                    labels[batch_size:] = 1.0
                    obs_batch = np.concatenate([obs_batch, expert_obs_batch], axis=0)
                    nobs_batch = np.concatenate([nobs_batch, nexpert_obs_batch], axis=0)
                    act_batch = np.concatenate([act_batch, expert_act_batch], axis=0)
                    nact_batch = np.concatenate([nact_batch, nexpert_act_batch], axis=0)
                    lprobs_batch = np.expand_dims(np.concatenate([lprobs_batch, expert_lprobs_batch], axis=0), axis=1).astype(np.float32)
                    feed_dict = {
                        self.act_t: act_batch[:,:single_nstep],
                        self.obs_t: obs_batch[:,:single_nstep],
                        self.nobs_t: nobs_batch if self.max_nstep == single_nstep else obs_batch[:,single_nstep],
                        self.nact_t: nact_batch if self.max_nstep == single_nstep else act_batch[:,single_nstep],
                        self.labels: labels,
                        self.lprobs: lprobs_batch,
                        self.lr: lr
                        }

                    loss, _ = tf.get_default_session().run([self.loss[i][j], self.step[i][j]], feed_dict=feed_dict)
                    it.record('loss', loss)

            if self.score_discrim is False and self.score_method == 'teacher_student':
                nobs_batch, obs_batch, nact_batch, act_batch, lprobs_batch = \
                    self.sample_batch(obs_next, obs, acts_next, acts, path_probs, batch_size=batch_size)

                nexpert_obs_batch, expert_obs_batch, nexpert_act_batch, expert_act_batch, expert_lprobs_batch = \
                    self.sample_batch(expert_obs_next, expert_obs, expert_acts_next, expert_acts, expert_probs, batch_size=batch_size)

                # Build feed dict
                labels = np.zeros((batch_size*2, 1))
                labels[batch_size:] = 1.0                
                obs_batch = np.concatenate([obs_batch, expert_obs_batch], axis=0)
                nobs_batch = np.concatenate([nobs_batch, nexpert_obs_batch], axis=0)
                act_batch = np.concatenate([act_batch, expert_act_batch], axis=0)
                nact_batch = np.concatenate([nact_batch, nexpert_act_batch], axis=0)
                lprobs_batch = np.expand_dims(np.concatenate([lprobs_batch, expert_lprobs_batch], axis=0), axis=1).astype(np.float32)
                feed_dict = {
                    self.act_t: act_batch,
                    self.obs_t: obs_batch,
                    self.nobs_t: nobs_batch,
                    self.nact_t: nact_batch,
                    self.labels: labels,
                    self.lprobs: lprobs_batch,
                    self.lr: lr
                    }

                rel_loss, abs_loss, ens_loss, _ = tf.get_default_session().run([self.student_loss, self.student_absolute_loss, self.ensemble_loss, self.student_step], feed_dict=feed_dict)                

            if it.heartbeat:
                print(it.itr_message())
                mean_loss = it.pop_mean('loss')
                print('\tLoss:%f' % mean_loss)

        if logger:            
            logger.record_tabular('GCLMeanDiscrimLoss', mean_loss)
            sess = tf.get_default_session()
            if self.score_discrim is False and self.score_method == 'teacher_student':
                logger.record_tabular('GCLStudentRelativeLoss', rel_loss)
                logger.record_tabular('GCLStudentAbsoluteLoss', abs_loss)
                logger.record_tabular('GCLEnsembleLoss', ens_loss)
            else:
                e_loss, weights = sess.run([self.ensemble_loss, self.weights], feed_dict={self.act_t: all_acts, self.obs_t: all_obs, self.nobs_t: all_nobs,
                                                                    self.nact_t: all_nacts, self.labels: all_labels,
                                                                    self.lprobs: np.expand_dims(all_probs, axis=1)})
                logger.record_tabular('GCLEnsembleDiscrimLoss', e_loss)     
                # logger.record_tabular('TimeWeights', weights)

            if self.score_discrim is False and self.score_method == 'teacher_student':
                energy, logZ, dtau, s_rew, s_val, s_dtau = sess.run([self.t0_reward, self.t0_value, self.ensemble_discrim_output, \
                                                                    self.student_reward, self.student_value, self.student_discrim_output],
                                                                feed_dict={self.act_t: acts, self.obs_t: obs, self.nobs_t: obs_next,
                                                                    self.nact_t: acts_next,
                                                                self.lprobs: np.expand_dims(path_probs, axis=1)})
            else:
                energy, logZ, dtau = sess.run([self.t0_reward, self.t0_value, self.ensemble_discrim_output],
                                                feed_dict={self.act_t: acts, self.obs_t: obs, self.nobs_t: obs_next,
                                                self.nact_t: acts_next,
                                                self.lprobs: np.expand_dims(path_probs, axis=1)})                

            energy = -energy
            logger.record_tabular('GCLLogZ', np.mean(logZ))
            logger.record_tabular('GCLAverageEnergy', np.mean(energy))
            logger.record_tabular('GCLAverageLogPtau', np.mean(-energy - logZ))
            logger.record_tabular('GCLAverageLogQtau', np.mean(path_probs))
            logger.record_tabular('GCLMedianLogQtau', np.median(path_probs))
            logger.record_tabular('GCLAverageDtau', np.mean(dtau))

            if self.score_discrim is False and self.score_method == 'teacher_student':
                logger.record_tabular('GCLAverageStudentEnergy', np.mean(-s_rew))
                logger.record_tabular('GCLAverageStudentLogPtau', np.mean(s_rew - s_val))
                logger.record_tabular('GCLAverageStudentDtau', np.mean(s_dtau))           


            if self.score_discrim is False and self.score_method == 'teacher_student':            
                energy, logZ, dtau, s_rew, s_val, s_dtau = sess.run([self.t0_reward, self.t0_value, self.ensemble_discrim_output, \
                                                                    self.student_reward, self.student_value, self.student_discrim_output],
                        feed_dict={self.act_t: expert_acts, self.obs_t: expert_obs, self.nobs_t: expert_obs_next,
                                        self.nact_t: expert_acts_next,
                                        self.lprobs: np.expand_dims(expert_probs, axis=1)})
            else:
                energy, logZ, dtau = sess.run([self.t0_reward, self.t0_value, self.ensemble_discrim_output],
                        feed_dict={self.act_t: expert_acts, self.obs_t: expert_obs, self.nobs_t: expert_obs_next,
                                        self.nact_t: expert_acts_next,
                                        self.lprobs: np.expand_dims(expert_probs, axis=1)})                

            energy = -energy
            logger.record_tabular('GCLAverageExpertEnergy', np.mean(energy))
            logger.record_tabular('GCLAverageExpertLogPtau', np.mean(-energy - logZ))
            logger.record_tabular('GCLAverageExpertLogQtau', np.mean(expert_probs))
            logger.record_tabular('GCLMedianExpertLogQtau', np.median(expert_probs))
            logger.record_tabular('GCLAverageExpertDtau', np.mean(dtau))

            if self.score_discrim is False and self.score_method == 'teacher_student':
                logger.record_tabular('GCLAverageStudentExpertEnergy', np.mean(-s_rew))
                logger.record_tabular('GCLAverageStudentExpertLogPtau', np.mean(s_rew - s_val))
                logger.record_tabular('GCLAverageStudentExpertDtau', np.mean(s_dtau)) 

        return mean_loss

    def _compute_path_probs_modified(self, paths, pol_dist_type=None, insert=True,
                            insert_key='a_logprobs'):
        """
        Returns a N x T matrix of action probabilities
        """
        if insert_key in paths[0]:
            return np.array([path[insert_key] for path in paths])

        if pol_dist_type is None:
            # try to  infer distribution type
            path0 = paths[0]
            if 'log_std' in path0['agent_infos']:
                pol_dist_type = DIST_GAUSSIAN
            elif 'prob' in path0['agent_infos']:
                pol_dist_type = DIST_CATEGORICAL
            else:
                raise NotImplementedError()

        # compute path probs
        Npath = len(paths)
        actions = [path['actions'][0] for path in paths]
        if pol_dist_type == DIST_GAUSSIAN:
            params = [(path['agent_infos']['mean'], path['agent_infos']['log_std']) for path in paths]
            path_probs = [gauss_log_pdf(params[i], actions[i]) for i in range(Npath)]
        elif pol_dist_type == DIST_CATEGORICAL:
            params = [(path['agent_infos']['prob'],) for path in paths]
            path_probs = [categorical_log_pdf(params[i], actions[i]) for i in range(Npath)]
        else:
            raise NotImplementedError("Unknown distribution type")

        if insert:
            for i, path in enumerate(paths):
                path[insert_key] = path_probs[i]

        return np.array(path_probs)

    def eval(self, paths, **kwargs):
        """
        Return bonus
        """
        if self.score_discrim:
            self._compute_path_probs_modified(paths, insert=True)
            obs, obs_next, acts, path_probs = self.extract_paths(paths, keys=('multi_observations', 'observations_next', 'multi_actions', 'a_logprobs'))
            path_probs = np.expand_dims(path_probs, axis=1)
            scores = tf.get_default_session().run(self.ensemble_discrim_output,
                                              feed_dict={self.act_t: acts, self.obs_t: obs,
                                                         self.nobs_t: obs_next,
                                                         self.lprobs: path_probs})
            score = np.log(np.maximum(scores,1e-8)) - np.log(np.maximum(1 - scores,1e-8))
            score = score[:,0]
        else:
            if self.score_method == 'sample_rewards':
                obs, acts = self.extract_paths(paths, keys=('multi_observations', 'multi_actions'))
                sampled_idx = np.random.randint(self.n_rew_funct)
                reward = tf.get_default_session().run(self.reward[sampled_idx],
                                                    feed_dict={self.act_t: acts[:,:1], self.obs_t: obs[:,:1]})
                score = reward[:,0,0]
            elif self.score_method == 'average_rewards':
                obs, acts = self.extract_paths(paths, keys=('multi_observations', 'multi_actions'))
                reward = tf.get_default_session().run(self.avg_reward,
                                                    feed_dict={self.act_t: acts[:,:1], self.obs_t: obs[:,:1]})
                score = reward[:,0,0]
            elif self.score_method == 'teacher_student':
                obs, acts = self.extract_paths(paths, keys=('multi_observations', 'multi_actions'))
                reward = tf.get_default_session().run(self.student_reward,
                                                    feed_dict={self.act_t: acts[:,:1], self.obs_t: obs[:,:1]})
                score = reward[:,0]                
        return self.unpack(score, paths)

