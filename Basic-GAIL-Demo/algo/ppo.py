import tensorflow as tf
import copy


class PPOTrain:
    def __init__(self, Policy, Old_Policy, gamma=0.95, clip_value=0.2, c_1=1, c_2=0.01):
        """
        :param Policy:
        :param Old_Policy:
        :param gamma:
        :param clip_value:
        :param c_1: parameter for value difference
        :param c_2: parameter for entropy bonus
        """

        self.Policy = Policy
        self.Old_Policy = Old_Policy
        self.gamma = gamma

        pi_trainable = self.Policy.get_trainable_variables()
        old_pi_trainable = self.Old_Policy.get_trainable_variables()

        # assign_operations for policy parameter values to old policy parameters
        with tf.compat.v1.variable_scope('assign_op'):
            self.assign_ops = []
            for v_old, v in zip(old_pi_trainable, pi_trainable):
                self.assign_ops.append(tf.compat.v1.assign(v_old, v))  # v_old <- v

        # inputs for train_op
        with tf.compat.v1.variable_scope('train_inp'):
            self.actions = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None], name='actions')
            self.rewards = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None], name='rewards')
            self.v_preds_next = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None], name='v_preds_next')
            self.gaes = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None], name='gaes')

        act_probs = self.Policy.act_probs
        act_probs_old = self.Old_Policy.act_probs

        # probabilities of actions which agent took with policy
        act_probs = act_probs * tf.one_hot(indices=self.actions, depth=act_probs.shape[1])
        act_probs = tf.reduce_sum(act_probs, axis=1)

        # probabilities of actions which agent took with old policy
        act_probs_old = act_probs_old * tf.one_hot(indices=self.actions, depth=act_probs_old.shape[1])
        act_probs_old = tf.reduce_sum(act_probs_old, axis=1)

        with tf.compat.v1.variable_scope('loss'):  # GD 방법이 아니므로 단순하게 loss function 의 미분값 구하는게 아니다. -> 제약조건 내에 목적함수가 최대화되게 만드는것
            # construct computation graph for loss_clip (클립된 loss 계산)
            # ratios = tf.divide(act_probs, act_probs_old)
            ratios = tf.exp(tf.math.log(tf.clip_by_value(act_probs, 1e-10, 1.0))
                            - tf.math.log(tf.clip_by_value(act_probs_old, 1e-10, 1.0)))  # 이전 정책과 새 정책, 둘 사이의 변화량 비율
            clipped_ratios = tf.clip_by_value(ratios, clip_value_min=1 - clip_value, clip_value_max=1 + clip_value)  # 1이내의 범위로 클립
            loss_clip = tf.minimum(tf.multiply(self.gaes, ratios), tf.multiply(self.gaes, clipped_ratios))   # 둘 중 더 작은 값을 loss로 사용
            loss_clip = tf.reduce_mean(loss_clip)
            tf.compat.v1.summary.scalar('loss_clip', loss_clip)

            # construct computation graph for loss of entropy bonus (엔트로피 계산)
            entropy = -tf.reduce_sum(self.Policy.act_probs *
                                     tf.math.log(tf.clip_by_value(self.Policy.act_probs, 1e-10, 1.0)), axis=1)
            entropy = tf.reduce_mean(entropy, axis=0)  # mean of entropy of pi(obs)
            tf.compat.v1.summary.scalar('entropy', entropy)

            # construct computation graph for loss of value function (value function의 squared loss 계산)
            v_preds = self.Policy.v_preds
            loss_vf = tf.math.squared_difference(self.rewards + self.gamma * self.v_preds_next, v_preds)
            loss_vf = tf.reduce_mean(loss_vf)
            tf.compat.v1.summary.scalar('value_difference', loss_vf)

            # construct computation graph for loss
            loss = loss_clip - c_1 * loss_vf + c_2 * entropy

            # minimize -loss == maximize loss
            loss = -loss
            tf.compat.v1.summary.scalar('total', loss)

        self.merged = tf.compat.v1.summary.merge_all()
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=5e-5, epsilon=1e-5)
        self.gradients = optimizer.compute_gradients(loss, var_list=pi_trainable)
        self.train_op = optimizer.minimize(loss, var_list=pi_trainable)

    def train(self, obs, actions, gaes, rewards, v_preds_next):
        tf.compat.v1.get_default_session().run(self.train_op, feed_dict={self.Policy.obs: obs,
                                                               self.Old_Policy.obs: obs,
                                                               self.actions: actions,
                                                               self.rewards: rewards,
                                                               self.v_preds_next: v_preds_next,
                                                               self.gaes: gaes})

    def get_summary(self, obs, actions, gaes, rewards, v_preds_next):
        return tf.compat.v1.get_default_session().run(self.merged, feed_dict={self.Policy.obs: obs,
                                                                    self.Old_Policy.obs: obs,
                                                                    self.actions: actions,
                                                                    self.rewards: rewards,
                                                                    self.v_preds_next: v_preds_next,
                                                                    self.gaes: gaes})

    def assign_policy_parameters(self):
        # assign policy parameter values to old policy parameters
        return tf.compat.v1.get_default_session().run(self.assign_ops)

    def get_gaes(self, rewards, v_preds, v_preds_next):  # PPO의 목적함수
        deltas = [r_t + self.gamma * v_next - v for r_t, v_next, v in zip(rewards, v_preds_next, v_preds)]
        # calculate generative advantage estimator(lambda = 1) = A_t, see ppo paper eq(11)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(gaes) - 1)):  # is T-1, where T is time step which run policy  -> T 타임스텝 동안 모은 advantage --> 하나의 미니배치로 이용
            gaes[t] = gaes[t] + self.gamma * gaes[t + 1]
        return gaes

    def get_grad(self, obs, actions, gaes, rewards, v_preds_next):
        return tf.compat.v1.get_default_session().run(self.gradients, feed_dict={self.Policy.obs: obs,
                                                                       self.Old_Policy.obs: obs,
                                                                       self.actions: actions,
                                                                       self.rewards: rewards,
                                                                       self.v_preds_next: v_preds_next,
                                                                       self.gaes: gaes})
