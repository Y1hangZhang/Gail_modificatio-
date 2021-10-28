import argparse
import gym
import numpy as np
import tensorflow as tf
from network_models.policy_net import Policy_net
from algo.ppo import PPOTrain

#parse_args() 호출 시 저장되어 사용되는 프로그램 매개변수 정보 추가
def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', help='log directory', default='log/train/ppo')
    parser.add_argument('--savedir', help='save directory', default='trained_models/ppo')
    parser.add_argument('--gamma', default=0.95, type=float)
    parser.add_argument('--iteration', default=int(1e4), type=int)
    return parser.parse_args()


def main(args):  # PPO오로 policy 훈련, old_policy는 무슨 용도인지 모르겠다.
    env = gym.make('CartPole-v0') # 게임 환경 만들기
    env.seed(0) # 환경의 난수 시드 설정
    ob_space = env.observation_space # 관찰 공간  #스타에선 미정
    Policy = Policy_net('policy', env) # 전략 네트워크
    Old_Policy = Policy_net('old_policy', env)   # ppo알고리즘 때문에 필요
    PPO = PPOTrain(Policy, Old_Policy, gamma=args.gamma) # PPO 알고리즘 교육
    saver = tf.compat.v1.train.Saver()

    with tf.compat.v1.Session() as sess:
        writer = tf.compat.v1.summary.FileWriter(args.logdir, sess.graph)
        sess.run(tf.compat.v1.global_variables_initializer())
        obs = env.reset() # 환경을 재설정하고 초기 상태를 나타내는 값을 반환합니다.
        success_num = 0

        for iteration in range(args.iteration):
            observations = []
            actions = []
            rewards = []
            v_preds = []
            episode_length = 0
            while True:  # run policy RUN_POLICY_STEPS which is much less than episode length
                episode_length += 1
                obs = np.stack([obs]).astype(dtype=np.float32)  # prepare to feed placeholder Policy.obs
                act, v_pred = Policy.act(obs=obs, stochastic=True)

                act = act.item()
                v_pred = v_pred.item()
                # 환경과 상호 작용하고 다음 상태의 현재 보상을 반환하고 에피소드가 끝났는지 여부를 표시합니다.
                next_obs, reward, done, info = env.step(act)

                observations.append(obs)
                actions.append(act)
                rewards.append(reward)
                v_preds.append(v_pred)

                if done:
                    next_obs = np.stack([next_obs]).astype(dtype=np.float32)  # prepare to feed placeholder Policy.obs
                    _, v_pred = Policy.act(obs=next_obs, stochastic=True)
                    v_preds_next = v_preds[1:] + [v_pred.item()]
                    obs = env.reset() # 환경을 재설정하고 초기 상태를 나타내는 값을 반환합니다.
                    break
                else:
                    obs = next_obs

            writer.add_summary(tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag='episode_length', simple_value=episode_length)])
                               , iteration)
            writer.add_summary(tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag='episode_reward', simple_value=sum(rewards))])
                               , iteration)

            if sum(rewards) >= 195:
                success_num += 1
                if success_num >= 100:
                    saver.save(sess, args.savedir+'/model.ckpt')
                    print('Clear!! Model saved.')
                    break
            else:
                success_num = 0

            gaes = PPO.get_gaes(rewards=rewards, v_preds=v_preds, v_preds_next=v_preds_next)

            # convert list to numpy array for feeding tf.placeholder
            observations = np.reshape(observations, newshape=(-1,) + ob_space.shape)
            actions = np.array(actions).astype(dtype=np.int32)
            gaes = np.array(gaes).astype(dtype=np.float32)
            gaes = (gaes - gaes.mean()) / gaes.std()  # 정규화(?)
            rewards = np.array(rewards).astype(dtype=np.float32)
            v_preds_next = np.array(v_preds_next).astype(dtype=np.float32)

            PPO.assign_policy_parameters()

            inp = [observations, actions, gaes, rewards, v_preds_next]

            # train
            for epoch in range(6):
                # sample indices from [low, high)
                sample_indices = np.random.randint(low=0, high=observations.shape[0], size=32)
                sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]  # sample training data
                PPO.train(obs=sampled_inp[0],
                          actions=sampled_inp[1],
                          gaes=sampled_inp[2],
                          rewards=sampled_inp[3],
                          v_preds_next=sampled_inp[4])

            summary = PPO.get_summary(obs=inp[0],
                                      actions=inp[1],
                                      gaes=inp[2],
                                      rewards=inp[3],
                                      v_preds_next=inp[4])

            writer.add_summary(summary, iteration)
        writer.close()


if __name__ == '__main__':
    args = argparser()
    main(args)
