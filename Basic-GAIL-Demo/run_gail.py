import argparse
import gym
import numpy as np
import tensorflow as tf
from network_models.policy_net import Policy_net
from network_models.discriminator import Discriminator
from algo.ppo import PPOTrain

def argparser():  # 하이퍼 파라미터 불러오기, 저장 경로 설정
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', help='log directory', default='log/train/gail')
    parser.add_argument('--savedir', help='save directory', default='trained_models/gail')
    parser.add_argument('--gamma', default=0.95)
    parser.add_argument('--iteration', default=int(1e4))
    return parser.parse_args()


def main(args):
    env = gym.make('CartPole-v0')
    env.seed(0)
    ob_space = env.observation_space   # state 관측 환경 받는다
    Policy = Policy_net('policy', env)   # policy 로 expert obs,act을 받아서 적용하고 expert policy가 된다
    Old_Policy = Policy_net('old_policy', env)  # 구해야하는 agent policy  -> output: parameters
    PPO = PPOTrain(Policy, Old_Policy, gamma=args.gamma)
    D = Discriminator(env)  

    # 전문가의 관찰 및 액션 불러오기
    expert_observations = np.genfromtxt('trajectory/observations.csv')
    expert_actions = np.genfromtxt('trajectory/actions.csv', dtype=np.int32)

    saver = tf.compat.v1.train.Saver()

    with tf.compat.v1.Session() as sess:
        writer = tf.compat.v1.summary.FileWriter(args.logdir, sess.graph)
        sess.run(tf.compat.v1.global_variables_initializer())

        obs = env.reset() # obs -> [4], 예시 -> [-0.04456399  0.04653909  0.01326909 -0.02099827] , obs.shape = (4)
        success_num = 0

        for iteration in range(args.iteration):
            observations = []
            actions = []
            rewards = []
            v_preds = []
            run_policy_steps = 0

            while True:
                run_policy_steps += 1
                obs = np.stack([obs]).astype(dtype=np.float32)
                act, v_pred = Policy.act(obs = obs,stochastic = True)

                act = act.item()
                v_pred = v_pred.item()

                next_obs,reward,done,info = env.step(act)  # old_policy 가 데이터를 수집하는 것

                observations.append(obs)
                actions.append(act)
                rewards.append(reward)
                v_preds.append(v_pred)

                if done:  # 게임 terminal
                    next_obs = np.stack([next_obs]).astype(dtype=np.float32)
                    _, v_pred = Policy.act(obs=next_obs, stochastic=True)
                    v_preds_next = v_preds[1:] + [v_pred.item()]
                    obs = env.reset()
                    break
                else:
                    obs = next_obs

            writer.add_summary(tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag='episode_length', simple_value=run_policy_steps)])
                               , iteration)
            writer.add_summary(tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag='episode_reward', simple_value=sum(rewards))])
                               , iteration)

            if sum(rewards) >= 195:  # reward, 성공횟수 기준으로 expert 와 비슷하다고 생각하고 끝낸다
                success_num += 1
                if success_num >= 100:  # Policy의 파라미터 저장
                    saver.save(sess, args.savedir + '/model.ckpt')
                    print('Clear!! Model saved.')
                    break
            else:
                success_num = 0
                
            # observations <- ob_space 가 stack 되어있는것
            observations = np.reshape(observations,newshape=[-1] + list(ob_space.shape))  # 예시> (13,1,4) -> (13,4)
            actions = np.array(actions).astype(dtype = np.int32)

            for i in range(2):     # Question 하이퍼 파라미터, 아니면 expert는 1이 되도록, learner은 0이 되도록 훈련 ?
                D.train(expert_s = expert_observations,
                        expert_a = expert_actions,
                        agent_s = observations,
                        agent_a = actions)  # discriminator 훈련 -> 1. expert와 policy의 행동이 , 둘을 잘 구별하도록 discriminator 훈련


            d_rewards = D.get_rewards(agent_s=observations,agent_a = actions)   # 얼마나 전문가와 행동이 비슷한지 결과값이 reward
            d_rewards = np.reshape(d_rewards,newshape=[-1]).astype(dtype=np.float32)

            gaes = PPO.get_gaes(rewards=d_rewards, v_preds=v_preds, v_preds_next=v_preds_next)   # policy 와 old policy를 비교하는 cost function
            gaes = np.array(gaes).astype(dtype=np.float32)  # pre processing
            # gaes = (gaes - gaes.mean()) / gaes.std()   # 원래 이렇게 되어 있었다
            v_preds_next = np.array(v_preds_next).astype(dtype=np.float32)

            # train policy --> (에이전트의 행동이 전문가와 비슷하게 되도록 policy 업데이트)
            inp = [observations, actions, gaes, d_rewards, v_preds_next]  # input
            PPO.assign_policy_parameters()  #old_policy 파라미터를 policy 파라미터로 업데이트
            for epoch in range(6):
                sample_indices = np.random.randint(low=0, high=observations.shape[0],
                                                   size=32)  # indices are in [low, high)  # training data 에서 샘플하는 랜덤 변수
                sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]  # sample training data
                PPO.train(obs=sampled_inp[0],
                          actions=sampled_inp[1],
                          gaes=sampled_inp[2],
                          rewards=sampled_inp[3],
                          v_preds_next=sampled_inp[4])
    # policy가 환경에서 적용되어 데이터를 수집하고, old_policy 와 policy 사이의 차를 다시 구해서  old_policy 업데이트,
    # policy, old_policy 둘다 업데이트(policy는 자체적으로 expert data와 유사한 결과내게 업데이트),
    # old_policy 파라미터 가져오기 (old policy에 파라미터를 업데이트할 데이터를 제공, old_policy 업데이트)

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