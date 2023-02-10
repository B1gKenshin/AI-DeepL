import gym
# import learntraing
##pip install rl-agents==0.1.1
import keras
from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten,Conv2D
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy,EpsGreedyQPolicy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# session,_ = ti.getSession()
# df = ti.downloadData('BTCUSDT',session)
# df.to_csv('IA\Datasheets\Data2000_indicators.csv')


df = pd.read_csv('IA\Datasheets\Data2000.csv')

env = gym.make('learn_trading:learnstocks-v0', df=df, frame_bound=(0,100), window_size=30,new_frame = 50)
env.reset()
#setting up our environment for training 
nb_actions = env.action_space.n
nb_observations = env.observation_space.shape

model = Sequential()
model.add(Flatten(input_shape=(10,)+nb_observations))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(nb_actions, activation='linear'))

memory = SequentialMemory(limit=1000000, window_length=10)
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(),
                                attr='eps',
                                value_max=1.0,
                                value_min=0.1,
                                value_test=0.05,
                                nb_steps=50000)

dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=30,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
TRAIN = True
if TRAIN:
    dqn.fit(env,nb_steps=50000,visualize=False,verbose=1)
    dqn.save_weights('IA\Trainded_datas\Stocks_all_CNN.h5f',overwrite=True)
    dqn.test(env,nb_episodes=100,visualize=False)
if not TRAIN:
    dqn.load_weights('IA\Trainded_datas\Stocks_all_CNN.h5f')
    dqn.test(env,nb_episodes=100,visualize=True)
env.close()
