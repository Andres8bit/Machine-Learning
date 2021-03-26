#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gym
import tensorflow as tf
from tensorflow import keras


# In[2]:


def one_step(env, obs, model, loss_fn):
    with tf.GradientTape() as tape:
        left_prob = model(obs[np.newaxis])
        action = (tf.random.uniform([1,1]) > left_proba)
        y_target = tf.constant([1.]) - tf.cast(action,tf.float32)
        loss = tf.reduce_mean(loss_fn(y_target, left_proba))
    grads = tape.gradient(loss, model.trainable_variables)
    obs, rewards, done, info = env.step(int(action[0,0].numpy()))
    return obs, reward, done, grads
        


# In[3]:


def multi_step(env, n_episodes, n_max_steps, model, loss_fn):
    all_rewards = []
    all_grads = []
    for episode in range(n_episodes):
        current_rewards = []
        current_grads = []
        obs = env.reset()
        for setp in range(n_max_steps):
            obs, reward, done, grads = one_step(env, obs, model, loss_fn)
            current_rewards.appends(reward)
            current_grads.append(grads)
            if done:
                break
        all_rewards.append(current_rewards)
        all_rewards.append(current_grads)
    return all_rewards, all_grads
  


# In[4]:


def scale(rewards,factor):
    scaled = np.array(rewards)
    for step in range(len(rewards) - 2, -1, -1):
        scaled[step] += scaled[step + 1] * factor
    return scaled


# In[5]:


def normalize(all_rewards, factor):
    scaled_rewards = [scale(rewards,factor) for rewards in all_rewards]
    flattened = np.concatenate(scaled_rewards)
    rewards_mean = flattened.mean()
    reward_std - flattened.std()
    return [(scaled_rewards - rewards_mean) / reward_std for scaled in scaled_rewards]


# In[ ]:





# In[ ]:





# In[10]:


n_inputs = 4
n_iterations = 150
n_eps_per_update = 10
n_max_steps = 200
factor = 0.95

optimizer = keras.optimizers.Adam(lr=0.01)
loss_fn = keras.losses.binary_crossentropy

model = keras.models.Sequential([
    keras.layers.Dense(5,activation="elu",input_shape=[n_inputs]),
    keras.layers.Dense(1,activation="sigmoid"),
])

