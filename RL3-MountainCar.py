import gym
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

import copy
import time

env = gym.make('MountainCar-v0')
env.reset()
# values >200 are useless because in gym, mountain car returns true after 200 actions, or if reached uphill
goal_steps = 200
score_requirement = -197
intial_games = 10000
play_games = 100


# play random games and gather data for NN
def model_data_preparation():
    training_data = []
    accepted_scores = [] # scores above 'score_requirement' are kept
    time_start = time.time()
    for game_index in range(intial_games):
        score = 0
        game_memory = []

        # to track the work
        if (game_index % 100 == 0):
            print("data gathering: ", game_index, " of ", intial_games, "\n time: ", str( time.time() - time_start))

        previous_observation = env.reset()  # initial observation #[]
        for step_index in range(goal_steps):
            action = env.action_space.sample() #random action
            observation, reward, done, info = env.step(action)
            env.render()# just to see how agent is playing

            #if len(previous_observation) > 0:
            game_memory.append([previous_observation, action])

            previous_observation = observation
            # subtle change in rewarding method
            if observation[0] > -0.2:
                reward = 1

            score += reward
            if done:
                break

        if score >= score_requirement:
            accepted_scores.append(score)
            # training_data.extend(game_memory)
            for data in game_memory:
                if data[1] == 1:
                    output = [0, 1, 0]
                elif data[1] == 0:
                    output = [1, 0, 0]
                elif data[1] == 2:
                    output = [0, 0, 1]
                training_data.append([data[0], output])

        env.reset()

    print(accepted_scores)
    print("game_memory_length: ", len(game_memory))
    print("training_data length: ", len(training_data))
    print(" END OF TRAINING DATA GATHERING ")
    return training_data


# Build Neural Network Model
def build_model(input_size, output_size):
    model = Sequential()
    model.add(Dense(128, input_dim=input_size, activation='relu'))
    model.add(Dense(52, activation='relu'))
    model.add(Dense(output_size, activation='linear'))
    model.compile(loss='mse', optimizer=Adam())
    return model


def train_model(training_data):
    X_obs = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]))
    X_act = np.array([i[1] for i in training_data]).reshape(-1, len(training_data[0][1]))
    model = build_model(input_size=len(X_obs[0]), output_size=len(X_act[0]))

    model.fit(X_obs, X_act, epochs=5)
    # Save the model
    model.save('trained-models/mountaincar.h5')
    return model

def play_AI():
    scores = []
    choices = []
    for each_game in range(play_games):
        print("playing game: ", each_game,  " of ", play_games)
        score = 0
        game_memory = []
        prev_obs = env.reset() #[] CAN I RESET EACH TIME?
        for step_index in range(goal_steps):
            env.render()
            # check if this works
            # if len(prev_obs) == 0:
            #     action = env.action_space.sample() #random action #random.randrange(0, 2)
            # else:
            action = np.argmax(trained_model.predict(prev_obs.reshape(-1, len(prev_obs)))[0])

            choices.append(action)
            new_observation, reward, done, info = env.step(action)
            prev_obs = new_observation
            game_memory.append([new_observation, action])
            score += reward
            if done:
                break
        #env.reset()
        scores.append(score)


    print(scores)
    print('Average Score:', sum(scores) / len(scores))
    print('choice 1 (moving backward): {}  choice 0 (No movement): {} choice 2 (moving forward): {}'.format(choices.count(1) / len(choices), choices.count(0) / len(choices),
                                                        choices.count(2) / len(choices)))


if __name__ == "__main__":

    training_data = model_data_preparation()
    trained_model = train_model(training_data)
    play_AI()