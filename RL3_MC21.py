import gym
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam


import time
import os


env = gym.make('MountainCar-v0')
env.reset()
# values >200 are useless because in gym, mountain car returns true after 200 actions, or if reached uphill
goal_steps = 200
score_requirement = -196
intial_games = 5000 # previous trained by 7000
play_games = 20
training_epoch = 15


# play random games and gather data for NN, Saves the results in csv file
def model_data_preparation():
    training_data = []
    accepted_scores = []  # scores above 'score_requirement' are kept
    time_start = time.time()
    for game_index in range(intial_games):
        score = 0
        game_memory = []

        # to track the work
        if (game_index % 100 == 0):
            print("data gathering: ", game_index, " of ", intial_games, "\n time: ", str(time.time() - time_start))

        previous_observation = env.reset()  # initial observation #[]
        for step_index in range(goal_steps):
            action = env.action_space.sample()  # random action
            observation, reward, done, info = env.step(action)
            #print("Observation: ", observation, "Action: ", action)
            env.render()  # just to see how agent is playing

            # if len(previous_observation) > 0:
            game_memory.append([previous_observation, action])

            previous_observation = observation
            # subtle change in rewarding method
            if observation[0] > -0.2:
                reward = 1
            # elif observation[0] > -0.4:
            #     reward = 1

            score += reward
            if done:
                break

        if score >= score_requirement:
            accepted_scores.append(score)
            #training_data.extend(game_memory)
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
    #print("game_memory_length: ", len(game_memory))
    print("training_data length: ", len(training_data))
    print(" END OF TRAINING DATA GATHERING ")
    try:
        # save to csv file for later use
        data = np.asarray(training_data)
        file_name = "training_data" + str(time.time()) + ".csv"
        np.savetxt(file_name, data,fmt='%s', delimiter=',') #'%s'
    except Exception as e:
        s = str(e)
        print(" !!!! Attempt to save training data Failed !!!! ", s)
    return training_data


# Build Neural Network Model
# Model with 3 hidden layers and mse loss function and linear activation in the output as our actions are discrete, and linear is satisfactory
def build_model(input_size, output_size):
    model = Sequential()
    model.add(Dense(52, input_dim=input_size, activation='relu'))
    model.add(Dense(36, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(output_size, activation='linear'))

    model.compile(loss='mse', optimizer=Adam())
    return model


def train_model(training_data):
    start_time = time.time()
    X_obs = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]))
    X_act = np.array([i[1] for i in training_data]).reshape(-1, len(training_data[0][1]))
    #print(X_obs)
    #print(X_act)
    # model = build_model(input_size=len(X_obs[0]), output_size=len(X_act[0]))

    model = build_model(input_size=len(X_obs[0]), output_size=len(X_act[0]))
    model.fit(X_obs, X_act, epochs=training_epoch)
    # Save the model
    try:
        model.save('mountaincar3.h5')
    except Exception as e:
        print(" Model Not Saved ... " , str(e))

    print(" TIME to Train the model: ", str(time.time() - start_time))
    return model


def play_AI(trained_model: Sequential):
    scores = []
    selected_actions = []
    Success_Times = 0
    for each_game in range(play_games):
        print("playing game: ", each_game, " of ", play_games)
        score = 0
        game_memory = []
        prev_obs = env.reset()  # [] CAN I RESET EACH TIME?
        for step_index in range(goal_steps):
            env.render()
            # check if this works
            # if len(prev_obs) == 0:
            #     action = env.action_space.sample() #random action #random.randrange(0, 2)
            # else:
            action = np.argmax(trained_model.predict(prev_obs.reshape(-1, len(prev_obs)))[0])

            selected_actions.append(action)
            new_observation, reward, done, info = env.step(action)
            prev_obs = new_observation
            game_memory.append([new_observation, action])
            score += reward
            if done:
                if new_observation[0] >= 0.5:
                    print(" Cart reached the Goal ")
                    Success_Times += 1
                break
            # env.reset()
            scores.append(score)

    env.close()
    print(scores)
    print('Average Score:', sum(scores) / len(scores))
    print('choice 1 (moving backward): {}  choice 0 (No movement): {} choice 2 (moving forward): {}'.format(
        selected_actions.count(1) / len(selected_actions), selected_actions.count(0) / len(selected_actions),
        selected_actions.count(2) / len(selected_actions)))
    print(" The cart reached the goal ", Success_Times, "Times ")

def main(path_model="mountaincar2.h5"):
    # try:
    #     # check if data is empty
    #     #training_data = np.loadtxt('training_data.csv', delimiter=',')
    #     training_data = np.genfromtxt('training_data.csv', delimiter=',',dtype=None)
    #     #print(type(training_data[0]), type(training_data[1]), type(training_data[0][0]))
    #     print(training_data)
    #     print(" Loaded data successfully ******************** ")
    # except Exception as e:
    #     training_data = model_data_preparation()
    #     print(" HAD TO Gather data from scratch, Error: ", str(e))

    # training_data = model_data_preparation()
    try:
        if os.path.isfile(path_model):
            model = load_model(path_model)
            print(" Model Loaded Successfully ")
        else:
            training_data = model_data_preparation()
            model = train_model(training_data)
            print(" Model built from Scratch. And data gathered from scratch ")
    except Exception as e:
        training_data = model_data_preparation()
        model = train_model(training_data)
        print(" Model built from Scratch. Error: ", str(e))

    #trained_model = model
    play_AI(model)

if __name__ == "__main__":

    path_trained_model = "mountaincar2.h5"
    print("hello")
    main(path_trained_model)

