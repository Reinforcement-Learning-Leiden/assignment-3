import gym
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam

import time
import os
import matplotlib.pyplot as plt

env = gym.make('MountainCar-v0')
env.reset()
# values >200 are useless because in gym, mountain car returns true after 200 actions, or if reached uphill
goal_steps = 200
score_requirement = -196
intial_games = 7000  # previous trained by 7000
test_games = 700
play_games = 100
training_epoch = 15


# play random games and gather data for NN, Saves the results in csv file
def model_data_preparation(policy='2'):
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
            # print("Observation: ", observation, "Action: ", action)
            env.render()  # just to see how agent is playing

            # if len(previous_observation) > 0:
            game_memory.append([previous_observation, action])

            previous_observation = observation
            # subtle change in rewarding method
            if policy == '2':
                if observation[0] > -0.2:
                    reward = 1
            # elif observation[0] > -0.4:
            #     reward = 1

            score += reward
            if done:
                break

        if score >= score_requirement:
            accepted_scores.append(score)
            # training_data.extend(game_memory)
            for data in game_memory:
                if data[1] == 1:
                    output = np.asarray([0, 1, 0])
                elif data[1] == 0:
                    output = np.asanyarray([1, 0, 0])
                elif data[1] == 2:
                    output = np.asarray([0, 0, 1])
                training_data.append([data[0], output])
                # print("type of training_data cells: ", type(training_data[0]))
                # print("type of training_data cells cells: ", type(training_data[0][0]), type(training_data[0][1]))

        env.reset()

    print(accepted_scores)
    # print("game_memory_length: ", len(game_memory))
    print("training_data length: ", len(training_data))
    print(" END OF TRAINING DATA GATHERING ")
    try:
        # save to csv file for later use
        data = np.asarray(training_data)
        file_name = "training_data" + str(time.time()) + ".csv"
        np.savetxt(file_name, data, fmt='%s', delimiter=',')  # '%s'
    except Exception as e:
        s = str(e)
        print(" !!!! Attempt to save training data Failed !!!! ", s)
    return training_data


def test_data_preparation():
    test_data = []

    time_start = time.time()
    for test_index in range(test_games):
        score = 0
        test_memory = []

        # to track the work
        if (test_index % 100 == 0):
            print("test data gathering: ", test_index, " of ", test_games, "\n time: ", str(time.time() - time_start))

        previous_observation = env.reset()  # initial observation #[]
        for step_index in range(goal_steps):
            action = env.action_space.sample()  # random action
            observation, reward, done, info = env.step(action)
            # print("Observation: ", observation, "Action: ", action)
            env.render()  # just to see how agent is playing

            # if len(previous_observation) > 0:
            test_memory.append([previous_observation, action])

            previous_observation = observation
            # subtle change in rewarding method

            score += reward
            if done:
                break


        for data in test_memory:
            if data[1] == 1:
               output = np.asarray([0, 1, 0])
            elif data[1] == 0:
               output = np.asanyarray([1, 0, 0])
            elif data[1] == 2:
               output = np.asarray([0, 0, 1])
            test_data.append([data[0], output])
                # print("type of training_data cells: ", type(training_data[0]))
                # print("type of training_data cells cells: ", type(training_data[0][0]), type(training_data[0][1]))

        env.reset()


    # print("game_memory_length: ", len(game_memory))
    print("test_data length: ", len(test_data))
    print(" END OF TEST DATA GATHERING ")

    return test_data


# Build Neural Network Model
# Model with 3 hidden layers and mse loss function and linear activation in the output as our actions are discrete, and linear is satisfactory
def build_model(input_size, output_size):
    model = Sequential()
    print("Inside build model. Input size= ", input_size, " output size= ", output_size)
    model.add(Dense(52, input_dim=input_size, activation='relu'))
    model.add(Dense(36, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(output_size, activation='linear'))

    model.compile(loss='mse', optimizer=Adam())
    return model


def train_model(training_data, path_for_save='mountaincar0.h5'):
    start_time = time.time()
    if len(training_data) == 0:
        print("No data to train the model. Exiting program.")
        exit(1)
    
    X_obs = np.array([i[0] for i in training_data]).reshape(-1, 2)  # len(training_data[0][0])
    X_act = np.array([i[1] for i in training_data]).reshape(-1, 3)  # len(training_data[0][1])

    model = build_model(input_size=len(X_obs[0]), output_size=len(X_act[0]))
    model.fit(X_obs, X_act, epochs=training_epoch)
    # Save the model
    try:
        model.save(path_for_save)
    except Exception as e:
        print(" Model Not Saved ... ", str(e))

    print(" TIME to Build and Train the model from availbale data: ", str(time.time() - start_time))
    return model


def play_AI(trained_model: Sequential):
    scores = []
    selected_actions = []
    Success_Times = 0
    reached_positions = []
    for each_game in range(play_games):  # number of episodes
        print("playing game: ", each_game + 1, " of ", play_games)
        score = 0
        game_memory = []
        prev_obs = env.reset()  #
        for step_index in range(goal_steps):
            env.render()

            action = np.argmax(trained_model.predict(prev_obs.reshape(-1, len(prev_obs)))[0])

            selected_actions.append(action)
            new_observation, reward, done, info = env.step(action)
            prev_obs = new_observation
            game_memory.append([new_observation, action])
            score += reward
            if done:
                reached_positions.append(new_observation[0])
                if new_observation[0] >= 0.5:
                    print(" Cart reached the Goal with total reward of ", score)
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

    data_test = test_data_preparation()
    print(model_evaluate(trained_model, data_test))
    return reached_positions


def main(path_model="mountaincar2.h5", path_rawdata="training_data.csv", policy="2"):
    # training_data = model_data_preparation()
    try:
        if os.path.isfile(path_model):
            model = load_model(path_model)
            print(" Model Loaded Successfully ")
        else:
            if os.path.isfile(path_rawdata):

                # check if data is empty
                # training_data = np.loadtxt('training_data.csv', delimiter=',')
                training_data = np.genfromtxt(path_rawdata, delimiter=',', dtype=None)
                # print(type(training_data[0]), type(training_data[1]), type(training_data[0][0]))
                print(training_data)
                print(" Loaded data successfully ******************** ")
            else:
                training_data = model_data_preparation(policy)
                print(" Data Gathered from scratch...")

            # training_data = model_data_preparation()

            model = train_model(training_data, path_model)
            print(" Model built from Scratch. And data gathered from scratch ")
    except Exception as e:
        training_data = model_data_preparation(policy)
        model = train_model(training_data, path_model)
        print(" Model built from Scratch. Cause: ", str(e))

    # trained_model = model
    reached = play_AI(model)
    plot_position(reached, path_model, policy)




def plot_position(list_position, path_model, policy):
    game = range(len(list_position))
    # position = [0, 100, 200, 300]
    if policy == '1':
        color = "b"
    elif policy == '2':
        color = "r"
    else:
        color = 'g'
    plt.plot(game, list_position, label=path_model, color=color)
    plt.xlabel('Game number')
    plt.ylabel('Position reached')
    plt.ylim((-0.6, 0.6))
    plt.legend()
    plt.show()
    plt.savefig('plot_' + path_model + '.png')

def model_evaluate(trained_model:Sequential, test_data):
    T_obs = np.array([i[0] for i in test_data]).reshape(-1, 2)
    T_act = np.array([i[1] for i in test_data]).reshape(-1, 3)
    loss_and_metrics = trained_model.evaluate(T_obs, T_act, batch_size=52)
    return loss_and_metrics

if __name__ == "__main__":
    print("hello")

    #path_trained_model_1 = "mountaincar1.h5"
    path_trained_model_2 = "mountaincar2.h5"
    #path_gathered_data_1 = "training_data1.csv"
    path_gathered_data_2 = "training_data2.csv"
    # main(path_model= path_trained_model_1, path_rawdata=path_gathered_data_1, policy = '1')
    main(path_model=path_trained_model_2, path_rawdata=path_gathered_data_2, policy='2')


