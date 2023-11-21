import os
import numpy as np
import gymnasium as gym
import time
import pygame

from sdc_wrapper import SDC_Wrapper


def load_demonstrations(data_folder):
    """
    1.1 a)
    Given the folder containing the expert demonstrations, the data gets loaded and
    stored it in two lists: observations and actions.
                    N = number of (observation, action) - pairs
    data_folder:    python string, the path to the folder containing the
                    observation_%05d.npy and action_%05d.npy files
    return:
    observations:   python list of N numpy.ndarrays of size (96, 96, 3)
    actions:        python list of N numpy.ndarrays of size 3
    """
    # load the files from the data_folder
    actions = []
    observations = []

    i = 0
    while True:
        try:
            # add the action and observation to the lists
            actions.append(np.load(os.path.join(data_folder, f'action_{i}.npy')))
            observations.append(np.load(os.path.join(data_folder, f'observation_{i}.npy')))

            i += 1

        except:
            break
    
    return np.array(observations), np.array(actions)


def save_demonstrations(data_folder, actions, observations):
    """
    1.1 f)
    Save the lists actions and observations in numpy .npy files that can be read
    by the function load_demonstrations.
                    N = number of (observation, action) - pairs
    data_folder:    python string, the path to the folder containing the
                    observation_%05d.npy and action_%05d.npy files
    observations:   python list of N numpy.ndarrays of size (96, 96, 3)
    actions:        python list of N numpy.ndarrays of size 3
    """
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    start_idx = len(os.listdir(data_folder)) // 2

    # save the files to the data_folder
    for i in range(len(actions)):
        # save the action and observation
        np.save(os.path.join(data_folder, f'action_{i + start_idx}.npy'), actions[i])
        np.save(os.path.join(data_folder, f'observation_{i + start_idx}.npy'), observations[i])


class ControlStatus:
    """
    Class to keep track of key presses while recording demonstrations.
    """

    def __init__(self):
        self.stop = False
        self.save = False
        self.quit = False

        self.steer = 0.0
        self.accelerate = 0.0
        self.brake = 0.0

    def update(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit = True

            if event.type == pygame.KEYDOWN:
                self.key_press(event)

        keys = pygame.key.get_pressed()
        self.accelerate = 0.5 if keys[pygame.K_UP] else 0
        self.brake = 0.8 if keys[pygame.K_DOWN] else 0
        self.steer = 1 if keys[pygame.K_RIGHT] else (-1 if keys[pygame.K_LEFT] else 0)

    def key_press(self, event):
        if event.key == pygame.K_ESCAPE:
            self.quit = True
        if event.key == pygame.K_SPACE:
            self.stop = True
        if event.key == pygame.K_TAB:
            self.save = True


def record_demonstrations(demonstrations_folder):
    """
    Function to record own demonstrations by driving the car in the gym car-racing
    environment.
    demonstrations_folder:  python string, the path to where the recorded demonstrations
                        are to be saved

    The controls are:
    arrow keys:         control the car; steer left, steer right, gas, brake
    ESC:                quit and close
    SPACE:              restart on a new track
    TAB:                save the current run
    """

    env = SDC_Wrapper(gym.make('CarRacing-v2', render_mode='human'), remove_score=True, return_linear_velocity=False)
    try:
        _, _ = env.reset(seed=int(np.random.randint(0, 1e6)))
    except:
        print("Please note that you can't collect data on the cluster.")
        return

    status = ControlStatus()
    total_reward = 0.0

    while not status.quit:
        observations = []
        actions = []
        # get an observation from the environment
        observation, _ = env.reset()

        while not status.stop and not status.save and not status.quit:
            status.update()

            # collect all observations and actions
            observations.append(observation.copy())
            actions.append(np.array([status.steer, status.accelerate,
                                     status.brake]))
            # submit the users' action to the environment and get the reward
            # for that step as well as the new observation (status)
            observation, reward, done, trunc, info = env.step([status.steer,
                                                               status.accelerate,
                                                               status.brake])

            total_reward += reward
            time.sleep(0.01)

        if status.save:
            save_demonstrations(demonstrations_folder, actions, observations)
            status.save = False

        status.stop = False

    env.close()


# test the functions
if __name__ == '__main__':
    # test record_demonstrations
    record_demonstrations('data_new')




'''if __name__ == '__main__':
    # print current folder
    print(os.getcwd())
    # print content of current folder
    print(os.listdir())
    # observations, actions = load_demonstrations('ex_01_IL_challenge/template/data')
    observations, actions = load_demonstrations('data')
    print('actions: ', len(actions))
    print('observations: ', len(observations))

    # create a counting map for the actions
    actions_map = {}

    for action in actions:
        if str(action) in actions_map:
            actions_map[str(action)] += 1
        else:
            actions_map[str(action)] = 1

    # sort by the number of occurences
    actions_map = dict(sorted(actions_map.items(), key=lambda item: item[1], reverse=True))
    print(actions_map)


    def actions_to_classes(actions):
        """
        1.1 c)
        For a given set of actions map every action to its corresponding
        action-class representation. Every action is represented by a 1-dim vector
        with the entry corresponding to the class number.
        actions:        python list of N torch.Tensors of size 3
        return          python list of N torch.Tensors of size 1
        """
        # 3 actions: steer, gas, brake

        action_mapping = {
            (0., 0., 0.): 0,  # no action
            (0., 0.5, 0.): 1,  # gas
            (-1., 0., 0.): 2,  # steer left, no gas
            (-1., 0.5, 0.): 3,  # steer left, gas
            (1., 0., 0.): 4,  # steer right, no gas
            (1., 0.5, 0.): 5,  # steer right, gas
        }

        return np.array([action_mapping[tuple(action)] for action in actions])


    # try out this function
    classes = actions_to_classes(actions)
    print(classes)

    # count the classes
    print(np.unique(classes, return_counts=True))
'''