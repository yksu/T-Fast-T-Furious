import torch
import numpy as np

# The flag below controls whether to allow TF32 on matmul. This flag defaults to True.
torch.backends.cuda.matmul.allow_tf32 = False
# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = False

class ClassificationNetwork(torch.nn.Module):
    def __init__(self, use_obs_layer=True):
        """
        1.1 d)
        Implementation of the network layers. The image size of the input
        observations is 96x96 pixels.
        """
        super().__init__()
        self.use_obs_layer = use_obs_layer
        print(f"Using observation layer: {use_obs_layer}")

        n_observation_outputs = 32 if self.use_obs_layer else 0
        n_outputs = 9

        # Define the activation function and other reusable components
        # self.activation = torch.nn.ReLU()
        self.activation = torch.nn.LeakyReLU(0.1)

        # Define the convolutional layers

        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),
            self.activation,
            torch.nn.BatchNorm2d(num_features=32),
            torch.nn.Conv2d(32, 64, kernel_size=5, padding=1, stride=2),
            self.activation,
            torch.nn.BatchNorm2d(num_features=64)
        )


        if self.use_obs_layer:
            self.obs_layer = torch.nn.Sequential(
                torch.nn.Linear(7, n_observation_outputs),
                self.activation,
                torch.nn.Dropout(0.25)
             )

        # Fully connected layers
        self.fc_layers = torch.nn.Sequential(
            torch.nn.Linear(64 * 47 * 47 + n_observation_outputs, 256),
            self.activation,
            torch.nn.Dropout(0.25),
            torch.nn.Linear(256, 64),
            self.activation,
            torch.nn.Dropout(0.25),
            torch.nn.Linear(64, n_outputs)
        )

        # Setting device on GPU if available, else CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.conv_layers.to(self.device)
        if self.use_obs_layer:
            self.obs_layer.to(self.device)
        self.fc_layers.to(self.device)
        self.to(self.device)

    def forward(self, observation):
        """
        1.1 e)
        The forward pass of the network. Returns the prediction for the given
        input observation.
        observation:   torch.Tensor of size (batch_size, 96, 96, 3)
        return         torch.Tensor of size (batch_size, C)
        """
        # for name, param in self.named_parameters():
        #    print(name, param.shape)
        if self.use_obs_layer:
            sensor_values = self.extract_sensor_values(observation, observation.shape[0])
            sensor_values = torch.cat(sensor_values, dim=1)
            sensor_output = self.obs_layer(sensor_values)
        
        # (N, 96, 96, 3) -> (N, 3, 96, 96)
        observation = observation.permute(0, 3, 1, 2)

        # Pass the observation through the convolutional layers and flatten the output
        x = self.conv_layers(observation)
        x = x.reshape(x.size(0), -1)

        # Pass the flattened output through the fully connected layers, concatenate the sensor output
        #  and pass the result through the fully connected layers again
        if self.use_obs_layer:
            x = self.fc_layers(torch.cat((x, sensor_output), dim=1))
        else:
            x = self.fc_layers(x)
        return x

    def _to_greyscale(self, rgb):
        # (N, 96, 96, 3) -> (N, 1, 96, 96)
        red, green, blue = rgb[:, :, :, 0], rgb[:, :, :, 1], rgb[:, :, :, 2]
        return 0.2989 * red + 0.5870 * green + 0.1140 * blue

    def _single_action_to_class(self, action):
        left_threshold = -0.5
        right_threshold = 0.5
        gas_threshold = 0.25
        brake_threshold = 0.4

        steer = -1 if action[0] < left_threshold else (1 if action[0] > right_threshold else 0)
        gas = 1 if action[1] > gas_threshold else 0
        brake = 1 if action[2] > brake_threshold else 0

        action_tuple = (steer, gas, brake)
        action_class_dict = {
            (0, 0, 0): 0,      # no action
            (0, 1, 0): 1,      # gas
            (-1, 0, 0): 2,     # steer left, no gas
            (-1, 1, 0): 3,     # steer left, gas
            (1, 0, 0): 4,      # steer right, no gas
            (1, 1, 0): 5,      # steer right, gas
            (0, 0, 1): 6,      # brake
            (-1, 0, 1): 7,     # steer left, brake
            (1, 0, 1): 8,      # steer right, brake
        }
        return torch.tensor([action_class_dict[action_tuple]])


    def actions_to_classes(self, actions):
        """
        1.1 c)
        For a given set of actions map every action to its corresponding
        action-class representation. Every action is represented by a 1-dim vector
        with the entry corresponding to the class number.
        actions:        python list of N torch.Tensors of size 3
        return          python list of N torch.Tensors of size 1
        """
        return [self._single_action_to_class(action) for action in actions]

    def scores_to_action(self, scores):
        """
        1.1 c)
        Maps the scores predicted by the network to an action-class and returns
        the corresponding action [steer, gas, brake].
                        C = number of classes
        scores:         python list of torch.Tensors of size C
        return          (float, float, float)
        """
        # Get the index of the highest score
        index = torch.argmax(scores)
        index = int(index)

        # Map the index to an action-class
        action_class_dict = {
            0: (0, 0, 0),      # no action
            1: (0, 1, 0),      # gas
            2: (-1, 0, 0),     # steer left, no gas
            3: (-1, 1, 0),     # steer left, gas
            4: (1, 0, 0),      # steer right, no gas
            5: (1, 1, 0),      # steer right, gas
            6: (0, 0, 1),      # brake
            7: (-1, 0, 1),     # steer left, brake
            8: (1, 0, 1),      # steer right, brake
        }

        # default values
        gas_val = 0.5
        brake_val = 0.8
        
        steer, gas, brake = action_class_dict[index]
        gas = gas_val if gas == 1 else 0
        brake = brake_val if brake == 1 else 0
        return (steer, gas, brake)


    def extract_sensor_values(self, observation, batch_size):
        # just approximately normalized, usually this suffices.
        # can be changed by you
        speed_crop = observation[:, 84:94, 12, 0].reshape(batch_size, -1)
        speed = speed_crop.sum(dim=1, keepdim=True) / 255 / 5

        abs_crop = observation[:, 84:94, 18:25:2, 2].reshape(batch_size, 10, 4)
        abs_sensors = abs_crop.sum(dim=1) / 255 / 5

        steer_crop = observation[:, 88, 38:58, 1].reshape(batch_size, -1) / 255 / 10
        steer_crop[:, :10] *= -1
        steering = steer_crop.sum(dim=1, keepdim=True)

        gyro_crop = observation[:, 88, 58:86, 0].reshape(batch_size, -1) / 255 / 5
        gyro_crop[:, :14] *= -1
        gyroscope = gyro_crop.sum(dim=1, keepdim=True)

        return speed, abs_sensors.reshape(batch_size, 4), steering, gyroscope
