import tensorflow as tf
from tensorflow import keras
from keras import layers


class RvBLearner:

    def __init__(self, num_joint_actions=9, viewport_size=11):
        # information about the game
        self.num_joint_actions = num_joint_actions
        self.viewport_size = viewport_size

        # learning bits
        self.optimizer = keras.optimizers.Adam()
        self.loss_function = keras.losses.Huber()

        # actual neural nets to update
        self.model = self.create_q_model()
        self.target_model = self.create_q_model()

    def update(self, state_sample, masks, updated_q_values, learner_name=None, epoch=0):

        # state_sample = tf.convert_to_tensor(state_sample)
        # state_sample = tf.convert_to_tensor([tf.convert_to_tensor(state_sample[0]), tf.convert_to_tensor(state_sample[1])])
        state_sample = [tf.convert_to_tensor(state_sample[0]), tf.convert_to_tensor(state_sample[1])]
        masks = tf.convert_to_tensor(masks)
        updated_q_values = tf.convert_to_tensor(updated_q_values, dtype=tf.float32)

        # update the main model
        with tf.GradientTape() as tape:
            # Train the model on the states and updated Q-values
            # by first predicting the q values
            q_values = self.model(state_sample)

            # Apply the masks to the Q-values to get the Q-value for action taken
            q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
            # q_action = -tf.reduce_sum(tf.multiply(p1_q_values, masks), axis=1)

            # Calculate loss between new Q-value and old Q-value
            # can use sample_weight to apply individual loss scaling
            loss = self.loss_function(updated_q_values, q_action)


        # calculate and apply the gradients to the model
        grads = tape.gradient(loss, self.model.trainable_variables)
        grads = [tf.clip_by_norm(g, 2) for g in grads]
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        # periodic tensorboard logging
        if learner_name is not None and epoch % 10 == 0:
            tf.summary.scalar('loss_' + learner_name, loss, step=epoch)
            top = tf.abs(q_action - updated_q_values)
            run_error = top / tf.reduce_sum(top)
            avg_error = tf.math.reduce_mean(run_error)
            max_error = tf.math.reduce_max(top)
            tf.summary.scalar('avg_error_' + learner_name, avg_error, step=epoch)
            tf.summary.scalar('max_error_' + learner_name, max_error, step=epoch)

            max_norm = tf.reduce_max([tf.norm(grad) for grad in grads])
            tf.summary.scalar('max_gradient_norm_' + learner_name, max_norm, step=epoch)

        # error = tf.abs(q_action - updated_q_values).numpy()
        # (tf.abs(q_action - updated_q_values).numpy() ** 0.7 / (tf.abs(q_action - updated_q_values) ** 0.7).numpy().sum()).max()
        # tf.reduce_sum((tf.abs(q_action - updated_q_values).numpy() ** 0.7 / tf.reduce_sum(tf.abs(q_action - updated_q_values) ** 0.7)))

    def update_target_network(self):
        # swap out the target's brain with the main brain
        self.target_model.set_weights(self.model.get_weights())

    def predict_with_target_model(self, sample):
        # basic "target network" prediction
        sample = [tf.convert_to_tensor(sample[0]), tf.convert_to_tensor(sample[1])]
        return self.target_model(sample, training=False).numpy()

    def predict(self, sample, training=True):
        # basic prediction with the primary model
        sample = [tf.convert_to_tensor(sample[0]), tf.convert_to_tensor(sample[1])]
        return self.model(sample, training=training)

    def create_q_model(self):

        # create the networks
        # TODO: should we add some dropout layers?
        # TODO: may need to reduce the number of filters on the conv layers

        inputs_map = layers.Input(shape=(self.viewport_size, self.viewport_size, 5))
        # inputs_goal_vectors = layers.Input(shape=10)
        inputs_goal_vectors = layers.Input(shape=8)

        # map convolutional layers
        layer1 = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='swish')(inputs_map)
        layer2 = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='swish')(layer1)
        layer3 = layers.MaxPooling2D(pool_size=(2, 2))(layer2)
        layer4 = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='swish')(layer3)
        layer5 = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='swish')(layer4)
        layer6 = layers.MaxPooling2D(pool_size=(2, 2))(layer5)
        layer7 = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='swish')(layer6)
        layer_map_flatten = layers.Flatten()(layer7)

        # goal unit vector layers
        layer8 = layers.Dense(64, activation='swish')(inputs_goal_vectors)
        layer9 = layers.Dense(32, activation='swish')(layer8)
        layer_vectors_flatten = layers.Flatten()(layer9)

        # concatenation layer
        layer_concatenation = layers.Concatenate(axis=1)([layer_map_flatten, layer_vectors_flatten])

        # dense policy layers
        dense_layer10 = layers.Dense(128, activation='swish')(layer_concatenation)
        dense_layer11 = layers.Dense(128, activation='swish')(dense_layer10)
        dense_layer12 = layers.Dense(64, activation='swish')(dense_layer11)

        # policy output
        policy_output = layers.Dense(self.num_joint_actions, activation=None)(dense_layer12)

        model = keras.Model(inputs=[inputs_map, inputs_goal_vectors], outputs=policy_output)
        model.compile(optimizer=self.optimizer, loss=self.loss_function)

        # print(model.summary())

        return model

    def save_model(self, save_file_name):
        self.model.save(save_file_name)

    def load_model(self, load_file_name):
        self.model = tf.keras.models.load_model(load_file_name)
        self.target_model = tf.keras.models.load_model(load_file_name)


class SmallRvBLearner(RvBLearner):

    def __init__(self, num_joint_actions=9, viewport_size=11):
        super().__init__(num_joint_actions=num_joint_actions, viewport_size=viewport_size)

    def create_q_model(self):

        # create the networks
        # TODO: should we add some dropout layers?
        # TODO: may need to reduce the number of filters on the conv layers

        inputs_map = layers.Input(shape=(self.viewport_size, self.viewport_size, 5))
        # inputs_goal_vectors = layers.Input(shape=10)
        inputs_goal_vectors = layers.Input(shape=8)

        # map convolutional layers
        layer1 = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='swish')(inputs_map)
        layer2 = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='swish')(layer1)
        layer3 = layers.MaxPooling2D(pool_size=(2, 2))(layer2)
        # layer4 = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='swish')(layer3)
        # layer5 = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='swish')(layer4)
        # layer6 = layers.MaxPooling2D(pool_size=(2, 2))(layer5)
        # layer7 = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='swish')(layer6)
        # layer_map_flatten = layers.Flatten()(layer7)
        layer_map_flatten = layers.Flatten()(layer3)

        # goal unit vector layers
        layer8 = layers.Dense(64, activation='swish')(inputs_goal_vectors)
        layer9 = layers.Dense(32, activation='swish')(layer8)
        layer_vectors_flatten = layers.Flatten()(layer9)

        # concatenation layer
        layer_concatenation = layers.Concatenate(axis=1)([layer_map_flatten, layer_vectors_flatten])

        # dense policy layers
        dense_layer10 = layers.Dense(128, activation='swish')(layer_concatenation)
        # dense_layer11 = layers.Dense(128, activation='swish')(dense_layer10)
        # dense_layer12 = layers.Dense(64, activation='swish')(dense_layer11)
        dense_layer12 = layers.Dense(64, activation='swish')(dense_layer10)

        # policy output
        policy_output = layers.Dense(self.num_joint_actions, activation=None)(dense_layer12)

        model = keras.Model(inputs=[inputs_map, inputs_goal_vectors], outputs=policy_output)
        model.compile(optimizer=self.optimizer, loss=self.loss_function)

        return model


class MediumRvBLearner(RvBLearner):

    def __init__(self, num_joint_actions=9, viewport_size=11):
        super().__init__(num_joint_actions=num_joint_actions, viewport_size=viewport_size)

    def create_q_model(self):

        # create the networks
        inputs_map = layers.Input(shape=(self.viewport_size, self.viewport_size, 5))
        inputs_goal_vectors = layers.Input(shape=8)

        # map convolutional layers
        layer1 = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='swish')(inputs_map)
        layer2 = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='swish')(layer1)
        layer3 = layers.MaxPooling2D(pool_size=(2, 2))(layer2)
        layer4 = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='swish')(layer3)
        layer5 = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='swish')(layer4)
        layer6 = layers.MaxPooling2D(pool_size=(2, 2))(layer5)
        layer_map_flatten = layers.Flatten()(layer6)

        # goal unit vector layers
        layer8 = layers.Dense(64, activation='swish')(inputs_goal_vectors)
        layer9 = layers.Dense(32, activation='swish')(layer8)
        layer_vectors_flatten = layers.Flatten()(layer9)

        # concatenation layer
        layer_concatenation = layers.Concatenate(axis=1)([layer_map_flatten, layer_vectors_flatten])

        # dense policy layers
        dense_layer10 = layers.Dense(128, activation='swish')(layer_concatenation)
        dense_layer11 = layers.Dense(128, activation='swish')(dense_layer10)
        dense_layer12 = layers.Dense(64, activation='swish')(dense_layer11)

        # policy output
        policy_output = layers.Dense(self.num_joint_actions, activation=None)(dense_layer12)

        model = keras.Model(inputs=[inputs_map, inputs_goal_vectors], outputs=policy_output)
        model.compile(optimizer=self.optimizer, loss=self.loss_function)

        return model

