import numpy as np
import tensorflow as tf
from data.load_data import load_data

tf.random.set_seed(42)
np.random.seed(42)

from tensorflow_probability import distributions as tfd
from tensorflow.keras.layers import Input, Dense, Activation, Concatenate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np


class MDN(tf.keras.Model):
    def __init__(self, neurons=100, components=2):
        super(MDN, self).__init__(name="MDN")
        self.neurons = neurons
        self.components = components

        self.h1 = Dense(neurons, activation="relu", name="h1")
        self.h2 = Dense(neurons, activation="relu", name="h2")

        self.alphas = Dense(components, activation="softmax", name="alphas")
        self.mus = Dense(components, name="mus")
        self.sigmas = Dense(components, activation="nnelu", name="sigmas")
        self.pvec = Concatenate(name="pvec")

    def call(self, inputs):
        x = self.h1(inputs)
        x = self.h2(x)

        alpha_v = self.alphas(x)
        mu_v = self.mus(x)
        sigma_v = self.sigmas(x)

        return self.pvec([alpha_v, mu_v, sigma_v])


def nnelu(input):
    """ Computes the Non-Negative Exponential Linear Unit
    """
    return tf.add(tf.constant(1, dtype=tf.float32), tf.nn.elu(input))


def slice_parameter_vectors(parameter_vector):
    """ Returns an unpacked list of paramter vectors.
    """
    return [parameter_vector[:, i * components:(i + 1) * components] for i in range(no_parameters)]


def gnll_loss(y, parameter_vector):
    """ Computes the mean negative log-likelihood loss of y given the mixture parameters.
    """
    alpha, mu, sigma = slice_parameter_vectors(parameter_vector)  # Unpack parameter vectors
    gm = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=alpha),
        components_distribution=tfd.Normal(
            loc=mu,
            scale=sigma))

    log_likelihood = gm.log_prob(tf.transpose(y))  # Evaluate log-probability of y

    return -tf.reduce_mean(log_likelihood, axis=-1)


def eval_mdn_model(x_test, y_test, mdn_model):

    y_pred = mdn_model.predict(x_test)
    alpha_pred, mu_pred, sigma_pred = slice_parameter_vectors(y_pred)
    print(alpha_pred)
    list = tf.losses.mean_squared_error(np.multiply(alpha_pred, mu_pred).sum(axis=-1)[:, np.newaxis], y_test).numpy()
    print("MDN-MSE: {:1.3f}".format(np.sum(list) / len(list)))
    # print("MDN-NLL: {:1.3f}\n".format(gnll_eval(y_test.astype(np.float32), alpha_pred, mu_pred, sigma_pred).numpy()[0]))


tf.keras.utils.get_custom_objects().update({'nnelu': Activation(nnelu)})

no_parameters = 3
components = 3
neurons = 200

opt = tf.optimizers.Adam(1e-3)

x, y = load_data()
x = x.numpy()
y = y.numpy().astype(np.float32)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42,  shuffle=True)

min_max_scaler = MinMaxScaler()

x_train = min_max_scaler.fit_transform(x_train)
x_test = min_max_scaler.transform(x_test)

mdn_3 = MDN(neurons=neurons, components=components)
mdn_3.compile(loss=gnll_loss, optimizer=opt)

mdn_3.fit(x=x_train, y=y_train, epochs=300, validation_data=(x_test, y_test), batch_size=128, verbose=1)  #callbacks=[mon],

eval_mdn_model(x_test, y_test, mdn_3)


