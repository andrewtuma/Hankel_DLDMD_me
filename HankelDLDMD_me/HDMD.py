"""
    Author:
        Jay Lago, SDSU, 2021
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *

class HDMD(keras.Model):
    def __init__(self, hyp_params, **kwargs):
        super(HDMD, self).__init__(**kwargs)

        # Parameters
        self.batch_size = hyp_params['batch_size']
        self.phys_dim = hyp_params['phys_dim']
        self.latent_dim = hyp_params['latent_dim']
        self.num_time_steps = int(hyp_params['num_time_steps'])
        self.num_pred_steps = int(hyp_params['num_pred_steps'])
        self.time_final = hyp_params['time_final']
        self.num_en_layers = hyp_params['num_en_layers']
        self.num_neurons = hyp_params['num_en_neurons']
        self.delta_t = hyp_params['delta_t']
        self.kernel_size = 1
        self.enc_input = (self.num_time_steps, self.phys_dim)
        self.dec_input = (self.num_time_steps, self.latent_dim)
        self.precision = hyp_params['precision']
        if self.precision == 'float32':
            self.precision_complex = tf.complex64
        else:
            self.precision_complex = tf.complex128
        self.dmd_threshold = tf.constant(-10.0, dtype=self.precision)
        self.log10 = tf.cast(tf.math.log(10.0), dtype=self.precision)
        self.window = hyp_params['window']

        # Construct the ENCODER network
        self.encoder = keras.Sequential(name="encoder")
        self.encoder.add(Dense(self.num_neurons,
                               input_shape=self.enc_input,
                               activation=hyp_params['hidden_activation'],
                               kernel_initializer=hyp_params['kernel_init_enc'],
                               bias_initializer=hyp_params['bias_initializer'],
                               trainable=True, name='enc_in'))
        for ii in range(self.num_en_layers):
            self.encoder.add(Dense(self.num_neurons,
                                   activation=hyp_params['hidden_activation'],
                                   kernel_initializer=hyp_params['kernel_init_enc'],
                                   bias_initializer=hyp_params['bias_initializer'],
                                   trainable=True, name='enc_' + str(ii)))
        self.encoder.add(Dense(self.latent_dim,
                               activation=hyp_params['ae_output_activation'],
                               kernel_initializer=hyp_params['kernel_init_enc'],
                               bias_initializer=hyp_params['bias_initializer'],
                               trainable=True, name='enc_out'))

        # Construct the DECODER network
        self.decoder = keras.Sequential(name="decoder")
        self.decoder.add(Dense(self.num_neurons,
                               input_shape=self.dec_input,
                               activation=hyp_params['hidden_activation'],
                               kernel_initializer=hyp_params['kernel_init_enc'],
                               bias_initializer=hyp_params['bias_initializer'],
                               trainable=True, name='dec_in'))
        for ii in range(self.num_en_layers):
            self.decoder.add(Dense(self.num_neurons,
                                   activation=hyp_params['hidden_activation'],
                                   kernel_initializer=hyp_params['kernel_init_dec'],
                                   bias_initializer=hyp_params['bias_initializer'],
                                   trainable=True, name='dec_' + str(ii)))
        self.decoder.add(Dense(self.phys_dim,
                               activation=hyp_params['ae_output_activation'],
                               kernel_initializer=hyp_params['kernel_init_dec'],
                               bias_initializer=hyp_params['bias_initializer'],
                               trainable=True, name='dec_out'))


    def call(self, x):
        # Encoder on the entire time series
        y = self.encoder(x)
        x_ae = self.decoder(y)

        # Generate latent time series using DMD prediction
        y_adv = self.hankel_dmd(tf.transpose(y, [0, 2, 1]))

        # Decode the latent trajectories
        x_adv = self.decoder(y_adv)

        # Model weights
        weights = self.trainable_weights

        return [y, x_ae, x_adv, y_adv, weights]


    def hankel_dmd(self, y):
        num_ic, num_dims, num_time = y.shape
        num_rows_hankel = num_time - (self.window - 1)

        # Create Hankel matrix
        big_hankel = tf.TensorArray(dtype=self.precision, size=self.latent_dim)
        for jj in tf.range(self.latent_dim):
            obs = y[:, jj, :]
            hmat = tf.TensorArray(dtype=self.precision, size=num_rows_hankel)
            for kk in tf.range(num_rows_hankel):
                hmat = hmat.write(kk, obs[:, kk:(kk + self.window)])
            big_hankel = big_hankel.write(jj, hmat.stack())
        big_hankel = tf.reshape(tf.transpose(big_hankel.stack(), [2, 0, 1, 3]),
                                [self.batch_size, self.latent_dim * num_rows_hankel, self.window])

        # Perform standard DMD
        stride = tf.cast(big_hankel.shape[1] / self.latent_dim, dtype=tf.int64)
        Y_m = tf.identity(big_hankel[:, :, :-1])
        Y_p = tf.identity(big_hankel[:, :, 1:])

        sig, U, V = tf.linalg.svd(Y_m, compute_uv=True, full_matrices=False)
        r = sig.shape[-1]
        sigr_inv = tf.linalg.diag(1.0 / sig[:, :r])
        Ur = U[:, :, :r]
        Urh = tf.linalg.adjoint(Ur)
        Vr = V[:, :, :r]

        A = Y_p @ Vr @ sigr_inv @ Urh
        evals, evecs = tf.linalg.eig(A)
        Phi = tf.linalg.solve(evecs, tf.cast(Y_m, dtype=self.precision_complex))
        y0 = tf.identity(Phi[:, :, 0])
        y0 = y0[:, :, tf.newaxis]
        evecs = evecs[:, ::stride, :]

        recon = tf.TensorArray(self.precision, size=self.num_pred_steps)
        recon = recon.write(0, tf.math.real(evecs @ y0))
        evals_k = tf.identity(evals)
        for ii in tf.range(1, self.num_pred_steps):
            tmp = tf.math.real(evecs @ (tf.linalg.diag(evals_k) @ y0))
            recon = recon.write(ii, tmp)
            evals_k = evals_k * evals
        recon = tf.transpose(tf.squeeze(recon.stack()), perm=[1, 0, 2])

        return recon


    def get_config(self):
        base_config = super().get_config()
        return {**base_config,
                'encoder': self.encoder,
                'decoder': self.decoder}
