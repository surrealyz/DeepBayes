from __future__ import print_function

import numpy as np
import tensorflow as tf
import time
import sys

def logsumexp(x):
    x_max = tf.reduce_max(x, 0)
    x_ = x - x_max	# (dimY, N)
    tmp = tf.log(tf.clip_by_value(tf.reduce_sum(tf.exp(x_), 0), 1e-9, np.inf))
    return tmp + x_max

def probx(x, enc, dec, ll, lowerbound, K = 1, beta=1.0, categorical=False):
    enc_conv, enc_mlp = enc
    fea = enc_conv(x)
    N = x.get_shape().as_list()[0]
    logpx = []
    bound_sum=0
    mctimes=50
    for jj in range(mctimes):
        bound, debug_list = lowerbound(x, fea, enc_mlp, dec, ll, K, IS=True, beta=beta, categorical=categorical)
        bound_sum += bound

    logpx.append(bound_sum/mctimes)
    px = tf.math.exp(logpx)
    return logpx, px

def bayes_classifier(x, enc, dec, ll, dimY, lowerbound, K = 1, beta=1.0, categorical=False):
    enc_conv, enc_mlp = enc
    fea = enc_conv(x)
    N = x.get_shape().as_list()[0]
    logpxy = []
    for i in range(dimY):

        y = np.zeros([N, dimY])
        y[:, i] = 1
        y = tf.constant(np.asarray(y, dtype='f'))

        bound_sum=0
        mctimes=50
        for jj in range(mctimes):
            bound, debug_list = lowerbound(x, fea, y, enc_mlp, dec, ll, K, IS=True, beta=beta, categorical=categorical)
            bound_sum += bound

        logpxy.append(tf.expand_dims(bound_sum/mctimes, 1))
    logpxy = tf.concat(logpxy, 1)
    pyx = tf.nn.softmax(logpxy)
    return pyx


# def y_classifier_using_z_x(x, enc, dec, l1, dimY):
#     enc_conv, enc_mlp = enc
#     fea = enc_conv(x)
#     mu_qz, log_sig_qz = enc_mlp(fea, y)

def categorize(X, bin_num):
    X = X * (bin_num-1)
    return X.astype(int)

def construct_optimizer(X_ph, enc, dec, ll, K, vae_type='A', categorical=False):

    # loss function
    enc_conv, enc_mlp = enc
    #fea = enc_conv(X_ph)
    if ll in ['l1_logistic', 'l2_logistic', 'gaussian_logistic', 'laplace_logistic']:
        alpha = 0.01
        X_ = alpha + (1 - alpha*2) * X_ph
        X_ = tf.log(X_) - tf.log(1 - X_)
        ll_ = ll.split('_')[0]
    else:
        X_ = X_ph
        ll_ = ll
    fea = enc_conv(X_)

    if vae_type == 'X':
        from alg.lowerbound_functions_x import lowerbound_F as lowerbound_func

    beta_ph = tf.placeholder(tf.float32, shape=(), name='beta')
    bound, debug_list = lowerbound_func(X_, fea, enc_mlp, dec, ll_, K, IS=True, beta=beta_ph, categorical = categorical)
    bound = tf.reduce_mean(bound)
    batch_size = X_ph.get_shape().as_list()[0]

    # also evaluate approx likelihood accuracy
    logpx, px = probx(X_, enc, dec, ll_, lowerbound_func, K=10, beta=beta_ph, categorical=categorical) # TODO: weired
    logprob = tf.reduce_mean(tf.cast(logpx, tf.float32))

    # now construct optimizers
    lr_ph = tf.placeholder(tf.float32, shape=())
    opt = tf.train.AdamOptimizer(learning_rate=lr_ph).minimize(-bound) #Maximizing ELBO, bound is ELBO
    ops = [opt, bound] + debug_list
    def train(sess, X, lr, beta):
        _, cost, logpx_z, negKL_value = sess.run(ops, feed_dict={X_ph: X, lr_ph: lr, beta_ph: beta})
        return cost, logpx_z, negKL_value

    def train_cat(sess, X, X_catogrical, lr, beta):
        _, cost, logpx_z, negKL_value = sess.run(ops, feed_dict={X_ph: X, lr_ph: lr, beta_ph: beta})
        return cost, logpx_z, negKL_value

    def fit(sess, X, n_iter, lr, beta):
        N = X.shape[0]
        print("training for %d epochs with lr=%.5f, beta=%.2f" % (n_iter, lr, beta))
        begin = time.time()
        n_iter_vae = int(N / batch_size)
        for iteration in range(1, n_iter + 1):
            ind_s = np.random.permutation(range(N))
            bound_total = 0.0
            logpx_z_total = 0.0
            negKL_value_total = 0.0
            for j in range(0, n_iter_vae):
                indl = j * batch_size
                indr = (j+1) * batch_size
                ind = ind_s[indl:min(indr, N)]
                if indr > N:
                    ind = np.concatenate((ind, ind_s[:(indr-N)]))
                batch = X[ind]
                #batch = np.clip(batch + np.random.uniform(size=batch.shape) * 1./255., 0.0, 1.0)
                cost, logpx_z, negKL_value = train(sess, batch, Y[ind], lr, beta)
                bound_total += cost / n_iter_vae
                logpx_z_total += logpx_z / n_iter_vae
                negKL_value_total += negKL_value / n_iter_vae

            end = time.time()
            print("Iter %d, ELBO=%.5f, p(x|z)=%.5f, KL=%.5f, time=%.2f" \
                  % (iteration, bound_total, logpx_z_total, -negKL_value_total, end - begin))
            begin = end
            sys.stdout.flush()

    def fit_categorical(sess, X, n_iter, lr, beta):
        X_categorical = categorize(np.copy(X), bin_num=2)
        N = X.shape[0]
        print("training for %d epochs with lr=%.5f, beta=%.2f" % (n_iter, lr, beta))
        begin = time.time()
        n_iter_vae = int(N / batch_size)
        for iteration in range(1, n_iter + 1):
            ind_s = np.random.permutation(range(N))
            bound_total = 0.0
            logpx_z_total = 0.0
            negKL_value_total = 0.0
            for j in range(0, n_iter_vae):
                indl = j * batch_size
                indr = (j+1) * batch_size
                ind = ind_s[indl:min(indr, N)]
                if indr > N:
                    ind = np.concatenate((ind, ind_s[:(indr-N)]))
                batch = X[ind]
                batch_x_cat = X_categorical[ind]
                #batch = np.clip(batch + np.random.uniform(size=batch.shape) * 1./255., 0.0, 1.0)
                cost, logpx_z, negKL_value = train_cat(sess, batch, batch_x_cat, lr, beta)
                bound_total += cost / n_iter_vae
                logpx_z_total += logpx_z / n_iter_vae
                negKL_value_total += negKL_value / n_iter_vae

            end = time.time()
            print("Iter %d, ELBO=%.5f, p(x|z)=%.5f, KL=%.5f, time=%.2f" \
                  % (iteration, bound_total, logpx_z_total, -negKL_value_total, end - begin))
            begin = end
            sys.stdout.flush()

    def eval_categorical(sess, X, data_name = 'train', beta=1.0):
        # X_categorical = categorize(np.copy(X), bin_num=128)
        N = X.shape[0]
        begin = time.time()
        n_batch = int(N / batch_size)
        logprob_sum = 0.0
        bound_total = 0.0
        for j in range(0, n_batch):
            indl = j * batch_size
            indr = min((j+1) * batch_size, N)
            res_logpx, res_px, res1, res2 = sess.run((logpx, px, logprob, bound), feed_dict={X_ph:X[indl:indr],
                                                                 # x_cat: X_categorical[indl:indr],
                                                                 beta_ph: beta})
            #print('res_logpx:', res_logpx, 'res_px:', res_px)
            logprob_sum += res1 / n_batch
            bound_total += res2 / n_batch
        end = time.time()
        print("%s data approx log likelihood logpx=%.2f, bound=%.2f, time=%.2f, beta=%.2f" \
              % (data_name, logprob_sum, bound_total, end - begin, beta))
        sys.stdout.flush()
        return logprob_sum, bound_total

    def eval(sess, X, data_name = 'train', beta=1.0):
        N = X.shape[0]
        begin = time.time()
        n_batch = int(N / batch_size)
        logprob_sum = 0.0
        bound_total = 0.0
        for j in range(0, n_batch):
            indl = j * batch_size
            indr = min((j+1) * batch_size, N)
            res_logpx, res_px, res1, res2 = sess.run((logpx, px, logprob, bound), feed_dict={X_ph:X[indl:indr],
                                                                 beta_ph: beta})
            #print('res_logpx:', res_logpx, 'res_px:', res_px)
            logprob_sum += res1 / n_batch
            bound_total += res2 / n_batch
        end = time.time()
        print("%s data approx log likelihood logpx=%.2f, bound=%.2f, time=%.2f, beta=%.2f" \
              % (data_name, logprob_sum, bound_total, end - begin, beta))
        sys.stdout.flush()
        return logprob_sum, bound_total

    if categorical:
        return fit_categorical, eval_categorical
    else:
        return fit, eval
