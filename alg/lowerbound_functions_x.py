import numpy as np
import tensorflow as tf

def sample_gaussian(mu, log_sig, K):
    mu = tf.tile(mu, [K, 1])
    log_sig = tf.tile(log_sig, [K, 1])
    z =  mu + tf.exp(log_sig) * tf.random_normal(mu.get_shape())
    return mu, log_sig, z

def sample_gaussian_fix_randomness(mu, log_sig, K, seed):
    N = mu.get_shape().as_list()[0]
    mu = tf.tile(mu, [K, 1])
    log_sig = tf.tile(log_sig, [K, 1])
    np.random.seed(seed*100)
    eps = np.random.randn(K, mu.get_shape().as_list()[1])
    eps = np.repeat(eps, N, 0)
    eps = tf.constant(np.asarray(eps, dtype='f'))
    z = mu + tf.exp(log_sig) * eps
    return mu, log_sig, z

# define log densities
def log_gaussian_prob(x, mu=0.0, log_sig=0.0):
    logprob = -(0.5 * np.log(2 * np.pi) + log_sig) \
                - 0.5 * ((x - mu) / tf.exp(log_sig)) ** 2
    ind = list(range(1, len(x.get_shape().as_list())))
    return tf.reduce_sum(logprob, ind)

def log_bernoulli_prob(x, p=0.5):
    logprob = x * tf.log(tf.clip_by_value(p, 1e-9, 1.0)) \
              + (1 - x) * tf.log(tf.clip_by_value(1.0 - p, 1e-9, 1.0))
    ind = list(range(1, len(x.get_shape().as_list())))
    return tf.reduce_sum(logprob, ind)

def log_logistic_cdf_prob(x, mu, log_scale):
    binsize = np.asarray(1/255.0, dtype='f')
    scale = tf.exp(log_scale)
    sample = (tf.floor(x / binsize) * binsize - mu) / scale
    #prob = tf.sigmoid(sample + binsize / scale) - tf.sigmoid(sample)
    #logprob = tf.log(prob + 1e-5)

    logprob = tf.log(1 - tf.exp(-binsize / scale))
    logprob -= tf.nn.softplus(sample)
    logprob -= tf.nn.softplus(-sample - binsize/scale)
    ind = list(range(1, len(x.get_shape().as_list())))
    return tf.reduce_sum(logprob, ind)

def logsumexp(x):
    x_max = tf.reduce_max(x, 0)
    x_ = x - x_max	# (dimY, N)
    tmp = tf.log(tf.clip_by_value(tf.reduce_sum(tf.exp(x_), 0), 1e-20, np.inf))
    return tmp + x_max

def encoding(enc_mlp, fea, K, use_mean=False, fix_samples=False, seed=0):
    mu_qz, log_sig_qz = enc_mlp(fea)

    if use_mean:
        z = mu_qz
    elif fix_samples:
        mu_qz, log_sig_qz, z = sample_gaussian_fix_randomness(mu_qz, log_sig_qz, K, seed)
    else:
        mu_qz, log_sig_qz, z = sample_gaussian(mu_qz, log_sig_qz, K)

    logq = log_gaussian_prob(z, mu_qz, log_sig_qz)

    return z, logq

def lowerbound_F(x, fea, enc_mlp, dec, ll, K=1, IS=False,
               use_mean=False, fix_samples=False, seed=0, z=None, beta=1.0, categorical=False):
    if use_mean:
        K = 1
        fix_samples=False
    K=1
    if z is None:
        # print("z is None\n#")
        # print()
        z, logq = encoding(enc_mlp, fea, K, use_mean, fix_samples, seed)
        # print('z shape', z.shape)
    else:
        mu_qz, log_sig_qz = enc_mlp(fea)
        logq = log_gaussian_prob(z, mu_qz, log_sig_qz)
        # print('z shape', z.shape)

    # if len(x.get_shape().as_list()) == 2:
    #     x_rep = tf.tile(x, [K, 1])  #TODO: why replicate?

    # if len(x.get_shape().as_list()) == 4:
    #     x_rep = tf.tile(x, [K, 1, 1, 1])
    # y_rep = tf.tile(y, [K, 1])

    x_rep = x

    if categorical:
        # x_rep = x_cat
        bin_num=2
        x_rep = tf.dtypes.cast(x_rep * (bin_num-1), tf.int32)

    # prior
    log_prior_z = log_gaussian_prob(z, 0.0, 0.0)

    # decoders
    pxz = dec
    mu_x = pxz(z)

    if ll == 'bernoulli':
        logpx = log_bernoulli_prob(x_rep, mu_x)
    if ll == 'l2':
        ind = list(range(1, len(x_rep.get_shape().as_list())))
        logpx = -tf.reduce_sum((x_rep - mu_x)**2, ind)
    if ll == 'l1':
        ind = list(range(1, len(x_rep.get_shape().as_list())))
        logpx = -tf.reduce_sum(tf.abs(x_rep - mu_x), ind)
    if ll =='xe':
        # print('label shape', x_rep.shape, mu_x.shape)
        tmp = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=x_rep, logits=mu_x)
        # print("tmp", tmp.shape)
        logpx = - tf.reduce_sum(tmp, axis=1)

    #print('shape ', logpx.shape, (log_prior_z - logq).shape)
    #bound = logp + log_pyzx + beta * (log_prior_z - logq)
    bound = logpx + beta * (log_prior_z - logq)  #TODO: this is ELBO, need to be maximized
    negKL = log_prior_z - logq

    # if IS and K > 1:	# importance sampling estimate
    #     N = x.get_shape().as_list()[0]
    #     bound = tf.reshape(bound, [K, N])
    #     bound = logsumexp(bound) - tf.log(float(K))

    return bound, [tf.reduce_mean(logpx), tf.reduce_mean(negKL)]
