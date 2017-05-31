import utils as recovery_utils
import fxns as recovery_fxns
import numpy as np
import scipy
import string
import functools
import itertools


def get_posterior(s_a, s_b, s_c, l_a, l_b, l_c, l_m, n_steps, num_chains, random_seed, s_ns_train, x_ns_train, ts_ns_train, ys_ns_train, s_ns_test, x_ns_test):

    phi_a_dist = recovery_utils.frozen_dist(functools.partial(recovery_utils.sample_truncated_exponential,l_a))
    phi_b_dist = recovery_utils.frozen_dist(functools.partial(recovery_utils.sample_truncated_exponential,l_b))
    phi_c_dist = recovery_utils.frozen_dist(functools.partial(recovery_utils.sample_truncated_exponential,l_c))

    B_phis_dist_fitter = recovery_fxns.B_phis_dist_fitter(s_a, s_b, s_c, phi_a_dist, phi_b_dist, phi_c_dist)
    noise_dist_dist = recovery_fxns.noise_dist_dist(l_m)
    everything_dist_fitter = recovery_fxns.everything_dist_fitter(B_phis_dist_fitter, noise_dist_dist)
    traces_helper = recovery_fxns.get_everything_with_test_dist_traces_helper

    everything_dist = everything_dist_fitter((s_ns_train, x_ns_train, ts_ns_train), ys_ns_train)
    traces = traces_helper(n_steps, random_seed, num_chains, everything_dist, s_ns_train, x_ns_train, ts_ns_train, ys_ns_train, s_ns_test, x_ns_test)
    test_recovery_curve_samples = [
        [recovery_fxns.recovery_curve(s,a,b,c) for (a,b,c) in itertools.izip(a_i_samples, b_i_samples, c_i_samples)]
        for (s, a_i_samples, b_i_samples, c_i_samples) in itertools.izip(s_ns_test, traces.param_to_trace('as_test').T, traces.param_to_trace('bs_test').T, traces.param_to_trace('cs_test').T)
        ]

    return (traces.param_to_trace('B_a'), traces.param_to_trace('B_b'), traces.param_to_trace('B_c'), traces.param_to_trace('phi_a'), traces.param_to_trace('phi_b'), traces.param_to_trace('phi_c'), traces.param_to_trace('p'), traces.param_to_trace('theta')), test_recovery_curve_samples


def plot_recovery_curve_samples(ax, recovery_curve_samples, t_display_low=0., t_display_high=40., sample_color='k', sample_lw=0.5, sample_alpha=0.1, mean_color='r', mean_lw=5, mean_alpha=0.8):
    num_t_display = 51
    ts_display = np.linspace(t_display_low, t_display_high, num_t_display)
    ys_samples = np.array([[recovery_curve(t) for t in ts_display] for recovery_curve in recovery_curve_samples])
    for ys in ys_samples:
        ax.plot(ts_display, ys, color=sample_color, lw=sample_lw, alpha=sample_alpha)
    ys_mean = np.mean(ys_samples, axis=0)
    ax.plot(ts_display, ys_mean, color=mean_color, lw=mean_lw, alpha=mean_alpha, label='posterior mean')


def simulate_data(N, K):
    stdev = 1.
    x_ns = np.array([scipy.stats.norm.rvs(loc=0, scale=stdev, size=K) for n in xrange(N)])
    s_ns = np.random.uniform(size=N)
    ts = np.array([1,2,4,8,12,18,24,30,36,42])
    num_not_missing = 7
    ts_ns = [np.array(sorted(np.random.choice(ts, num_not_missing, replace=False))) for i in xrange(N)]
    b_a, b_b, b_c = 1., 2., 3.
    B_a = b_a * np.ones(K)
    B_b = b_b * np.ones(K)
    B_c = b_c * np.ones(K)
    phi_a, phi_b, phi_c, phi_m = 0.01, 0.01, 0.01, 0.01
    theta, p = 0.1, 0.3
    pop_a, pop_b, pop_c = 0.4, 0.7, 5.
    a_dist = recovery_fxns.a_dist(pop_a)
    b_dist = recovery_fxns.b_dist(pop_b)
    c_dist = recovery_fxns.c_dist(pop_c)
    abc_ns_dist = recovery_fxns.abc_ns_dist(a_dist, b_dist, c_dist)
    constant_B_phis_dist = recovery_utils.constant_dist(recovery_fxns.B_phis(B_a, B_b, B_c, phi_a, phi_b, phi_c))
    constant_abc_ns_dist = abc_ns_dist # this means abc_ns_dist does not contain within it a source of randomness, ie f(x;g); g is random
    constant_noise_dist_dist = recovery_utils.constant_dist(recovery_fxns.noise_dist(theta,p,phi_m))
    constant_everything_dist = recovery_fxns.everything_dist(constant_B_phis_dist, constant_abc_ns_dist, constant_noise_dist_dist)
    everything_sample = constant_everything_dist.sample(s_ns, x_ns, ts_ns)
    ys_ns = everything_sample.ys_ns
    return s_ns, x_ns, ts_ns, ys_ns


def write_ragged_array(arr, path, flat=False):
    if flat:
        arr = arr.reshape((len(arr),1))
    f = open(path, 'w')
    f.write(string.join([string.join(['%.2f' % v for v in row], sep=',') for row in arr], sep='\n'))
    f.close()


def read_ragged_array(path, flat=False):
    f = open(path, 'r')
    arr = [np.array(map(float, string.split(row.strip(),sep=','))) for row in f]
    f.close()
    if flat:
        return np.array([row[0] for row in arr])
    else:
        return arr


def write_data(s_ns, x_ns, ts_ns, ys_ns, s_ns_path, x_ns_path, ts_ns_path, ys_ns_path):
    write_ragged_array(s_ns, s_ns_path, flat=True)
    write_ragged_array(x_ns, x_ns_path)
    write_ragged_array(ts_ns, ts_ns_path)
    write_ragged_array(ys_ns, ys_ns_path)


def read_data(s_ns_path, x_ns_path, ts_ns_path, ys_ns_path):
    s_ns = read_ragged_array(s_ns_path, flat=True)
    x_ns = np.array(read_ragged_array(x_ns_path))
    ts_ns = read_ragged_array(ts_ns_path)
    ys_ns = read_ragged_array(ys_ns_path)
    return s_ns, x_ns, ts_ns, ys_ns
