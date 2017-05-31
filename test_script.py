import recovery_curve.public as public
import os
import matplotlib.pyplot as plt
import itertools

# define data paths
data_folder = 'data'
s_ns_path = '%s/s_ns.csv' % data_folder
x_ns_path = '%s/x_ns.csv' % data_folder
ts_ns_path = '%s/ts_ns.csv' % data_folder
ys_ns_path = '%s/ys_ns.csv' % data_folder
output_folder = 'predictions'

# read in data
s_ns, x_ns, ts_ns, ys_ns = public.read_data(s_ns_path, x_ns_path, ts_ns_path, ys_ns_path)
assert len(s_ns) == len(x_ns) == len(ts_ns) == len(ys_ns)
N = len(s_ns)

# divide data into training and test
N_test = 10
N_train = N - N_test
s_ns_train, x_ns_train, ts_ns_train, ys_ns_train = s_ns[0:N_train], x_ns[0:N_train], ts_ns[0:N_train], ys_ns[0:N_train]
s_ns_test, x_ns_test, ts_ns_test, ys_ns_test = s_ns[0:N_test], x_ns[0:N_test], ts_ns[0:N_test], ys_ns[0:N_test]

# define hyperparameters of model
s_a, s_b, s_c = 1., 1., 1.
l_a, l_b, l_c, l_m = 10., 10., 10., 10.
n_steps = 2500
num_chains = 4
random_seed = 42

# get posterior samples
(B_a_samples, B_b_samples, B_c_samples, phi_a_samples, phi_b_samples, phi_c_samples, p_samples, theta_samples), test_recovery_curve_samples_ns = public.get_posterior(s_a, s_b, s_c, l_a, l_b, l_c, l_m, n_steps, num_chains, random_seed, s_ns_train, x_ns_train, ts_ns_train, ys_ns_train, s_ns_test, x_ns_test)

# plot posterior predictive for each test sample and 
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
for (i, (recovery_curve_samples, ts, ys)) in enumerate(itertools.izip(test_recovery_curve_samples_ns, ts_ns_test, ys_ns_test)):
    fig, ax = plt.subplots()
    public.plot_recovery_curve_samples(ax, recovery_curve_samples)
    ax.plot(ts, ys, label='true', lw=5, alpha=0.8, color='b')
    ax.set_xlabel('time')
    ax.set_ylabel('fxn value')
    ax.legend()
    fig.savefig('%s/%d.png' % (output_folder, i))
