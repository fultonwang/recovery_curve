import recovery_curve.public as public
import os

# define data paths
path = os.path.dirname(os.path.realpath(__file__))
data_folder = '%s/data' % path
s_ns_path = '%s/s_ns.csv' % data_folder
x_ns_path = '%s/x_ns.csv' % data_folder
ts_ns_path = '%s/ts_ns.csv' % data_folder
ys_ns_path = '%s/ys_ns.csv' % data_folder


# generate data
N = 500
K = 1
s_ns, x_ns, ts_ns, ys_ns = public.simulate_data(N, K)

# write data
if not os.path.exists(data_folder):
    os.makedirs(data_folder)
public.write_data(s_ns, x_ns, ts_ns, ys_ns, s_ns_path, x_ns_path, ts_ns_path, ys_ns_path)
