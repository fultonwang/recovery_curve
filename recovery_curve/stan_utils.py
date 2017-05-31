import pystan
from collections import namedtuple
import numpy as np

traces_base = namedtuple('traces',['permuted','unpermuted','data'])


def get_grid_fig_axes(n_rows, n_cols, n, figsize=None):
    import matplotlib.pyplot as plt
    per_fig = n_rows * n_cols
    num_pages = (int(n) / per_fig) + (1 if (n % per_fig) > 0 else 0)
    figs = []
    axes = []
    for i in xrange(num_pages):
        start = per_fig * i
        end = min(per_fig * (i+1), n)
        assert (end - start) > 0
        if not (figsize is None):
            fig = plt.figure(figsize=figsize)
        else:
            fig = plt.figure()
        for k in range(start,end):
            ax = fig.add_subplot(n_rows, n_cols, (k % per_fig)+1)
            axes.append(ax)
        #fig.tight_layout()
        figs.append(fig)
    return figs, axes


class traces(traces_base):

    def param_to_chain_trace(self, param, chain):
        """
        first dimension is traces.  remaining dimensions are that of original param
        """
        
        # figure out ending point of params
        param_lens = np.zeros(len(self.permuted.keys()))
        for (i,key) in enumerate(self.permuted.keys()):
            param_lens[i] = self.param_flattened_len(key)
        ends = np.cumsum(param_lens)

        # figure out index of param
        idx = None
        for i in range(len(self.permuted.keys())):
            if self.permuted.keys()[i] == param:
                idx = i
                break
        assert idx is not None
        if idx == 0:
            start_pos = 0
        else:
            start_pos = ends[idx-1]
        end_pos = ends[idx]

        return self.unpermuted[:,chain,start_pos:end_pos]
        
        # return 3d array if original param was a matrix
        if len(self.permuted[param].shape) == 2:
            return self.unpermuted[:,chain,start_pos:end_pos]\
              .reshape((self.unpermuted.shape[0], self.permuted[param].shape[1], self.permuted[param].shape[2]))
        else:
            return self.unpermuted[:,chain,start_pos:end_pos]

    @property
    def num_chains(self):
        return self.unpermuted.shape[1]

    @property
    def num_samples(self):
        return self.unpermuted.shape[0]
        
    def param_to_trace(self, param):
        return np.concatenate([self.param_to_chain_trace(param,chain) for chain in xrange(self.num_chains)])

    def param_flattened_len(self, param):
        try:
            return self.permuted[param].shape[1] * self.permuted[param].shape[2]
        except IndexError:
            try:
                return self.permuted[param].shape[1]
            except IndexError:
                return 1
    
    def gelman_statistic(self, param):
        within = np.var(self.param_to_trace(param))
        between = np.mean(np.array([np.var(self.param_to_chain_trace(param, i)) for i in xrange(self.num_chains)]))
        return (within, between)

    def trace_figs(self, params, thin=1):
        figs = []
        for param in params:
            dim = self.param_flattened_len(param)
            n_rows, n_cols = (1,1) if dim == 1 else (3,1)
            param_figs, param_axes = get_grid_fig_axes(n_rows, n_cols, dim)
            assert len(param_axes) == dim
            for (i,param_ax) in enumerate(param_axes):
                param_ax.set_title('%s %d' % (param,i))
                for j in xrange(self.num_chains):
                    component_trace = self.param_to_chain_trace(param,j)[:,i]
                    param_ax.plot(component_trace[0:len(component_trace):thin], alpha=0.5)
            figs += param_figs
        return figs
