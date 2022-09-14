import h5py
import torch
import matplotlib.pyplot as plt
import lava.lib.dl.slayer as slayer



class VPRNetwork(torch.nn.Module):
    def __init__(self, input_size, output_size, threshold):
        super(VPRNetwork, self).__init__()

        neuron_params = {
                'threshold'     : threshold, # 1.25, 0.5
                'current_decay' : 0.25,
                'voltage_decay' : 0.03,
                'tau_grad'      : 0.03,
                'scale_grad'    : 3,
                'requires_grad' : True,
            }
        neuron_params_drop = {**neuron_params, 'dropout' : slayer.neuron.Dropout(p=0.05),}

        self.blocks = torch.nn.ModuleList([
                slayer.block.cuba.Dense(neuron_params_drop, input_size*input_size*2, 512, weight_norm=True, delay=True), #input is 34 x 34
                slayer.block.cuba.Dense(neuron_params_drop, 512, 512, weight_norm=True, delay=True),
                slayer.block.cuba.Dense(neuron_params, 512, output_size, weight_norm=True), # outputsize is 82
            ])

    def forward(self, spike):
        for block in self.blocks:
            spike = block(spike)
        return spike

    def grad_flow(self, path):
        # helps monitor the gradient flow
        grad = [b.synapse.grad_norm for b in self.blocks if hasattr(b, 'synapse')]

        plt.figure()
        plt.semilogy(grad)
        plt.savefig(path + 'gradFlow.png')
        plt.close()

        return grad

    def export_hdf5(self, filename):
        # network export to hdf5 format
        h = h5py.File(filename, 'w')
        layer = h.create_group('layer')
        for i, b in enumerate(self.blocks):
            b.export_hdf5(layer.create_group(f'{i}'))
