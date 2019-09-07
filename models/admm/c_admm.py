import torch


class ConstrainedADMM_P(torch.nn.Module):
    def __init__(self, net_model, width_ub):
        super(ConstrainedADMM_P, self).__init__()
        self.net = net_model
        self.s = torch.nn.Parameter(torch.Tensor(width_ub))


class ConstrainedADMM_D(torch.nn.Module):
    def __init__(self, n_layers, z_init=0.0, y_init=0.0):
        super(ConstrainedADMM_D, self).__init__()
        self.z = torch.nn.Parameter(torch.Tensor(float(z_init)))
        self.y = torch.nn.Parameter(torch.Tensor(int(n_layers)))
        self.y.data.fill_(y_init)

    def get_param_dicts(self, zlr, ylr):
        return [{"params": self.z, "lr": zlr}, {"params": self.y, "lr": ylr}]
