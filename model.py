from copy import deepcopy
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

#%%

class ValueRNN_Timescales(nn.Module):
    def __init__(self,
                input_size=1, output_size=1, hidden_size=1, 
                gamma=0.9, bias=True,
                alpha_dist=None,
                learn_weights=True, learn_initial_state=False):
        super().__init__()

        self.gamma = gamma
        self.input_size = input_size # input dimensionality
        self.output_size = output_size # output dimensionality
        self.hidden_size = hidden_size # number of hidden recurrent units

        self.rnn = TimescaleRNN(input_size=input_size, hidden_size=hidden_size, alpha_dist=alpha_dist)
        self.recurrent_cell = 'RNN' # for compatibility with other code

        if learn_weights:
            self.value = nn.Linear(in_features=hidden_size, out_features=output_size, bias=False)
        else:
            self.value = lambda x: torch.sum(x,2)[:,:,None]
        self.learn_weights = learn_weights
        
        self.bias = nn.Parameter(torch.tensor([0.0]*output_size))
        self.learn_bias = bias
        self.bias.requires_grad = self.learn_bias
        self.learn_initial_state = learn_initial_state
        self.predict_next_input = False # for compatibility with other code
        if learn_initial_state:
            self.initial_state = nn.Parameter(torch.zeros(hidden_size))
            self.initial_state.requires_grad = learn_initial_state

        self.saved_weights = {}
        self.reset()

    def forward(self, X, inactivation_indices=None, h0=None, return_hiddens=False, y=None, auto_readout_lr=0.0):
        """ v(t) = w.dot(z(t)), and z(t) = f(x(t), z(t-1)) """

        if inactivation_indices is not None:
            raise Exception("inactivation_indices not implemented for ValueRNN_Timescales")
        if auto_readout_lr > 0:
            raise Exception("auto_readout_lr not implemented for ValueRNN_Timescales")
        
        if type(X) is torch.nn.utils.rnn.PackedSequence:
            batch_size = len(X[2])
        else:
            assert len(X.shape) == 3
            batch_size = X.shape[1]
        
        # get initial state of RNN
        if h0 is None and self.learn_initial_state:
            h0 = torch.tile(self.initial_state, (batch_size,1))[None,:]
        
        # pass inputs through RNN
        Z, last_hidden = self.rnn(X, hx=h0)
        
        if type(Z) is torch.nn.utils.rnn.PackedSequence:
            Z, _ = pad_packed_sequence(Z, batch_first=False)

        V = self.bias + self.value(Z)
        hiddens = Z

        return V, (hiddens if return_hiddens else last_hidden)

    def freeze_weights(self, substr=None):
        for name, p in self.named_parameters():
            if substr is None or substr in name:
                p.requires_grad = False
    
    def unfreeze_weights(self, substr=None):
        for name, p in self.named_parameters():
            if substr is None or substr in name:
                p.requires_grad = True

    def reset(self, seed=None):
        self.bias.data *= 0
        if self.learn_initial_state:
            self.initial_state.data *= 0

        if seed is not None:
            torch.manual_seed(seed)
        for layer in self.children():
           if hasattr(layer, 'reset_parameters'):
               layer.reset_parameters()
        self.initial_weights = self.checkpoint_weights()
               
    def checkpoint_weights(self):
        self.saved_weights = pickle.loads(pickle.dumps(self.state_dict()))
        return self.saved_weights
        
    def restore_weights(self, weights=None):
        weights = self.saved_weights if weights is None else weights
        if weights:
            self.load_state_dict(weights)

    def n_parameters(self):
        return sum([p.numel() for p in self.parameters()])
    
    def save_weights_to_path(self, path, weights=None):
        torch.save(self.state_dict() if weights is None else weights, path)
        
    def load_weights_from_path(self, path):
        self.load_state_dict(torch.load(path))
        
    def get_features(self, name):
        def hook(mdl, input, output):
            self.features[name] = output
        return hook
    
    def prepare_to_gather_activity(self):
        if hasattr(self, 'handle'):
            self.handle.remove()
        self.features = {}
        self.hook = self.get_features('hidden')
        self.handle = self.rnn.register_forward_hook(self.hook)

#%%

class TimescaleRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, alpha_dist=None):
        """
        alpha_dist: time scales for each unit, should be a tensor of shape (hidden_size,)
            or if None, will sample uniformly from [0, 1] for each unit
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Weight matrices
        self.W_in = nn.Parameter(torch.randn(hidden_size, input_size) / input_size**0.5)
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size) / hidden_size**0.5)
        self.bias = nn.Parameter(torch.zeros(hidden_size))

        # Sample alphas
        if alpha_dist is None:
            # Default: uniform [0,1]
            alpha_dist = torch.rand(hidden_size)
        self.register_buffer("alpha", alpha_dist)

    def forward(self, x, h_prev):
        # Candidate state
        h_tilde = torch.tanh(F.linear(x, self.W_in) + F.linear(h_prev, self.W_hh) + self.bias)
        # Leaky integration
        h = (1 - self.alpha) * h_prev + self.alpha * h_tilde
        return h

class TimescaleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, alpha_dist=None, batch_first=False):
        super().__init__()
        self.cell = TimescaleRNNCell(input_size, hidden_size, alpha_dist)
        self.batch_first = batch_first

    def forward(self, input, hx=None):
        if isinstance(input, torch.nn.utils.rnn.PackedSequence):
            # Unpack the sequence
            padded, lengths = pad_packed_sequence(input, batch_first=self.batch_first)
            output, hn = self._forward_padded(padded, lengths, hx)
            # Pack it back
            packed_output = pack_padded_sequence(output, lengths, batch_first=self.batch_first, enforce_sorted=False)
            return packed_output, hn
        else:
            # Regular dense tensor case
            return self._forward_padded(input, None, hx)

    def _forward_padded(self, padded_input, lengths, hx):
        if self.batch_first:
            padded_input = padded_input.transpose(0, 1)  # (seq, batch, feat)

        seq_len, batch_size, _ = padded_input.shape
        if hx is None:
            h_t = torch.zeros(batch_size, self.cell.hidden_size, device=padded_input.device)
        else:
            h_t = hx

        outputs = []
        for t in range(seq_len):
            h_t = self.cell(padded_input[t], h_t)
            outputs.append(h_t.unsqueeze(0))

        output = torch.cat(outputs, dim=0)  # (seq, batch, hidden)
        if self.batch_first:
            output = output.transpose(0, 1)  # back to (batch, seq, hidden)
        return output, h_t
