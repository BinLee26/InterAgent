import zarr
import torch
import torch.nn as nn
import numpy as np


def _fit(
    data, mode='limits',
    output_max=1., output_min=-1., range_eps=1e-4,
    fit_offset=True
):
    assert mode in ['limits', 'gaussian']
    assert output_max > output_min

    # convert data to torch and type
    if isinstance(data, zarr.Array):
        data = data[:]
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)

    dim = data.shape[-1]
    data = data.reshape(-1, dim).to(torch.float32)

    # compute input stats min max mean std
    input_min, _ = data.min(axis=0)
    input_max, _ = data.max(axis=0)
    input_mean = data.mean(axis=0)
    input_std = data.std(axis=0)

    # compute scale and offset
    if mode == 'limits':
        if fit_offset:
            # unit scale
            input_range = input_max - input_min
            ignore_dim = input_range < range_eps
            input_range[ignore_dim] = output_max - output_min
            scale = (output_max - output_min) / input_range
            offset = output_min - scale * input_min
            offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]
            # ignore dims scaled to mean of output max and min
        else:
            # use this when data is pre-zero-centered.
            assert output_max > 0
            assert output_min < 0
            # unit abs
            output_abs = min(abs(output_min), abs(output_max))
            input_abs = torch.maximum(torch.abs(input_min), torch.abs(input_max))
            ignore_dim = input_abs < range_eps
            input_abs[ignore_dim] = output_abs
            # don't scale constant channels 
            scale = output_abs / input_abs
            offset = torch.zeros_like(input_mean)
    elif mode == 'gaussian':
        ignore_dim = input_std < range_eps
        scale = input_std.clone()
        scale[ignore_dim] = 1
        scale = 1 / scale
        if fit_offset:
            offset = - input_mean * scale
        else:
            offset = torch.zeros_like(input_mean)
    
    this_params = nn.ParameterDict({
        'scale': scale,
        'offset': offset,
        'input_stats': nn.ParameterDict({
            'min': input_min,
            'max': input_max,
            'mean': input_mean,
            'std': input_std
        })
    })
    for p in this_params.parameters(): p.requires_grad_(False)
    return this_params


class LinearNormalizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.params_dict = nn.ParameterDict()
    
    @torch.no_grad()
    def fit(
        self,
        data,
        mode='limits',
        output_max=1.,
        output_min=-1.,
        range_eps=1e-4,
        fit_offset=True
    ):
        assert isinstance(data, dict), f"Expected LinearNormalizer data to be a dictionary, but got {type(data)}"

        for key, value in data.items():
            self.params_dict[key] =  _fit(
                value,
                mode=mode,
                output_max=output_max,
                output_min=output_min,
                range_eps=range_eps,
                fit_offset=fit_offset
            )

        self.params_dict.requires_grad_(False)
    
    def __getitem__(self, key: str):
        return self.params_dict[key]

    def _normalize_impl(self, d, forward=True):
        assert isinstance(d, dict)
        result = dict()
        for key, x in d.items():
            params = self.params_dict[key]
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)
                
            scale = params['scale']
            offset = params['offset']

            x = x.to(device=scale.device, dtype=scale.dtype)
            src_shape = x.shape
            x = x.reshape(-1, scale.shape[0])
            if forward:
                x = x * scale[:src_shape[-1]] + offset[:src_shape[-1]]
            else:
                x = (x - offset) / scale
            x = x.reshape(src_shape)
            result[key] = x

        return result

    def normalize(self, x):
        return self._normalize_impl(x, forward=True)

    def unnormalize(self, x):
        return self._normalize_impl(x, forward=False)

    def forward(self, x):
        return self.normalize(x)
