import paddle.fluid as fluid
from paddle.fluid.dygraph import Conv2D

class DeformConv2d(fluid.dygraph.Layer):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
        """
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        # self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = Conv2D(inc, outc, filter_size=kernel_size, stride=kernel_size, bias_attr=bias)

        self.p_conv = Conv2D(inc, 2*kernel_size*kernel_size, filter_size=3, padding=1, stride=stride, param_attr=fluid.initializer.ConstantInitializer(value=0))
        # TODO: register_backward_hook
        # self.p_conv.register_backward_hook(self._set_lr)
        self.p_conv.register_forward_post_hook(self._set_lr)

        self.modulation = modulation
        if modulation:
            self.m_conv = Conv2D(inc, kernel_size*kernel_size, filter_size=3, padding=1, stride=stride, param_attr=fluid.initializer.ConstantInitializer(value=0))
            self.m_conv.register_forward_post_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):

        offset = self.p_conv(x)
        if self.modulation:
            m = fluid.layers.sigmoid(self.m_conv(x))
        # TODO: dtype = offset.data.type()
        dtype = offset.dtype

        ks = self.kernel_size
        N = offset.shape[1] // 2

        if self.padding:
            # TODO x = self.zero_padding(x)
            x = fluid.layers.pad(x=x, paddings=[0, 0, 0, 0, self.padding, self.padding, self.padding, self.padding])

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        # TODO p = p.contiguous().permute(0, 2, 3, 1)
        p = fluid.layers.transpose(p, perm=[0, 2, 3, 1])
        q_lt = fluid.layers.floor(p.detach())
        q_rb = q_lt + 1
        q_lt = fluid.layers.concat([fluid.layers.clip(q_lt[:, :, :, :N], 0, x.shape[2]-1), fluid.layers.clip(q_lt[:, :, :, N:], 0, x.shape[3]-1)], axis=-1)
        q_rb = fluid.layers.concat([fluid.layers.clip(q_rb[:, :, :, :N], 0, x.shape[2]-1), fluid.layers.clip(q_rb[:, :, :, N:], 0, x.shape[3]-1)], axis=-1)
        q_lb = fluid.layers.concat([q_lt[:, :, :, :N], q_rb[:, :, :, N:]], axis=-1)
        q_rt = fluid.layers.concat([q_rb[:, :, :, :N], q_lt[:, :, :, N:]], axis=-1)

        q_lt = fluid.layers.cast(q_lt, 'int64')
        q_rb = fluid.layers.cast(q_rb, 'int64')
        q_lb = fluid.layers.cast(q_lb, 'int64')
        q_rt = fluid.layers.cast(q_rt, 'int64')

        # clip p
        p = fluid.layers.concat([fluid.layers.clip(p[:, :, :, :N], 0, x.shape[2]-1), fluid.layers.clip(p[:, :, :, N:], 0, x.shape[3]-1)], axis=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (fluid.layers.cast(q_lt[:, :, :, :N], p.dtype) - p[:, :, :, :N])) * (1 + (fluid.layers.cast(q_lt[:, :, :, N:], p.dtype) - p[:, :, :, N:]))
        g_rb = (1 - (fluid.layers.cast(q_rb[:, :, :, :N], p.dtype) - p[:, :, :, :N])) * (1 - (fluid.layers.cast(q_rb[:, :, :, N:], p.dtype) - p[:, :, :, N:]))
        g_lb = (1 + (fluid.layers.cast(q_lb[:, :, :, :N], p.dtype) - p[:, :, :, :N])) * (1 - (fluid.layers.cast(q_lb[:, :, :, N:], p.dtype) - p[:, :, :, N:]))
        g_rt = (1 - (fluid.layers.cast(q_rt[:, :, :, :N], p.dtype) - p[:, :, :, :N])) * (1 + (fluid.layers.cast(q_rt[:, :, :, N:], p.dtype) - p[:, :, :, N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = fluid.layers.unsqueeze(g_lt, axes=1) * x_q_lt + \
                   fluid.layers.unsqueeze(g_rb, axes=1) * x_q_rb + \
                   fluid.layers.unsqueeze(g_lb, axes=1) * x_q_lb + \
                   fluid.layers.unsqueeze(g_rt, axes=1) * x_q_rt

        # TODO: this part didnot finished
        # modulation
        if self.modulation:
            m = fluid.layers.transpose(m, perm=[0, 2, 3, 1])
            m = fluid.layers.unsqueeze(m, axes=1)
            m = fluid.layers.concat([m for _ in range(x_offset.shape[1])], axis=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = fluid.layers.meshgrid(
        [fluid.layers.range(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1, step=1, dtype=dtype),
         fluid.layers.range(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1, step=1, dtype=dtype)])
        # (2N, 1)
        p_n = fluid.layers.concat([fluid.layers.flatten(p_n_x), fluid.layers.flatten(p_n_y)], 0)
        p_n = fluid.layers.reshape(x=p_n, shape=[1, 2*N, 1, 1])
        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = fluid.layers.meshgrid(
            [fluid.layers.range(1, h * self.stride + 1, self.stride, dtype=dtype),
             fluid.layers.range(1, w * self.stride + 1, self.stride, dtype=dtype)])
        p_0_x = fluid.layers.expand(fluid.layers.reshape(p_0_x, shape=[1, 1, h, w]), expand_times=[1, N, 1, 1])
        p_0_y = fluid.layers.expand(fluid.layers.reshape(p_0_y, shape=[1, 1, h, w]), expand_times=[1, N, 1, 1])
        p_0 = fluid.layers.concat([p_0_x, p_0_y], 1)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.shape[1]//2, offset.shape[2], offset.shape[3]

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.shape
        padded_w = x.shape[3]
        c = x.shape[1]
        # (b, c, h*w)
        # TODO: x = x.contiguous().view(b, c, -1)
        x = fluid.layers.reshape(x, shape=[b, c, -1])

        # (b, h, w, N)
        index = q[:, :, :, :N]*padded_w + q[:, :, :, N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = fluid.layers.reshape(
            fluid.layers.expand(fluid.layers.unsqueeze(index, axes=1), expand_times=[1, c, 1, 1, 1]),
            shape=[b, c, -1])

        index = fluid.layers.cast(index, 'int64')

        x_offset = fluid.layers.concat([
            fluid.layers.reshape(
                fluid.layers.concat([
                    fluid.layers.reshape(fluid.layers.gather(x[i, j, :], index[i, j, :]), [1, -1])
                    for j in range(c)], axis=0)
            , [1, c, -1]) for i in range(b)], axis=0)

        x_offset = fluid.layers.reshape(x_offset, [b, c, h, w, N])
        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.shape
        x_offset = fluid.layers.concat([fluid.layers.reshape(x_offset[:, :, :, :, s: s + ks], [b, c, h, w*ks]) for s in range(0, N, ks)], axis=-1)
        x_offset = fluid.layers.reshape(x_offset, shape=[b, c, h*ks, w*ks])
        return x_offset