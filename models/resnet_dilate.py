from paddle.fluid.dygraph import Sequential
from paddle.fluid.dygraph import BatchNorm
from paddle.fluid.dygraph import Dropout
from paddle.fluid.layers import relu

from models.deform_conv_v2 import *
from models.resnet import *
from lib.rpn_util import *

class ResNetDilate(fluid.dygraph.Layer):
  def __init__(self, num_layer=50):
    super(ResNetDilate, self).__init__()
    if num_layer == 50:
        model_resnet = resnet50(pretrained=True)
    if num_layer == 101:
        model_resnet = resnet101(pretrained=True)
    self.conv1 = model_resnet.conv1
    self.bn1 = model_resnet.bn1
    self.maxpool = model_resnet.maxpool
    self.layer1 = model_resnet.layer1
    self.layer2 = model_resnet.layer2
    self.layer3 = model_resnet.layer3
    self.layer4 = model_resnet.layer4

    # for n, m in self.layer3.named_modules():
    #     if 'conv2' in n:
    #         m._dilation, m._padding, m._stride = (2, 2), (2, 2), (1, 1)
    #     elif 'downsample.0' in n:
    #         m._stride = (1, 1)

    for n, m in self.layer4.named_sublayers():
        if 'conv2' in n:  # conv1 for resnet34
            m._dilation, m._padding, m._stride = (2, 2), (2, 2), (1, 1)
        elif 'downsample.0' in n:
            m._stride = (1, 1)

    # self.avgpool = nn.AdaptiveMaxPool2d((1,1))
    #
    # self.fc = nn.Linear(model_resnet.fc.in_features, class_num)
    # self.fc.apply(init_weights)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    # x = self.avgpool(x)
    # x = x.view(x.size(0), -1)
    # x = self.fc(x)
    return x

def dynamic_local_filtering(x, depth, dilated=1):
    # TODO: padding = nn.ReflectionPad2d(dilated)  # ConstantPad2d(1, 0)
    pad_depth = fluid.layers.pad2d(input=depth, paddings=[dilated, dilated, dilated, dilated], mode='reflect')
    n, c, h, w = x.shape
    # y = torch.cat((x[:, int(c/2):, :, :], x[:, :int(c/2), :, :]), dim=1)
    # x = x + y
    y = fluid.layers.concat(input=[x[:, -1:, :, :], x[:, :-1, :, :]], axis=1)
    z = fluid.layers.concat(input=[x[:, -2:, :, :], x[:, :-2, :, :]], axis=1)
    x = (x + y + z) / 3
    pad_x = fluid.layers.pad2d(input=x, paddings=[dilated, dilated, dilated, dilated], mode='reflect')
    filter = fluid.layers.assign(pad_depth[:, :, dilated: dilated + h, dilated: dilated + w] * pad_x[:, :, dilated: dilated + h, dilated: dilated + w])
    for i in [-dilated, 0, dilated]:
        for j in [-dilated, 0, dilated]:
            if i != 0 or j != 0:
                filter += fluid.layers.assign(pad_depth[:, :, dilated + i: dilated + i + h, dilated + j: dilated + j + w] * pad_x[:, :, dilated + i: dilated + i + h, dilated + j: dilated + j + w])
    return filter / 9

class RPN(fluid.dygraph.Layer):
    def __init__(self, phase, conf):
        super(RPN, self).__init__()

        self.base = ResNetDilate(conf.base_model)
        self.adaptive_diated = conf.adaptive_diated
        self.dropout_position = conf.dropout_position
        self.use_dropout = conf.use_dropout
        self.drop_channel = conf.drop_channel
        self.use_corner = conf.use_corner
        self.corner_in_3d = conf.corner_in_3d
        self.deformable = conf.deformable

        if conf.use_rcnn_pretrain:
            # print(self.base.state_dict().keys())
            if conf.base_model == 101:
                pretrained_model = fluid.io.load_program_state('faster_rcnn_1_10_14657.pth')['models']
                rename_dict = {'RCNN_top.0': 'layer4', 'RCNN_base.0': 'conv1', 'RCNN_base.1': 'bn1', 'RCNN_base.2': 'relu',
                               'RCNN_base.3': 'maxpool', 'RCNN_base.4': 'layer1',
                               'RCNN_base.5': 'layer2', 'RCNN_base.6': 'layer3'}
                change_dict = {}
                for item in pretrained_model.keys():
                    for rcnn_name in rename_dict.keys():
                        if rcnn_name in item:
                            change_dict[item] = item.replace(rcnn_name, rename_dict[rcnn_name])
                            break
                pretrained_model = {change_dict[k]: v for k, v in pretrained_model.items() if k in change_dict}
                self.base.load_state_dict(pretrained_model)

            elif conf.base_model == 50:
                pretrained_model =fluid.io.load_program_state('res50_faster_rcnn_iter_1190000.pth',
                                              map_location=lambda storage, loc: storage)
                pretrained_model = {k.replace('resnet.', ''): v for k, v in pretrained_model.items() if 'resnet' in k}
                # print(pretrained_model.keys())
                self.base.load_state_dict(pretrained_model)

        self.depthnet = ResNetDilate(50)

        if self.adaptive_diated:

            self.adaptive_layers = Conv2D(512, 512 * 3, 3, padding=0)
            self.adaptive_bn = BatchNorm(512)

            self.adaptive_layers1 = Conv2D(1024, 1024 * 3, 3, padding=0)
            self.adaptive_bn1 = BatchNorm(1024)

        if self.deformable:
            self.deform_layer = DeformConv2d(512, 512, 3, padding=1, bias=False, modulation=True)

        # settings
        self.phase = phase
        self.num_classes = len(conf['lbls']) + 1
        self.anchors = np.array(conf.anchors)

        self.num_anchors = self.anchors.shape[0]

        self.prop_feats = Sequential(
            Conv2D(2048, 512, 3, padding=1))
        if self.use_dropout:
            self.dropout = Dropout(p=conf.dropout_rate, dropout_implementation='upscale_in_train')

        if self.drop_channel:
            self.dropout_channel = Dropout(p=0.3, dropout_implementation='upscale_in_train')

        # outputs
        self.cls = Conv2D(self.prop_feats[0]._num_filters, self.num_classes * self.num_anchors, 1)

        # bbox 2d
        self.bbox_x = Conv2D(self.prop_feats[0]._num_filters, self.num_anchors, 1)
        self.bbox_y = Conv2D(self.prop_feats[0]._num_filters, self.num_anchors, 1)
        self.bbox_w = Conv2D(self.prop_feats[0]._num_filters, self.num_anchors, 1)
        self.bbox_h = Conv2D(self.prop_feats[0]._num_filters, self.num_anchors, 1)

        # bbox 3d
        self.bbox_x3d = Conv2D(self.prop_feats[0]._num_filters, self.num_anchors, 1)
        self.bbox_y3d = Conv2D(self.prop_feats[0]._num_filters, self.num_anchors, 1)
        self.bbox_z3d = Conv2D(self.prop_feats[0]._num_filters, self.num_anchors, 1)
        self.bbox_w3d = Conv2D(self.prop_feats[0]._num_filters, self.num_anchors, 1)
        self.bbox_h3d = Conv2D(self.prop_feats[0]._num_filters, self.num_anchors, 1)
        self.bbox_l3d = Conv2D(self.prop_feats[0]._num_filters, self.num_anchors, 1)
        self.bbox_rY3d = Conv2D(self.prop_feats[0]._num_filters, self.num_anchors, 1)

        if self.corner_in_3d:
            self.bbox_3d_corners = Conv2D(self.prop_feats[0]._num_filters, self.num_anchors * 18, 1)  # 2 * 8 + 2
            self.bbox_vertices = Conv2D(self.prop_feats[0]._num_filters, self.num_anchors * 24, 1)  # 3 * 8
        elif self.use_corner:
            self.bbox_vertices = Conv2D(self.prop_feats[0]._num_filters, self.num_anchors * 24, 1)


        self.feat_stride = conf.feat_stride
        self.feat_size = calc_output_size(np.array(conf.crop_size), self.feat_stride)
        self.rois = locate_anchors(self.anchors, self.feat_size, conf.feat_stride, convert_tensor=True)
        self.rois = fluid.layers.cast(self.rois, 'float32')



    def forward(self, x, depth):

        batch_size = x.shape[0]

        x = self.base.conv1(x)
        depth = self.depthnet.conv1(depth)
        x = self.base.bn1(x)
        depth = self.depthnet.bn1(depth)
        x = relu(x)
        depth = relu(depth)
        x = self.base.maxpool(x)
        depth = self.depthnet.maxpool(depth)

        x = self.base.layer1(x)
        depth = self.depthnet.layer1(depth)
        # x = dynamic_local_filtering(x, depth, dilated=1) + dynamic_local_filtering(x, depth, dilated=2) + dynamic_local_filtering(x, depth, dilated=3)

        x = self.base.layer2(x)
        depth = self.depthnet.layer2(depth)

        if self.deformable:
            depth = self.deform_layer(depth)
            x = x * depth

        if self.adaptive_diated:
            weight = fluid.layers.adaptive_pool2d(x, pool_size=3, pool_type='max')
            weight = fluid.layers.reshape(self.adaptive_layers(weight), [-1, 512, 1, 3])
            weight = fluid.layers.softmax(weight, axis=3)
            x = dynamic_local_filtering(x, depth, dilated=1) * weight[:, :, :, 0:1] \
                + dynamic_local_filtering(x, depth, dilated=2) * weight[:, :, :, 1:2] \
                + dynamic_local_filtering(x, depth, dilated=3) * weight[:, :, :, 2:3]
            x = self.adaptive_bn(x)
            x = relu(x)
        else:
            x = dynamic_local_filtering(x, depth, dilated=1) + dynamic_local_filtering(x, depth, dilated=2) + dynamic_local_filtering(x, depth, dilated=3)

        if self.use_dropout and self.dropout_position == 'adaptive':
            x = self.dropout(x)

        if self.drop_channel:
            x = self.dropout_channel(x)

        x = self.base.layer3(x)
        depth = self.depthnet.layer3(depth)

        if self.adaptive_diated:
            weight = fluid.layers.adaptive_pool2d(x, pool_size=3, pool_type='max')
            weight = fluid.layers.reshape(self.adaptive_layers1(weight),[-1, 1024, 1, 3])
            weight = fluid.layers.softmax(weight, axis=3)
            x = dynamic_local_filtering(x, depth, dilated=1) * weight[:, :, :, 0:1] \
                + dynamic_local_filtering(x, depth, dilated=2) * weight[:, :, :, 1:2] \
                + dynamic_local_filtering(x, depth, dilated=3) * weight[:, :, :, 2:3]
            x = self.adaptive_bn1(x)
            x = relu(x)
        else:
            x = x * depth

        # TODO layers4 is error compared with torch
        x = self.base.layer4(x)
        depth = self.depthnet.layer4(depth)
        x = x * depth

        if self.use_dropout and self.dropout_position == 'early':
            x = self.dropout(x)

        prop_feats = self.prop_feats(x)
        prop_feats = relu(prop_feats)

        if self.use_dropout and self.dropout_position == 'late':
            prop_feats = self.dropout(prop_feats)

        cls = self.cls(prop_feats)

        # bbox 2d
        bbox_x = self.bbox_x(prop_feats)
        bbox_y = self.bbox_y(prop_feats)
        bbox_w = self.bbox_w(prop_feats)
        bbox_h = self.bbox_h(prop_feats)

        # bbox 3d
        bbox_x3d = self.bbox_x3d(prop_feats)
        bbox_y3d = self.bbox_y3d(prop_feats)
        bbox_z3d = self.bbox_z3d(prop_feats)
        bbox_w3d = self.bbox_w3d(prop_feats)
        bbox_h3d = self.bbox_h3d(prop_feats)
        bbox_l3d = self.bbox_l3d(prop_feats)
        bbox_rY3d = self.bbox_rY3d(prop_feats)
        # targets_dx, targets_dy, delta_z, scale_w, scale_h, scale_l, deltaRotY

        feat_h = cls.shape[2]
        feat_w = cls.shape[3]

        # reshape for cross entropy
        cls = fluid.layers.reshape(cls, [batch_size, self.num_classes, feat_h * self.num_anchors, feat_w])

        # score probabilities
        prob = fluid.layers.softmax(cls, axis=1)

        # reshape for consistency
        # although it's the same with x.view(batch_size, -1, 1) when c == 1, useful when c > 1
        bbox_x = flatten_tensor(fluid.layers.reshape(bbox_x, [batch_size, 1, feat_h * self.num_anchors, feat_w]))
        bbox_y = flatten_tensor(fluid.layers.reshape(bbox_y, [batch_size, 1, feat_h * self.num_anchors, feat_w]))
        bbox_w = flatten_tensor(fluid.layers.reshape(bbox_w, [batch_size, 1, feat_h * self.num_anchors, feat_w]))
        bbox_h = flatten_tensor(fluid.layers.reshape(bbox_h, [batch_size, 1, feat_h * self.num_anchors, feat_w]))

        bbox_x3d = flatten_tensor(fluid.layers.reshape(bbox_x3d, [batch_size, 1, feat_h * self.num_anchors, feat_w]))
        bbox_y3d = flatten_tensor(fluid.layers.reshape(bbox_y3d, [batch_size, 1, feat_h * self.num_anchors, feat_w]))
        bbox_z3d = flatten_tensor(fluid.layers.reshape(bbox_z3d, [batch_size, 1, feat_h * self.num_anchors, feat_w]))
        bbox_w3d = flatten_tensor(fluid.layers.reshape(bbox_w3d, [batch_size, 1, feat_h * self.num_anchors, feat_w]))
        bbox_h3d = flatten_tensor(fluid.layers.reshape(bbox_h3d, [batch_size, 1, feat_h * self.num_anchors, feat_w]))
        bbox_l3d = flatten_tensor(fluid.layers.reshape(bbox_l3d, [batch_size, 1, feat_h * self.num_anchors, feat_w]))
        bbox_rY3d = flatten_tensor(fluid.layers.reshape(bbox_rY3d, [batch_size, 1, feat_h * self.num_anchors, feat_w]))

        # bundle
        bbox_2d = fluid.layers.concat([bbox_x, bbox_y, bbox_w, bbox_h], axis=2)
        bbox_3d = fluid.layers.concat([bbox_x3d, bbox_y3d, bbox_z3d, bbox_w3d, bbox_h3d, bbox_l3d, bbox_rY3d], axis=2)

        if self.corner_in_3d:
            corners_3d = self.bbox_3d_corners(prop_feats)
            corners_3d = flatten_tensor(corners_3d.view(batch_size, 18, feat_h * self.num_anchors, feat_w))
            bbox_vertices = self.bbox_vertices(prop_feats)
            bbox_vertices = flatten_tensor(bbox_vertices.view(batch_size, 24, feat_h * self.num_anchors, feat_w))
        elif self.use_corner:
            bbox_vertices = self.bbox_vertices(prop_feats)
            bbox_vertices = flatten_tensor(bbox_vertices.view(batch_size, 24, feat_h * self.num_anchors, feat_w))

        feat_size = [feat_h, feat_w]

        cls = flatten_tensor(cls)
        prob = flatten_tensor(prob)

        if self.training:
            #print(cls.size(), prob.size(), bbox_2d.size(), bbox_3d.size(), feat_size)
            if self.corner_in_3d:
                return cls, prob, bbox_2d, bbox_3d, fluid.dygraph.to_variable(np.array(feat_size)).cuda(), bbox_vertices, corners_3d
            elif self.use_corner:
                return cls, prob, bbox_2d, bbox_3d, fluid.dygraph.to_variable(np.array(feat_size)).cuda(), bbox_vertices
            else:
                return cls, prob, bbox_2d, bbox_3d, fluid.dygraph.to_variable(np.array(feat_size))

        else:

            if self.feat_size[0] != feat_h or self.feat_size[1] != feat_w:
                self.feat_size = [feat_h, feat_w]
                self.rois = locate_anchors(self.anchors, self.feat_size, self.feat_stride, convert_tensor=True)
                self.rois = fluid.layers.cast(self.rois, 'float32')

            return cls, prob, bbox_2d, bbox_3d, feat_size, self.rois


def build(conf, phase='train'):

    train = phase.lower() == 'train'

    rpn_net = RPN(phase, conf)
    print(rpn_net)
    if train: rpn_net.train()
    else: rpn_net.eval()

    return rpn_net