import numpy as np
import sys
import os
sys.dont_write_bytecode = True
sys.path.append(os.getcwd())
np.set_printoptions(suppress=True)

from visualdl import LogWriter
from scripts.reader import *
from lib.augmentations import *
from lib.loss.rpn_3d import *
from models.resnet_dilate import RPN

def main():

    conf = Config()
    conf_name = 'depth_guided_config'
    paths = init_training_paths(conf_name, conf.result_dir)
    init_log_file(paths.logs)
    writer = LogWriter(logdir=paths.logs)
    # TODO
    # visualdl --logdir ./output --port 8080

    # defaults
    start_iter = 0
    tracker = edict()
    iterator = None

    # dataset reader: Kitti
    transform = Augmentation(conf)
    train_reader = Reader(
        conf=conf,
        folder="data/training",
        list_file="data/train.txt",
        cache_folder='data/',
        transform=transform,
        shuffle=True
    )
    dataset = train_reader.create_reader()

    # generate_anchors & compute_bbox_stats has already done

    paths.output = os.path.join(paths.output, conf.result_dir)

    # store configuration
    with open(os.path.join(paths.output, 'conf.pkl'), 'wb') as file:
        pickle.dump(conf, file)

    # show configuration
    pretty = pretty_print('conf', conf)
    logging.info(pretty)

    # paddle mode set cuda
    if_cuda = False
    place = fluid.CUDAPlace(0) if if_cuda else fluid.CPUPlace()
    with fluid.dygraph.guard(place):

        # -----------------------------------------
        # network and loss
        # -----------------------------------------
        # training network
        rpn_net = RPN(phase='train', conf=conf)
        rpn_net.train()
        optimizer = fluid.optimizer.SGDOptimizer(learning_rate=conf.lr, parameter_list=rpn_net.parameters())

        # setup loss
        criterion_det = RPN_3D_loss(conf)

        # custom pretrained network
        if 'pretrained' in conf:
            load_weights(rpn_net, conf.pretrained)
        # resume training
        # if restore:
        #     start_iter = (restore - 1)
        #     resume_checkpoint(optimizer, rpn_net, paths.weights, restore)

        freeze_blacklist = None if 'freeze_blacklist' not in conf else conf.freeze_blacklist
        freeze_whitelist = None if 'freeze_whitelist' not in conf else conf.freeze_whitelist

        freeze_layers(rpn_net, freeze_blacklist, freeze_whitelist)

        optimizer.clear_gradients()

        start_time = time()

        # -----------------------------------------
        # train
        # -----------------------------------------

        iteration = start_iter
        while iteration < conf.max_iter:

            for batch_id, (rgb, depth, imobjs) in enumerate(dataset()):
                if iteration >= conf.max_iter:
                    break
                iteration += 1

                rgb = fluid.dygraph.to_variable(np.array(rgb))
                depth = fluid.dygraph.to_variable(np.array(depth))

                # forward
                if conf.corner_in_3d:
                    cls, prob, bbox_2d, bbox_3d, feat_size, bbox_vertices, corners_3d = rpn_net(rgb, depth)
                elif conf.use_corner:
                    cls, prob, bbox_2d, bbox_3d, feat_size, bbox_vertices = rpn_net(rgb, depth)
                else:
                    cls, prob, bbox_2d, bbox_3d, feat_size = rpn_net(rgb, depth)

                feat_size = feat_size[:2]  # feat_size = [32, 110]

                # loss
                if conf.corner_in_3d:
                    det_loss, det_stats = criterion_det(cls, prob, bbox_2d, bbox_3d, imobjs, feat_size, bbox_vertices, corners_3d)
                elif conf.use_corner:
                    det_loss, det_stats = criterion_det(cls, prob, bbox_2d, bbox_3d, imobjs, feat_size, bbox_vertices)
                else:
                    det_loss, det_stats = criterion_det(cls, prob, bbox_2d, bbox_3d, imobjs, feat_size)

                total_loss = det_loss
                stats = det_stats

                # backprop
                if total_loss > 0:
                    det_loss.backward()
                    det_loss = fluid.layers.cast(det_loss, 'float32')
                    optimizer.minimize(det_loss)
                    optimizer.clear_gradients()

                #  learning rate
                adjust_lr(conf, optimizer, iteration)

                # keep track of stats
                compute_stats(tracker, stats)
                Visual_stats(tracker, iteration, writer)

                # -----------------------------------------
                # display
                # -----------------------------------------
                if iteration % conf.display == 0 and iteration > start_iter:
                    # log results
                    log_stats(tracker, iteration, start_time, start_iter, conf.max_iter)

                    # reset tracker
                    tracker = edict()

                # -----------------------------------------
                # test network
                # -----------------------------------------
                if iteration % conf.snapshot_iter == 0 and iteration > start_iter:
                    # store checkpoint
                    save_checkpoint(optimizer, rpn_net, paths.weights, iteration)

                    # if conf.do_test:
                    #     rpn_net.eval()

if __name__ == "__main__":
    main()
