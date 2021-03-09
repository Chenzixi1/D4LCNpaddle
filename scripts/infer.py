# -----------------------------------------
# python modules
# -----------------------------------------
import paddle.fluid as fluid
from easydict import EasyDict as edict
import sys
import numpy as np
import os
import pickle

# stop python from writing so much bytecode
sys.dont_write_bytecode = True
sys.path.append(os.getcwd())
np.set_printoptions(suppress=True)

from models.resnet_dilate import RPN
from lib.augmentations import Preprocess
from lib.util import pretty_print
from scripts.reader import Reader
from lib.rpn_util import *

def main():


    conf_path = 'pretrain/conf.pkl'
    weights_path = 'pretrain/D4LCN_40000_pre.pdparams'

    # load config
    with open(conf_path, 'rb') as file:
        conf = edict(pickle.load(file))

    # make directory
    test_path = os.path.join(os.getcwd(), 'data')
    results_path = os.path.join('predict')
    if not os.path.exists(results_path):
        os.mkdir(results_path)

    # print(pretty_print('conf', conf))

    # -----------------------------------------
    # test kitti
    # -----------------------------------------

    # test dataset reader: Kitti
    preprocess = Preprocess([conf.test_scale], conf.image_means, conf.image_stds, conf.depth_mean,
                            conf.depth_std, conf.use_rcnn_pretrain)
    conf.batch_size = 1
    test_reader = Reader(
        conf=conf,
        folder="data/training",
        list_file="data/val.txt",
        transform=preprocess,
        shuffle=True
    )
    dataset = test_reader.create_reader()

    # paddle mode set cuda
    use_log = True
    if_cuda = False
    place = fluid.CUDAPlace(0) if if_cuda else fluid.CPUPlace()
    with fluid.dygraph.guard(place):
        # -----------------------------------------
        # setup network
        # -----------------------------------------

        # net
        net = RPN(phase='train', conf=conf)

        # load weights
        para_state_dict, _ = fluid.load_dygraph(weights_path)
        net.load_dict(para_state_dict)

        # switch modes for evaluation
        net.eval()

        # init
        test_start = time()

        # test_kitti_3d
        for imind, (rgb, depth, imobjs) in enumerate(dataset()):

            rgb = fluid.dygraph.to_variable(np.array(rgb))
            depth = fluid.dygraph.to_variable(np.array(depth))

            cls, prob, bbox_2d, bbox_3d, feat_size, rois = net(rgb, depth)

            # forward test batch
            aboxes = im_detect_3d(cls, prob, bbox_2d, bbox_3d, feat_size, rois, conf, imobjs[0].p2, imobjs[0].scale_factor)

            # write the results in each txt
            results_write_txt(aboxes, imobjs, results_path, conf, 0.85)

            # display stats
            if (imind + 1) % 50 == 0:
                time_str, dt = compute_eta(test_start, imind + 1, test_reader.len)

                print_str = 'testing {}/{}, dt: {:0.3f}, eta: {}'.format(imind + 1, test_reader.len, dt, time_str)

                if use_log:
                    logging.info(print_str)
                else:
                    print(print_str, flush=True)

def results_write_txt(aboxes, imobjs, results_path, conf, threshold=0.75):
    # read in calib
    p2 = imobjs[0].p2
    p2_inv = imobjs[0].p2_inv
    impath = imobjs[0].path

    base_path, name, ext = file_parts(impath)

    file = open(os.path.join(results_path, name + '.txt'), 'w')
    text_to_write = ''

    for boxind in range(0, min(conf.nms_topN_post, aboxes.shape[0])):

        box = aboxes[boxind, :]
        score = box[4]
        cls = conf.lbls[int(box[5] - 1)]

        if score >= threshold:
            # 2D box
            x1 = box[0]
            y1 = box[1]
            x2 = box[2]
            y2 = box[3]
            width = (x2 - x1 + 1)
            height = (y2 - y1 + 1)

            # plot 3D box
            x3d = box[6]
            y3d = box[7]
            z3d = box[8]
            w3d = box[9]
            h3d = box[10]
            l3d = box[11]
            ry3d = box[12]

            # Inverse matrix and scale, to 3d camera coordinate
            coord3d = np.linalg.inv(p2).dot(np.array([x3d * z3d, y3d * z3d, 1 * z3d, 1]))
            # convert alpha into ry3d
            ry3d = convertAlpha2Rot(ry3d, coord3d[2], coord3d[0])

            step_r = 0.3 * math.pi
            r_lim = 0.01
            box_2d = np.array([x1, y1, width, height])

            z3d, ry3d, verts_best = hill_climb(p2, p2_inv, box_2d, x3d, y3d, z3d, w3d, h3d, l3d, ry3d,
                                               step_r_init=step_r, r_lim=r_lim)

            # predict a more accurate projection
            coord3d = np.linalg.inv(p2).dot(np.array([x3d * z3d, y3d * z3d, 1 * z3d, 1]))
            alpha = convertRot2Alpha(ry3d, coord3d[2], coord3d[0])

            x3d = coord3d[0]
            y3d = coord3d[1]
            z3d = coord3d[2]

            y3d += h3d / 2

            text_to_write += (
                    '{} -1 -1 {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} '
                    + '{:.6f} {:.6f}\n').format(cls, alpha, x1, y1, x2, y2, h3d, w3d, l3d, x3d, y3d, z3d,
                                                ry3d, score)

    print(text_to_write)

    file.write(text_to_write)
    file.close()

if __name__ == '__main__':
    main()