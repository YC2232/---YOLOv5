# ===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ===----------------------------------------------------------------------===#
import os
import time
import argparse
import requests
import json
import cv2
import numpy as np
import sophon.sail as sail
from postprocess_numpy import PostProcess
from utils import COLORS, COCO_CLASSES
import logging

logging.basicConfig(level=logging.INFO)


# sail.set_print_flag(1)

class YOLOv5:
    def __init__(self, bmodel_path, dev_id):
        # load bmodel
        self.net = sail.Engine(bmodel_path, dev_id, sail.IOMode.SYSO)
        logging.debug("load {} success!".format(args.bmodel))
        # self.handle = self.net.get_handle()
        self.handle = sail.Handle(args.dev_id)
        self.bmcv = sail.Bmcv(self.handle)
        self.graph_name = self.net.get_graph_names()[0]
        self.name = os.path.basename(bmodel_path)
        # get input
        self.input_name = self.net.get_input_names(self.graph_name)[0]
        self.input_dtype = self.net.get_input_dtype(self.graph_name, self.input_name)
        self.img_dtype = self.bmcv.get_bm_image_data_format(self.input_dtype)
        self.input_scale = self.net.get_input_scale(self.graph_name, self.input_name)
        self.input_shape = self.net.get_input_shape(self.graph_name, self.input_name)
        self.input_shapes = {self.input_name: self.input_shape}

        # get output
        self.output_names = self.net.get_output_names(self.graph_name)
        if len(self.output_names) not in [1, 3]:
            raise ValueError('only suport 1 or 3 outputs, but got {} outputs bmodel'.format(len(self.output_names)))

        self.output_tensors = {}
        self.output_scales = {}
        self.output_shapes = []
        for output_name in self.output_names:
            output_shape = self.net.get_output_shape(self.graph_name, output_name)
            output_dtype = self.net.get_output_dtype(self.graph_name, output_name)
            output_scale = self.net.get_output_scale(self.graph_name, output_name)
            output = sail.Tensor(self.handle, output_shape, output_dtype, True, True)
            self.output_tensors[output_name] = output
            self.output_scales[output_name] = output_scale
            self.output_shapes.append(output_shape)

        # check batch size
        self.batch_size = self.input_shape[0]
        suppoort_batch_size = [1, 2, 3, 4, 8, 16, 32, 64, 128, 256]
        if self.batch_size not in suppoort_batch_size:
            raise ValueError('batch_size must be {} for bmcv, but got {}'.format(suppoort_batch_size, self.batch_size))
        self.net_h = self.input_shape[2]
        self.net_w = self.input_shape[3]

        # init preprocess
        self.use_resize_padding = True
        self.use_vpp = False
        self.ab = [x * self.input_scale / 255. for x in [1, 0, 1, 0, 1, 0]]

        # init postprocess
        self.conf_thresh = args.conf_thresh
        self.nms_thresh = args.nms_thresh
        if 'use_cpu_opt' in getattr(args, '__dict__', {}):
            self.use_cpu_opt = args.use_cpu_opt
        else:
            self.use_cpu_opt = False

        self.agnostic = False
        self.multi_label = True
        self.max_det = 1000
        self.postprocess = PostProcess(
            conf_thresh=self.conf_thresh,
            nms_thresh=self.nms_thresh,
            agnostic=self.agnostic,
            multi_label=self.multi_label,
            max_det=self.max_det,
        )

        # init time
        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0

    def init(self):
        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0

    def preprocess_bmcv(self, input_bmimg):
        rgb_planar_img = sail.BMImage(self.handle, input_bmimg.height(), input_bmimg.width(),
                                      sail.Format.FORMAT_RGB_PLANAR, sail.DATA_TYPE_EXT_1N_BYTE)
        self.bmcv.convert_format(input_bmimg, rgb_planar_img)
        resized_img_rgb, ratio, txy = self.resize_bmcv(rgb_planar_img)
        preprocessed_bmimg = sail.BMImage(self.handle, self.net_h, self.net_w, sail.Format.FORMAT_RGB_PLANAR,
                                          self.img_dtype)
        self.bmcv.convert_to(resized_img_rgb, preprocessed_bmimg, ((self.ab[0], self.ab[1]), \
                                                                   (self.ab[2], self.ab[3]), \
                                                                   (self.ab[4], self.ab[5])))
        return preprocessed_bmimg, ratio, txy

    def resize_bmcv(self, bmimg):
        """
        resize for single sail.BMImage
        :param bmimg:
        :return: a resize image of sail.BMImage
        """
        img_w = bmimg.width()
        img_h = bmimg.height()
        if self.use_resize_padding:
            r_w = self.net_w / img_w
            r_h = self.net_h / img_h
            if r_h > r_w:
                tw = self.net_w
                th = int(r_w * img_h)
                tx1 = tx2 = 0
                ty1 = int((self.net_h - th) / 2)
                ty2 = self.net_h - th - ty1
            else:
                tw = int(r_h * img_w)
                th = self.net_h
                tx1 = int((self.net_w - tw) / 2)
                tx2 = self.net_w - tw - tx1
                ty1 = ty2 = 0

            ratio = (min(r_w, r_h), min(r_w, r_h))
            txy = (tx1, ty1)
            attr = sail.PaddingAtrr()
            attr.set_stx(tx1)
            attr.set_sty(ty1)
            attr.set_w(tw)
            attr.set_h(th)
            attr.set_r(114)
            attr.set_g(114)
            attr.set_b(114)

            preprocess_fn = self.bmcv.vpp_crop_and_resize_padding if self.use_vpp else self.bmcv.crop_and_resize_padding
            resized_img_rgb = preprocess_fn(bmimg, 0, 0, img_w, img_h, self.net_w, self.net_h, attr)
        else:
            r_w = self.net_w / img_w
            r_h = self.net_h / img_h
            ratio = (r_w, r_h)
            txy = (0, 0)
            preprocess_fn = self.bmcv.vpp_resize if self.use_vpp else self.bmcv.resize
            resized_img_rgb = preprocess_fn(bmimg, self.net_w, self.net_h)
        return resized_img_rgb, ratio, txy

    def predict(self, input_tensor, img_num):
        """
        ensure output order: loc_data, conf_preds, mask_data, proto_data
        Args:
            input_tensor:
        Returns:
        """
        input_tensors = {self.input_name: input_tensor}
        self.net.process(self.graph_name, input_tensors, self.input_shapes, self.output_tensors)
        if self.use_cpu_opt:
            out = self.output_tensors
        else:
            outputs_dict = {}
            for name in self.output_names:
                # outputs_dict[name] = self.output_tensors[name].asnumpy()[:img_num] * self.output_scales[name]
                outputs_dict[name] = self.output_tensors[name].asnumpy()[:img_num]
            # resort
            out_keys = list(outputs_dict.keys())
            ord = []
            for n in self.output_names:
                for i, k in enumerate(out_keys):
                    if n in k:
                        ord.append(i)
                        break
            out = [outputs_dict[out_keys[i]] for i in ord]
        return out

    def __call__(self, bmimg_list):
        img_num = len(bmimg_list)
        ori_size_list = []
        ori_w_list = []
        ori_h_list = []
        ratio_list = []
        txy_list = []
        if self.batch_size == 1:
            ori_h, ori_w = bmimg_list[0].height(), bmimg_list[0].width()
            ori_size_list.append((ori_w, ori_h))
            ori_w_list.append(ori_w)
            ori_h_list.append(ori_h)
            start_time = time.time()
            preprocessed_bmimg, ratio, txy = self.preprocess_bmcv(bmimg_list[0])
            self.preprocess_time += time.time() - start_time
            ratio_list.append(ratio)
            txy_list.append(txy)

            input_tensor = sail.Tensor(self.handle, self.input_shape, self.input_dtype, False, False)
            self.bmcv.bm_image_to_tensor(preprocessed_bmimg, input_tensor)

        else:
            BMImageArray = eval('sail.BMImageArray{}D'.format(self.batch_size))
            bmimgs = BMImageArray()
            for i in range(img_num):
                ori_h, ori_w = bmimg_list[i].height(), bmimg_list[i].width()
                ori_size_list.append((ori_w, ori_h))
                ori_w_list.append(ori_w)
                ori_h_list.append(ori_h)
                start_time = time.time()
                preprocessed_bmimg, ratio, txy = self.preprocess_bmcv(bmimg_list[i])
                self.preprocess_time += time.time() - start_time
                ratio_list.append(ratio)
                txy_list.append(txy)
                bmimgs[i] = preprocessed_bmimg.data()
            input_tensor = sail.Tensor(self.handle, self.input_shape, self.input_dtype, False, False)
            self.bmcv.bm_image_to_tensor(bmimgs, input_tensor)

        start_time = time.time()
        outputs = self.predict(input_tensor, img_num)
        self.inference_time += time.time() - start_time

        start_time = time.time()
        if self.use_cpu_opt:
            cpu_opt_process = sail.algo_yolov5_post_cpu_opt(self.output_shapes)
            results = cpu_opt_process.process(outputs, ori_w_list, ori_h_list, [self.conf_thresh] * self.batch_size,
                                              [self.nms_thresh] * self.batch_size, True, True)
            results = np.array(results)
        else:
            results = self.postprocess(outputs, ori_size_list, ratio_list, txy_list)
        self.postprocess_time += time.time() - start_time

        return results

def get_rtsp_stream_url(url, channel_code, stream_type=0):
    payload = json.dumps({
        "channelCode": channel_code,
        "streamType": stream_type
    })
    headers = {'Content-Type': 'application/json'}

    response = requests.request("POST", url, headers=headers, data=payload)


    # 检查响应状态
    if response.status_code == 200:
        # 解析 JSON 响应
        response_data = response.json()
        # 提取并返回 URL
        return response_data.get('data', {}).get('url', None)

    else:
        print(f"Error: Unable to fetch URL, Status Code: {response.status_code}")
        return None


def draw_bmcv_car(bmcv, bmimg, boxes, classes_ids=None, conf_scores=None, save_path=""):

    CUSTOM_CLASS = "Car"

    if isinstance(bmimg, sail.BMImage):
        img_bgr_planar = bmimg.asmat()  # Convert BMImage to NumPy array
    else:
        img_bgr_planar = bmimg  # Already a NumPy array


    for idx in range(len(boxes)):
        # TODO
        x1, y1, x2, y2 = boxes[idx, :].astype(np.int32).tolist()
        score = conf_scores[idx]
        # 如果置信度小于0.25，则跳过该检测
        if conf_scores[idx]  < 0.25:
            continue


        class_id = int(classes_ids[idx])
        #detect car
        if class_id==2:
            color = np.array(COLORS[class_id % len(COLORS)]).astype(np.uint8).tolist()
            label = f'{CUSTOM_CLASS}:{score:.2f}'

            # Use OpenCV to draw rectangle and text
            cv2.rectangle(img_bgr_planar, (x1, y1), (x2, y2), [0,0,255], 3)
            cv2.putText(img_bgr_planar, "model 2 : Car", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, [0,0,255], 3)


    return img_bgr_planar



def draw_bmcv_extinguisher(bmcv, bmimg, boxes, classes_ids=None, conf_scores=None, save_path=""):
    CUSTOM_CLASS = "Extinguisher"
    thickness = 2
    text_thickness = 1
    font_scale = 0.5

    if isinstance(bmimg, sail.BMImage):
        img_bgr_planar = bmimg.asmat()  # Convert BMImage to NumPy array
    else:
        img_bgr_planar = bmimg  # Already a NumPy array


    for idx in range(len(boxes)):
        #TODO
        x1, y1, x2, y2 = boxes[idx, :].astype(np.int32).tolist()
        score = conf_scores[idx]
        # 如果置信度小于0.25，则跳过该检测
        if conf_scores[idx]  < 0.25:
            continue


        class_id = int(classes_ids[idx])
        #detect car

        color = np.array(COLORS[class_id % len(COLORS)]).astype(np.uint8).tolist()
        label = f'{CUSTOM_CLASS}:{score:.2f}'

        # Use OpenCV to draw rectangle and text
        cv2.rectangle(img_bgr_planar, (x1, y1), (x2, y2), [255,0,0], 3)
        cv2.putText(img_bgr_planar, "model 1 : Extinguisher", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, [255,0,0], 3)


    return img_bgr_planar


def main(args):

    #request video url

    stream_url=get_rtsp_stream_url(args.url,args.channelCode)

    # creat save path
    output_dir = "./results"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_img_dir = os.path.join(output_dir, 'images')
    if not os.path.exists(output_img_dir):
        os.mkdir(output_img_dir)

    # 创建模型名称到绘图函数的映射
    model_to_draw_function_map = {
        'yolov5s_v6.1_3output_fp32_1b_extinguisher.bmodel': draw_bmcv_extinguisher,
        'yolov5s_v6.1_3output_fp32_1b.bmodel': draw_bmcv_car,
    }
    # 初始化模型
    models = [YOLOv5(bmodel, args.dev_id) for bmodel in args.bmodel]

    # 为每个模型找到对应的绘图函数
    draw_functions = [model_to_draw_function_map[model.name] for model in models]

    batch_size = models[0].batch_size

    handle = sail.Handle(args.dev_id)
    bmcv = sail.Bmcv(handle)

    decode_time = 0.0

    decoder = sail.MultiDecoder(10, 0, 0)
    ch_idx = decoder.add_channel(stream_url, 0)

    video_name = os.path.splitext(os.path.split(stream_url)[1])[0]

    save_video_path = os.path.join(output_dir, video_name + '_processed.avi')
    print("-------------------------------------------------")
    print("saved in ",save_video_path)
    print("-------------------------------------------------")
    #Acquire video data dynamically
    cap = cv2.VideoCapture(stream_url)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out_video = cv2.VideoWriter(save_video_path, fourcc,fps, (width, height))

    cn = 0
    frame_list = []
    end_flag = False
    while not end_flag:
        frame = sail.BMImage()
        start_time = time.time()
        frame = decoder.read(ch_idx)
        decode_time += time.time() - start_time
        ret = False
        if ret:  # differ from cv.
            # end_flag = True
            print(222)
        else:
            frame_list.append(frame)
        if (len(frame_list) == batch_size or end_flag) and len(frame_list):

            if len(frame_list) > 0:
                bmcv1 = frame_list[0]

            for model_index, model in enumerate(models):
                results = model(frame_list)
                draw_func = draw_functions[model_index]

                for i, frame in enumerate(frame_list):
                    det = results[i]
                    cn += 1
                    logging.info("{}, det nums: {}".format(cn, det.shape[0]))
                    save_path = os.path.join(output_img_dir, video_name + '_' + str(cn) + '.jpg')

                    # 更新 bmcv1
                    bmcv1 = draw_func(bmcv, bmcv1, det[:, :4], classes_ids=det[:, -1], conf_scores=det[:, -2],
                                      save_path=save_path)

            if isinstance(bmcv1, sail.BMImage):
                res = bmcv1.asmat()# Convert BMImage to NumPy array
                out_video.write(res)
            else:
                out_video.write(bmcv1)




            frame_list.clear()
    out_video.release()
    logging.info("result saved in {}".format(output_img_dir))


def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--input', type=str, default='../datasets/test', help='path of input')
    parser.add_argument('--bmodel', nargs='+', type=str,
                        default=[ '/data/sophon-demo/sample/YOLOv5/models/BM1684X/yolov5s_v6.1_3output_fp32_1b_extinguisher.bmodel','/data/sophon-demo/sample/YOLOv5/models/BM1684X/yolov5s_v6.1_3output_fp32_1b.bmodel'
                                ],
                        help='paths of bmodels')
    parser.add_argument('--dev_id', type=int, default=0, help='dev id')
    parser.add_argument('--conf_thresh', type=float, default=0.1, help='confidence threshold')
    parser.add_argument('--nms_thresh', type=float, default=0.6, help='nms threshold')
    parser.add_argument('--use_cpu_opt', action="store_true", default=False, help='accelerate cpu postprocess')
    parser.add_argument('--channelCode', type=str, default='', help='channel code')
    parser.add_argument('--url', type=str, required=True, help='URL to get the stream')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argsparser()
    main(args)
    print('all done.')
