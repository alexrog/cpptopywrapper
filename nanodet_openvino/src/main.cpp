#include "nanodet_openvino.h"
#include "ndarray_converter.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <librealsense2/rs.hpp>
#include "cv-helpers.hpp"
#include <iostream>
#include <sstream>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

NanoDet detector;// = NanoDet("nanodet_model/nanodet.xml", "MYRIAD", 32);
bool modelIsInit = false;

struct object_rect {
    int x;
    int y;
    int width;
    int height;
};

int resize_uniform(cv::Mat& src, cv::Mat& dst, cv::Size dst_size, object_rect& effect_area)
{
    int w = src.cols;
    int h = src.rows;
    int dst_w = dst_size.width;
    int dst_h = dst_size.height;
    dst = cv::Mat(cv::Size(dst_w, dst_h), CV_8UC3, cv::Scalar(0));

    float ratio_src = w * 1.0 / h;
    float ratio_dst = dst_w * 1.0 / dst_h;

    int tmp_w = 0;
    int tmp_h = 0;
    if (ratio_src > ratio_dst) {
        tmp_w = dst_w;
        tmp_h = floor((dst_w * 1.0 / w) * h);
    }
    else if (ratio_src < ratio_dst) {
        tmp_h = dst_h;
        tmp_w = floor((dst_h * 1.0 / h) * w);
    }
    else {
        cv::resize(src, dst, dst_size);
        effect_area.x = 0;
        effect_area.y = 0;
        effect_area.width = dst_w;
        effect_area.height = dst_h;
        return 0;
    }

    cv::Mat tmp;
    cv::resize(src, tmp, cv::Size(tmp_w, tmp_h));

    if (tmp_w != dst_w) {
        int index_w = floor((dst_w - tmp_w) / 2.0);
        for (int i = 0; i < dst_h; i++) {
            memcpy(dst.data + i * dst_w * 3 + index_w * 3, tmp.data + i * tmp_w * 3, tmp_w * 3);
        }
        effect_area.x = index_w;
        effect_area.y = 0;
        effect_area.width = tmp_w;
        effect_area.height = tmp_h;
    }
    else if (tmp_h != dst_h) {
        int index_h = floor((dst_h - tmp_h) / 2.0);
        memcpy(dst.data + index_h * dst_w * 3, tmp.data, tmp_w * tmp_h * 3);
        effect_area.x = 0;
        effect_area.y = index_h;
        effect_area.width = tmp_w;
        effect_area.height = tmp_h;
    }
    else {
        printf("error\n");
    }
    return 0;
}

std::vector<int> get_bboxes(const cv::Mat& bgr, const std::vector<BoxInfo>& bboxes, object_rect effect_roi)
{
    cv::Mat image = bgr.clone();

    int src_w = image.cols;
    int src_h = image.rows;
    int dst_w = effect_roi.width;
    int dst_h = effect_roi.height;
    float width_ratio = (float)src_w / (float)dst_w;
    float height_ratio = (float)src_h / (float)dst_h;
    std::vector<int> boundingboxes;


    for (size_t i = 0; i < bboxes.size(); i++)
    {
        const BoxInfo& bbox = bboxes[i];

        int point = (bbox.x1 - effect_roi.x) * width_ratio;
        boundingboxes.push_back(point);
        point = (bbox.y1 - effect_roi.y) * height_ratio;
        boundingboxes.push_back(point);
        point = (bbox.x2 - effect_roi.x) * width_ratio;
        boundingboxes.push_back(point);
        point = (bbox.y2 - effect_roi.y) * height_ratio;
        boundingboxes.push_back(point);
        boundingboxes.push_back(bbox.score * 100);
    }

    return boundingboxes;
}

void initModel(const char* modelpath, const char* device) {
    detector = NanoDet(modelpath, device, 32);
    modelIsInit = true;
}

bool isModelInit() {
    return modelIsInit;
}

std::vector<int> inference(cv::Mat image) {
    if(!modelIsInit) {
        std::cout << "Model has not been intialized. Call modelInit() from python." << std::endl;
        throw;
    }
    int height = detector.input_size[0];
    int width = detector.input_size[1];

    object_rect effect_roi;
    cv::Mat resized_img;
    resize_uniform(image, resized_img, cv::Size(width, height), effect_roi);
    auto results = detector.detect(resized_img, 0.4, 0.5);
    
    std::vector<int> bboxes = get_bboxes(image, results, effect_roi);
    return bboxes;
}

namespace py = pybind11;

PYBIND11_MODULE(nanodet_openvino, m) {
    NDArrayConverter::init_numpy();
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: nanodet_openvino

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";

    m.def("inference", &inference, R"pbdoc(
        perform inference using nanodet on given image 
    )pbdoc");

    m.def("initModel", &initModel, R"pbdoc(
        initialize nanodet model with modelpath, device 
    )pbdoc");

    m.def("isModelInit", &isModelInit, R"pbdoc(
        returns if the model is intialized or not
    )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
