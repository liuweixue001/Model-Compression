// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "net.h"

#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <string>
using std::vector;
using std::string;
using std::endl;
using std::cout;
static const vector<string> classes{ "bird", "boat", "cake", "jellyfish", "king_crab"};
static int detect_mobilenet_v3(const cv::Mat& bgr, std::vector<float>& cls_scores)
{
    ncnn::Net mobilenet_v3;

    mobilenet_v3.opt.use_int8_inference = true;
    mobilenet_v3.load_param("./model/quanmodel.param");
    mobilenet_v3.load_model("./model/quanmodel.bin");
    
    ncnn::Mat in = ncnn::Mat::from_pixels(bgr.data, ncnn::Mat::PIXEL_RGB, bgr.cols, bgr.rows);
    const float mean_vals[3] = { 0.485f, 0.456f, 0.406f };
    const float std_vals[3] = { 1 / 0.229f, 1 / 0.224f, 1 / 0.225f };
    const float norm_vals[3] = { 1 / 255.f, 1 / 255.f, 1 / 255.f };
    //in.substract_mean_normalize(0, norm_vals);
    //in.substract_mean_normalize(mean_vals, std_vals);
    const float* ptr1 = in.channel(0);
    const float* ptr2 = in.channel(1);
    const float* ptr3 = in.channel(2);
    cout << ptr1[424] << " "<< ptr2[424] << " " << ptr3[424] << endl;
    ncnn::Extractor ex = mobilenet_v3.create_extractor();
    
    ex.input("input", in);
    ncnn::Mat out;
    ex.extract("output", out);

     //manually call softmax on the fc output
     //convert result into probability
     //skip if your model already has softmax operation
    {
        ncnn::Layer* softmax = ncnn::create_layer("Softmax");

        ncnn::ParamDict pd;
        softmax->load_param(pd);

        softmax->forward_inplace(out, mobilenet_v3.opt);

        delete softmax;
    }

    out = out.reshape(out.w * out.h * out.c);

    cls_scores.resize(out.w);
    for (int j = 0; j < out.w; j++)
    {
        cls_scores[j] = out[j];
    }

    return 0;
}

static int print_topk(const std::vector<float>& cls_scores, int topk)
{
    // partial sort topk with index
    int size = cls_scores.size();
    std::vector<std::pair<float, int> > vec;
    vec.resize(size);
    for (int i = 0; i < size; i++)
    {
        vec[i] = std::make_pair(cls_scores[i], i);
    }

    std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
        std::greater<std::pair<float, int> >());

    // print topk and score
    for (int i = 0; i < topk; i++)
    {
        float score = vec[i].first;
        int index = vec[i].second;
        cout.precision(3);
        cout << classes[index] << " = " << score << endl;
    }

    return 0;
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }

    const char* imagepath = argv[1];
    cv::Mat m = cv::imread(imagepath);
    cv::resize(m, m, cv::Size(224, 224));
    cv::cvtColor(m, m, cv::COLOR_BGR2RGB);
    cout << m.ptr<cv::Vec3b>(1)[200]<< endl;
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    std::vector<float> cls_scores;
    detect_mobilenet_v3(m, cls_scores);

    print_topk(cls_scores, 3);

    return 0;
}
