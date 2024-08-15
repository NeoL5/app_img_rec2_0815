# -*- coding: utf-8 -*-
# @Author  : Zhiyi Leung
# @Time    : 2024/8/15 11:31
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase


class LineDetectionTransformer(VideoTransformerBase):
    def transform(self, frame):
        # 将帧转换为OpenCV格式
        img = frame.to_ndarray(format="bgr24")

        # 将图像转为灰度
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 使用高斯模糊平滑图像
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # 使用Canny边缘检测器
        edges = cv2.Canny(blur, 50, 150, apertureSize=3)

        # 使用Hough变换检测直线
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)

        # 在原图上绘制检测到的直线
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return img


# 使用Streamlit构建Web界面
st.title("实时直线检测")
st.write("这是一个使用 OpenCV 和 Streamlit 构建的实时直线检测程序。")

webrtc_streamer(key="line-detection", video_transformer_factory=LineDetectionTransformer)
