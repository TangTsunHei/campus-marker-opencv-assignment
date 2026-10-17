#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>

void calibrateCameraFromVideo(const std::string& video_path) {
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) return;

    cv::Size board_size(7, 7); //告诉程序，标定板上有 7x7 个圆点
    float square_size = 30.0f;  //圆点之间的物理间距是 30mm


    std::vector<std::vector<cv::Point2f>> image_points;
    std::vector<std::vector<cv::Point3f>> object_points;

    // 构建7×7的3D 世界坐标点，Z 轴都为 0(因为标定板是一个平面)
    std::vector<cv::Point3f> obj;
    for (int i = 0; i < board_size.height; ++i)
        for (int j = 0; j < board_size.width; ++j)
            obj.push_back(cv::Point3f(j * square_size, i * square_size, 0));

    cv::Mat frame, gray;
    int frame_count = 0;
    int found_count = 0;
    cv::Size board_image_size;
    // 声明一个变量用来保存第一帧的尺寸

    // 每 30 帧处理一次，防止太慢
    while (cap.read(frame)) {
        //如果尺寸为空，就获取第一帧的尺寸
        if (board_image_size.empty()) {
            board_image_size = frame.size();
        }

        if (frame_count % 30 != 0) { frame_count++; continue; }//framecount代表着真实视频帧，
        //只要视频帧数不是30的倍数就直接continue，跳过,不进入下面的处理
        
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        std::vector<cv::Point2f> corners;//在找到角点后corners里保存 49 个点在图像中的像素坐标 (u, v)
        
        // 寻找棋盘格角点（这里的角点和装甲板识别中的四个顶点不同，这里的角点是标定板上49个圆点的圆心）
        bool found = cv::findCirclesGrid(gray, board_size, corners,
    cv::CALIB_CB_SYMMETRIC_GRID);
            
        if (found) {
            // 亚像素级优化，提高精度
            cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
                cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001));
            
            if (found_count < 30) { 
            image_points.push_back(corners);
            object_points.push_back(obj);//把同一个物理坐标数组obj,复制30份,分别塞进object_points里的30个位置
            found_count++;
            std::cout << "有效帧 " << found_count << ": " << frame_count << std::endl;
        } else {
            // 已经够了，直接跳出循环
            break; 
        }
    }

    // 开始标定
    cv::Mat camera_matrix, dist_coeffs;
    std::vector<cv::Mat> rvecs, tvecs;


    double rms = cv::calibrateCamera(object_points, image_points, board_image_size,
                                     camera_matrix, dist_coeffs, rvecs, tvecs);

    std::cout << "重投影误差 (RMS): " << rms << " 像素" << std::endl;//反应标定效果，rms越大表示效果越差
    std::cout << "内参矩阵 K = \n" << camera_matrix << std::endl;
    std::cout << "畸变系数 D = \n" << dist_coeffs << std::endl;
}
}
int main() {
    // 填入标定视频相对路径
    calibrateCameraFromVideo("../../../data/raw/calibration_video.avi");
    return 0;
}