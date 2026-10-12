#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>

void calibrateCameraFromVideo(const std::string& video_path) {
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) return;

    cv::Size board_size(7, 7); 
    float square_size = 30.0f;  


    std::vector<std::vector<cv::Point2f>> image_points;
    std::vector<std::vector<cv::Point3f>> object_points;

    // 3D 世界坐标点，Z 轴都为 0
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

        if (frame_count % 30 != 0) { frame_count++; continue; }
        
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        std::vector<cv::Point2f> corners;
        
        // 寻找棋盘格角点
        bool found = cv::findCirclesGrid(gray, board_size, corners,
    cv::CALIB_CB_SYMMETRIC_GRID);
            
        if (found) {
            // 亚像素级优化，提高精度
            cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
                cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001));
            
            if (found_count < 30) { 
            image_points.push_back(corners);
            object_points.push_back(obj);
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

    std::cout << "重投影误差 (RMS): " << rms << " 像素" << std::endl;
    std::cout << "内参矩阵 K = \n" << camera_matrix << std::endl;
    std::cout << "畸变系数 D = \n" << dist_coeffs << std::endl;
}
}
int main() {
    // 填入你的标定视频相对路径
    calibrateCameraFromVideo("../../../data/raw/calibration_video.avi");
    return 0;
}