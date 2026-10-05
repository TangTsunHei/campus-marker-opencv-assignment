#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct LightBar {
    cv::RotatedRect rectangle;
    cv::Point2f top;
    cv::Point2f bottom;
    float length = 0.0F;
    float angle_degrees = 0.0F;
};

struct Armor {
    std::size_t left = 0;
    std::size_t right = 0;
    float score = 0.0F;
    // 顺序固定为左上、右上、右下、左下，方便后续绘制或姿态解算。
    std::array<cv::Point2f, 4> corners;
};

struct FrameResult {
    cv::Mat annotated;
    std::size_t armor_count = 0;
};//删除了std::array<cv::Mat, 10> stages; ，因为在后续的处理过程中，暂时不需要保留每个阶段的图像，只需要最终的标注图像和最终装甲板识别视频

cv::Point pixelPoint(const cv::Point2f& point) {
    return {cvRound(point.x), cvRound(point.y)};
}  
//用于将浮点值point坐标转换成整型point坐标，方便后续函数使用

void drawRotatedRect(cv::Mat& image, const cv::RotatedRect& rectangle,
                     const cv::Scalar& color, int thickness = 2) {
    cv::Point2f vertices[4];
    rectangle.points(vertices);//points为rotatedrect内置函数，通过rotatedrect的成员变量计算四个顶点的坐标
    for (int i = 0; i < 4; ++i) {
        cv::line(image, pixelPoint(vertices[i]), pixelPoint(vertices[(i + 1) % 4]),
                 color, thickness, cv::LINE_AA);
    }
}
//根据输入图像绘制旋转矩形框，rotatedrect中有三个成员变量，center,angle,size(宽高)

bool makeLightBar(const std::vector<cv::Point>& contour, LightBar& bar) {
    if (cv::contourArea(contour) < 15.0) {
        return false;   //计算轮廓面积排除较小轮廓，避免噪点误判成待选灯条
    }

    const cv::RotatedRect rectangle = cv::minAreaRect(contour);//返回contour(四个顶点)最小外接矩形
    const float length = std::max(rectangle.size.width, rectangle.size.height);
    const float width = std::min(rectangle.size.width, rectangle.size.height);
    // 灯条偶尔会在二值图中变亮、变粗；保留长宽比至少 1.9 的区域，
    // 后续仍由两灯条的相对位置排除不合理的组合。
    if (width <= 0.0F || length < 18.0F || length / width < 1.9F) {
        return false;
    }

    cv::Point2f vertices[4];
    rectangle.points(vertices);
    cv::Point2f longest_edge;
    float longest_squared = -1.0F;
    for (int i = 0; i < 4; ++i) {
        const cv::Point2f edge = vertices[(i + 1) % 4] - vertices[i];
        const float squared = edge.dot(edge);
        if (squared > longest_squared) {
            longest_squared = squared;
            longest_edge = edge;
        }
    }
    if (longest_edge.y < 0.0F) {
        longest_edge *= -1.0F;
    }//保证所有计算出来的最长边（向量）均指向图片下方，统一指向（opencv图像坐标有点在左上角向下y增大）
    const cv::Point2f axis = longest_edge * (1.0F / std::sqrt(longest_squared));//计算单位方向向量
    const float angle_degrees = std::atan2(axis.x, axis.y) * 180.0F / CV_PI;//通常atan2(y, x)计算的是向量相对于x轴正方向的角度，此处参数位置调换计算的是相对于y轴的倾角
    if (std::abs(angle_degrees) > 30.0F) {
        return false;
    }//将倾角过大的矩形框排除在待选lightbar之外

    bar.rectangle = rectangle;
    bar.top = rectangle.center - axis * (length / 2.0F);
    bar.bottom = rectangle.center + axis * (length / 2.0F);
    bar.length = length;
    bar.angle_degrees = angle_degrees;
    return true;
}

// 第八步：灯条配对。所有长度和间距都用平均灯条长度归一化，
// 因而目标在画面里变大、变小时，阈值仍有相同的几何意义。
std::vector<Armor> pairLightBars(const std::vector<LightBar>& bars,
                                const cv::Mat& bright_pixels,
                                float max_gap_ratio) {
    std::vector<Armor> candidates;
    for (std::size_t left = 0; left < bars.size(); ++left) {
        for (std::size_t right = left + 1; right < bars.size(); ++right) {
            const LightBar& a = bars[left];
            const LightBar& b = bars[right];
            const float mean_length = (a.length + b.length) / 2.0F;
            const float length_ratio = std::max(a.length, b.length) /
                                       std::min(a.length, b.length);
            const float gap_ratio = (b.rectangle.center.x - a.rectangle.center.x) /
                                    mean_length;
            const float height_difference =
                std::abs(a.rectangle.center.y - b.rectangle.center.y) / mean_length;
            const float angle_difference =
                std::abs(a.angle_degrees - b.angle_degrees);

            if (length_ratio > 1.7F || gap_ratio < 1.0F ||
                gap_ratio > max_gap_ratio ||
                height_difference > 0.5F || angle_difference > 25.0F) {
                continue;
            }

            // 两根灯条之间如果已有一根同高度、相近长度的灯条，
            // 当前组合很可能跨过了另一块装甲板，应当丢弃。
            bool crosses_bar = false;
            const float pair_center_y =
                (a.rectangle.center.y + b.rectangle.center.y) / 2.0F;
            for (std::size_t middle = left + 1; middle < right; ++middle) {
                const LightBar& other = bars[middle];
                if (std::abs(other.rectangle.center.y - pair_center_y) <=
                        0.5F * mean_length &&
                    other.length >= 0.5F * std::min(a.length, b.length)) {
                    crosses_bar = true;
                    break;
                }
            }
            if (crosses_bar) {
                continue;
            }

            // 只看两根灯条之间的内部区域，避开灯条本身：这两段素材的
            // 真正装甲板中有浅色数字，而旁边的空支架通常很暗。
            // 用亮像素比例排除“灯条 + 空支架”的误配，不需要识别数字几。
            const float left_x = a.rectangle.center.x +
                                 0.15F * (b.rectangle.center.x - a.rectangle.center.x);
            const float right_x = b.rectangle.center.x -
                                  0.15F * (b.rectangle.center.x - a.rectangle.center.x);
            const int x1 = std::max(0, cvRound(left_x));
            const int x2 = std::min(bright_pixels.cols, cvRound(right_x));
            const int y1 = std::max(0, cvRound(pair_center_y - 0.45F * mean_length));
            const int y2 = std::min(bright_pixels.rows,
                                    cvRound(pair_center_y + 0.45F * mean_length));
            if (x1 >= x2 || y1 >= y2) {
                continue;
            }
            const cv::Rect interior(x1, y1, x2 - x1, y2 - y1); //rect(左上x坐标，左上y坐标，宽，高)
            const float bright_ratio = static_cast<float>(
                cv::countNonZero(bright_pixels(interior))) / interior.area();
            if (bright_ratio < 0.05F) {
                continue;
            }

            Armor armor;
            armor.left = left;
            armor.right = right;
            armor.score = std::abs(gap_ratio - 2.0F) +
                          0.8F * height_difference +
                          0.6F * (length_ratio - 1.0F) +
                          0.03F * angle_difference - 0.8F * bright_ratio;
            armor.corners = {a.top, b.top, b.bottom, a.bottom};
            candidates.push_back(armor);
        }
    }

    std::sort(candidates.begin(), candidates.end(), [](const Armor& a, const Armor& b) {
        return a.score < b.score;
    });

    // 每根灯条最多属于一块装甲板。按综合分数从小到大挑选时，
    // 如果候选对中的任一根灯条已经被选中，就跳过这一对；
    // 不论它在上一对中处于左侧还是右侧，都算已被占用。
    std::vector<bool> used(bars.size(), false);
    std::vector<Armor> selected;
    for (const Armor& armor : candidates) {
        if (used[armor.left] || used[armor.right]) {
            continue;
        }
        selected.push_back(armor);
        used[armor.left] = true;
        used[armor.right] = true;
    }
    return selected;
}


FrameResult processFrame(const cv::Mat& frame) {
    FrameResult result;
    // 第零步：BGR 原图。OpenCV 读取的视频帧默认按蓝、绿、红排列；
    // 保留一份原图，方便与后续每个处理阶段直接对照。
    cv::Mat image_progress_bgr = frame.clone();

    // 第一步：滤波去噪。中值滤波用邻域的中间值替换孤立噪点，
    // 3×3 窗口较小，能尽量保留狭窄的蓝色灯条。
    cv::Mat image_progress_denoise;
    cv::medianBlur(image_progress_bgr, image_progress_denoise, 3);

    // 第二步：亮度与颜色增强。先转到 HSV，把亮度 V 单独用 CLAHE
    // 做局部对比度增强；H（色相）和 S（饱和度）保持原样，便于找蓝色。
    /*  //待改动处2-----------------------------------------------------------
    cv::Mat hsv_enhanced;
    cv::cvtColor(image_progress_denoise, hsv_enhanced, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> hsv_channels;
    cv::split(hsv_enhanced, hsv_channels);
    cv::createCLAHE(2.0, cv::Size(8, 8))->apply(hsv_channels[2], hsv_channels[2]);
    cv::merge(hsv_channels, hsv_enhanced);
    cv::Mat image_progress_enhance;
    cv::cvtColor(hsv_enhanced, image_progress_enhance, cv::COLOR_HSV2BGR);
    */  //待改动处2（hsv通道没有保留的必要）-------------------------------------^

    // 第三步：二值化。参考 auto-aim-new 的颜色预处理思路，
    // 直接分离蓝色和红色通道，再把它们合成为一个单通道颜色差异图。
    // 灯条的目标颜色会在对应通道中更亮，灰白背景在两个通道中更接近，
    // 因而用固定阈值即可得到干净的候选区域。
    /*    //待改动处1----------------------------------------------------------
    std::vector<cv::Mat> bgr_channels;
    cv::split(image_progress_denoise, bgr_channels);
    const cv::Mat& blue_channel = bgr_channels[0];
    const cv::Mat& red_channel = bgr_channels[2];
    cv::Mat red_blue_difference;
    cv::absdiff(blue_channel, red_channel, red_blue_difference);
    cv::Mat image_progress_binary;
    cv::threshold(red_blue_difference, image_progress_binary, 40, 255,
                  cv::THRESH_BINARY);
    */   //待改动处1----------------------------------------------------------
    cv::Mat gray;   //因为作业的标识物灯条为白色，装甲板为灰黑色，所以放弃学长模板中的bgr红蓝通道分离，而是直接选用灰度图
    cv::cvtColor(image_progress_denoise, gray, cv::COLOR_BGR2GRAY);

    cv::Mat image_progress_binary;
    // 白色灯条比较亮，阈值可以取 180~220，根据实际视频调
    cv::threshold(gray, image_progress_binary, 200, 255, cv::THRESH_BINARY);
    // 改动替换部分1----------------------------------------------------------^
    // 第四步：形态学处理。闭运算等价于先膨胀、再腐蚀，能够连接
    // 灯条内部的小断点；开运算等价于先腐蚀、再膨胀，用来删除小亮点。
    // 这里把四步分别写出，方便新生观察 erode 和 dilate 的执行顺序。
    const cv::Mat close_kernel =
        cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 5));
    cv::Mat image_progress_close_dilated;
    cv::dilate(image_progress_binary, image_progress_close_dilated, close_kernel);
    cv::Mat image_progress_closed;
    cv::erode(image_progress_close_dilated, image_progress_closed, close_kernel);

    const cv::Mat open_kernel =
        cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::Mat image_progress_open_eroded;
    cv::erode(image_progress_closed, image_progress_open_eroded, open_kernel);
    cv::Mat image_progress_morphology;
    cv::dilate(image_progress_open_eroded, image_progress_morphology, open_kernel);

    // 第五步：轮廓提取。每个白色连通区域给出一条外轮廓；
    // 使用掩膜副本，保留原掩膜用于阶段图。
    cv::Mat contour_input = image_progress_morphology.clone();
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(contour_input, contours, cv::RETR_EXTERNAL,
                     cv::CHAIN_APPROX_SIMPLE);


    // 第七步：灯条几何筛选。只留下足够长、足够细、接近竖直的区域，
    // 并按照画面中的横坐标排序，为下一步左右配对做准备。
    std::vector<LightBar> bars;
    for (const auto& contour : contours) {
        LightBar bar;
        if (makeLightBar(contour, bar)) {
            bars.push_back(bar);
        }
    }
    std::sort(bars.begin(), bars.end(), [](const LightBar& a, const LightBar& b) {
        return a.rectangle.center.x < b.rectangle.center.x;
    });

    // 第八步：配对灯条。黄色线段连接通过几何和内部亮度检查的左右灯条。
    // 先使用较严格的间距排除跨板误配。只有整帧一块都找不到时，
    // 才稍微放宽间距，恢复二值图中灯条变粗的少数帧。
    // 从原图提取亮像素；不用 CLAHE 增强图，避免把暗支架误判成白色数字。
    cv::Mat bright_pixels;
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, bright_pixels, 80, 255, cv::THRESH_BINARY);
    std::vector<Armor> armors = pairLightBars(bars, bright_pixels, 2.5F);
    if (armors.empty()) {
        armors = pairLightBars(bars, bright_pixels, 2.6F);
    }

    // 第九步：生成装甲板四角点。左右灯条长轴的上下端点给出
    // 近似四角；红点和绿色四边形是最终输出，不代表精确的三维角点。
    cv::Mat image_progress_armor_corners = frame.clone();
    if (armors.empty()) {
            cv::putText(image_progress_armor_corners, "Not Detected",
                cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX,
                1.0, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);}
    for (std::size_t i = 0; i < armors.size(); ++i) {
        const Armor& armor = armors[i];
        for (int k = 0; k < 4; ++k) {
            cv::line(image_progress_armor_corners, pixelPoint(armor.corners[k]),
                     pixelPoint(armor.corners[(k + 1) % 4]),
                     cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
            cv::circle(image_progress_armor_corners, pixelPoint(armor.corners[k]), 4,
                       cv::Scalar(0, 0, 255), cv::FILLED, cv::LINE_AA);
        }
        const cv::Point label = pixelPoint(armor.corners[0]) + cv::Point(0, -8);
        
        cv::putText(image_progress_armor_corners,
                    "Armor " + std::to_string(i + 1), label,
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2,
                    cv::LINE_AA);
    }
    result.annotated = image_progress_armor_corners;
    result.armor_count = armors.size();
    return result;}
    //删除了学长模板中的中间步展示绘制部分，保留并返回最终标识视频
    void openVideoWriter(cv::VideoWriter& writer, const fs::path& path,
                     double fps, const cv::Size& size) {
    // 直接用 OpenCV 写入 mp4v 编码的 MP4，帧率和尺寸由调用方传入。
    writer.open(path.string(), cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                fps, size, true);
    if (!writer.isOpened()) {
        throw std::runtime_error("无法创建视频: " + path.string());
    }
}



}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "用法: " << argv[0] << " <输入视频.mp4> <输出目录>\n";
        return 1;
    }

    try {
        const fs::path input_path(argv[1]);
        const fs::path output_directory(argv[2]);
        if (!fs::is_regular_file(input_path)) {
            throw std::runtime_error("输入视频不存在: " + input_path.string());
        }

        cv::VideoCapture capture(input_path.string());
        if (!capture.isOpened()) {
            throw std::runtime_error("无法读取输入视频: " + input_path.string());
        }
        fs::create_directories(output_directory);
        const std::string stem = input_path.stem().string();
        const fs::path video_path = output_directory / (stem + "_annotated.mp4");

        double fps = capture.get(cv::CAP_PROP_FPS);
        if (!std::isfinite(fps) || fps <= 0.0) {
            fps = 30.0;
        }
        cv::VideoWriter annotated_writer;
        cv::Mat frame;
        int frame_index = 0;
        std::size_t total_armors = 0;
        int detected_frames = 0;
        while (capture.read(frame)) {
            if (frame.empty()) {
                throw std::runtime_error("视频中出现空帧");
            }

            FrameResult result = processFrame(frame);
            if (!annotated_writer.isOpened()) {
                openVideoWriter(annotated_writer, video_path, fps, frame.size());
            }
            annotated_writer.write(result.annotated);
            total_armors += result.armor_count;
            if (result.armor_count > 0) {
                ++detected_frames;
            }
            ++frame_index;
        }
        if (frame_index == 0) {
            throw std::runtime_error("视频没有可读取的帧");
        }
        annotated_writer.release();
        capture.release();

        std::cout << input_path.filename().string() << ": " << frame_index
                  << " 帧，检测到 " << total_armors << " 组装甲板（逐帧计数）\n"
                  << "至少检出一组的帧: " << detected_frames << "/" << frame_index
                  << " (" << std::fixed << std::setprecision(2)
                  << 100.0 * detected_frames / frame_index << "%)\n"
                  << "标注视频: " << video_path << "\n";

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "错误: " << error.what() << '\n';
        return 1;
    }
}