# README.md（包含从配置、编译到运行的完整命令）

# 环境配置：

本次作业在Linux环境中测试并通过：
Ubuntu 24.04 （在wsl ubuntu24.04上已验证）
GCC 13.3.0  （在wsl ubuntu24.04上已验证）
CMake >= 3.22  （在wsl ubuntu24.04上已验证，为3.28.3）
OpenCV 4.6.0  （在wsl ubuntu24.04上已验证）
C++17


# 依赖安装方法:
在全新环境中，请先执行以下命令安装必要依赖：
bash:
sudo apt update
sudo apt install build-essential cmake libopencv-dev

# 编译步骤：

创建并进入构建目录：
mkdir -p build
cd build

生成 Makefile 配置：
cmake ..

编译生成可执行文件：
cmake --build . -j

编译成功后，会在 build/ 目录下生成可执行文件 armor_detect

# 运行：

在当前目录为build下：
./armor_detect ../../../data/raw/marker_video.avi ../output/


第1个参数：输入视频路径（此处指向仓库中的 data/raw/marker_video.avi）
第2个参数：输出目录路径（此处指向 src/tangsunhei_50033100/output/）
运行成功后，终端会打印检测统计信息（如检测帧数百分比），并在 output/ 目录下生成 marker_video_annotated.mp4 视频文件