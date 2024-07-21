/*
multi thread runtime testj
thread 1: read video stream
thread 2: inference
thread 3: write video
thread 4: sreamer
*/
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "logger.h"
#include "common.h"
#include "buffers.h"
#include "utils/preprocess.h"
#include "utils/postprocess.h"
#include "utils/types.h"
#include "streamer/streamer.hpp"
#include "lib/httplib.h"
#include "lib/json.hpp"

#include <fstream>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <ctime>
// 用于测试
#define PRINT_STEP_TIME 1
#define PRINT_ALL_TIME 1

using json = nlohmann::json;
static cv::VideoWriter writer;                // 保存视频对象

// 定义数据结构
struct bufferItem
{
    cv::Mat frame;                // 原始图像
    std::vector<Detection> bboxs; // 检测结果
};
// 缓存大小
const int BUFFER_SIZE = 10;

// 每个阶段需要传递的缓存
std::queue<cv::Mat> stage_1_frame;
std::queue<bufferItem> stage_2_buffer;
std::queue<cv::Mat> stage_3_frame;

// 每个阶段的互斥锁
std::mutex stage_1_mutex;
std::mutex stage_2_mutex;
std::mutex stage_3_mutex;

// 每个阶段的not_full条件变量
std::condition_variable stage_1_not_full;
std::condition_variable stage_2_not_full;
std::condition_variable stage_3_not_full;

// 每个阶段的not_empty条件变量
std::condition_variable stage_1_not_empty;
std::condition_variable stage_2_not_empty;
std::condition_variable stage_3_not_empty;

class BirdApp
{
public:
    // destructor
    ~BirdApp()
    {
        std::cout << "BirdApp destructor" << std::endl;
    }
    // constructor
    BirdApp(const std::string &engine_file, const std::string &input_video_path, int do_stream, int bitrate)
        : do_stream{do_stream}, bitrate{bitrate}
    {
        // ========= 1. 创建推理运行时runtime =========
        auto runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));
        // ======== 2. 反序列化生成engine =========
        // 加载模型文件 
        auto plan = load_engine_file(engine_file);
        // 反序列化生成engine
        mEngine_ = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan.data(), plan.size()));
        // ======== 3. 创建执行上下文context =========
        context_ = std::unique_ptr<nvinfer1::IExecutionContext>(mEngine_->createExecutionContext());
        if (!context_) {
                std::cerr << "Failed to create execution context" << std::endl;
                return;
        }

        // 如果input_video_path是rtsp，则读取rtsp流
        if (input_video_path == "rtsp")
        {
            auto rtsp = "rtsp://192.168.1.241:8556/live1.sdp";
            std::cout << "当前使用的是RTSP流" << std::endl;
            cap = cv::VideoCapture(rtsp, cv::CAP_FFMPEG);
        }
        else
        {
            std::cout << "当前使用的是视频文件" << std::endl;
            cap = cv::VideoCapture(input_video_path);
        }
        // 获取画面尺寸
        frameSize_ = cv::Size(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        // 获取帧率
        video_fps_ = cap.get(cv::CAP_PROP_FPS);
        std::cout << "width: " << frameSize_.width << " height: " << frameSize_.height << " fps: " << video_fps_ << std::endl;
    };

    // 发送接口请求
    viod requestReportAlert()
    {
        
        std::string date = "2024-07-19";
        std::string _type = "fire";
        std::string code = "C1234";
        std::string video_url = "http://example.com/video.mp4";

        // 创建 JSON 对象
        json data = {
            {"alertDate", date},
            {"alertType", _type},
            {"cameraCode", code},
            {"videoUrl", video_url}
        };

        // 转换为字符串
        std::string post_body = data.dump();

        // 创建httplib::Client对象
        httplib::Client cli("http://192.168.0.103:8080");

        // 发送GET请求
        auto res = cli.Get("/user/get");
        if (res && res->status == 200) {
            std::cout << "GET Response: " << res->body << std::endl;
        } else {
            std::cerr << "GET Request Failed!" << std::endl;
            if (res) {
                std::cerr << "Status code: " << res->status << std::endl;
            }
        }

        // 发送POST请求
        auto post_res = cli.Post("/user/post", post_body, "application/json");
        if (post_res && post_res->status == 200) {
            std::cout << "POST Response: " << post_res->body << std::endl;
        } else {
            std::cerr << "POST Request Failed!" << std::endl;
            if (post_res) {
                std::cerr << "Status code: " << post_res->status << std::endl;
            }
        }
    }
    

    // 线程1： read frame
    void readFrame()
    {
        std::cout << "取帧线程启动" << std::endl;

        // product frame for inference
        cv::Mat frame;
        while (cap.isOpened())
        {
            // step1 start
            auto start_1 = std::chrono::high_resolution_clock::now();
            cap >> frame;
            if (frame.empty())
            {
                std::cout << "文件处理完毕" << std::endl;
                file_processed_done = true;
                break;
            }
            // step1 end
            auto end_1 = std::chrono::high_resolution_clock::now();
            auto elapsed_1 = std::chrono::duration_cast<std::chrono::microseconds>(end_1 - start_1).count() / 1000.f;
#if PRINT_STEP_TIME
            std::cout << "step1: " << elapsed_1 << "ms"
                      << ", fps: " << 1000.f / elapsed_1 << std::endl;
#endif
            // 互斥锁
            std::unique_lock<std::mutex> lock(stage_1_mutex);
            // 如果缓存满了，就等待
            stage_1_not_full.wait(lock, []
                                  { return stage_1_frame.size() < BUFFER_SIZE; });
            // 增加一个元素
            stage_1_frame.push(frame);
            // 通知下一个线程可以开始了
            stage_1_not_empty.notify_one();
        }
    }
    // 线程2： inference
    void inference()
    {
        std::cout << "推理线程启动" << std::endl;
        // ========== 4. 创建输入输出缓冲区 =========
        samplesCommon::BufferManager buffers(mEngine_);

        cv::Mat frame;

        int img_size = frameSize_.width * frameSize_.height;
        cuda_preprocess_init(img_size); // 申请cuda内存
        while (true)
        {

            // 检查是否退出
            if (file_processed_done && stage_1_frame.empty())
            {
                std::cout << "推理线程退出" << std::endl;
                break;
            }
            // 使用{} 限制作用域，否则锁会在一次循环结束后才释放
            {
                // 使用互斥锁
                std::unique_lock<std::mutex> lock(stage_1_mutex);
                // 如果缓存为空，就等待
                stage_1_not_empty.wait(lock, []
                                       { return !stage_1_frame.empty(); });
                // 取出一个元素
                frame = stage_1_frame.front();
                stage_1_frame.pop();
                // 通知上一个线程可以开始了
                stage_1_not_full.notify_one();
            }

            // step2 start
            auto start_2 = std::chrono::high_resolution_clock::now();

            // 使用cuda预处理所有步骤letterbox，归一化、BGR2RGB、NHWC to NCHW
            process_input_gpu(frame, (float *)buffers.getDeviceBuffer(kInputTensorName));

            // ========== 5. 执行推理 =========
            context_->executeV2(buffers.getDeviceBindings().data());
            // 拷贝回host
            buffers.copyOutputToHost();

            // 从buffer manager中获取模型输出
            int32_t *num_det = (int32_t *)buffers.getHostBuffer(kOutNumDet); // 检测到的目标个数
            int32_t *cls = (int32_t *)buffers.getHostBuffer(kOutDetCls);     // 检测到的目标类别
            float *conf = (float *)buffers.getHostBuffer(kOutDetScores);     // 检测到的目标置信度
            float *bbox = (float *)buffers.getHostBuffer(kOutDetBBoxes);     // 检测到的目标框
            // 执行nms（非极大值抑制），得到最后的检测框
            std::vector<Detection> bboxs;
            yolo_nms(bboxs, num_det, cls, conf, bbox, kConfThresh, kNmsThresh);

            bufferItem item;
            // copy frmae to item
            item.frame = frame.clone();
            // item.frame = frame;
            item.bboxs = bboxs;

            // step2 end
            auto end_2 = std::chrono::high_resolution_clock::now();
            elapsed_2 = std::chrono::duration_cast<std::chrono::microseconds>(end_2 - start_2).count() / 1000.f;

#if PRINT_STEP_TIME
            std::cout << "step2: " << elapsed_2 << "ms"
                      << ", fps: " << 1000.f / elapsed_2 << std::endl;
#endif
            {
                // 使用互斥锁
                std::unique_lock<std::mutex> lock2(stage_2_mutex);
                // not full
                stage_2_not_full.wait(lock2, []
                                      { return stage_2_buffer.size() < BUFFER_SIZE; });
                // push
                stage_2_buffer.push(item);
                poped = false; // 缓冲区加入新元素的通知
                // not empty
                stage_2_not_empty.notify_one();
            }
        }
    }
    // 线程3： write video and process
    void writeAndProcessVideo()
    {

        std::cout << "保存视频线程启动" << std::endl;

        bufferItem item;
        cv::Mat frame;
        while (true)
        {
            // 检查是否退出
            if (file_processed_done && stage_2_buffer.empty())
            {
                std::cout << "保存视频线程退出" << std::endl;
                break;
            }

            {
                // 使用互斥锁
                std::unique_lock<std::mutex> lock(stage_2_mutex);
                // 如果缓存为空，就等待
                stage_2_not_empty.wait(lock, []
                                       { return !stage_2_buffer.empty(); });
                // 取出一个元素
                item = stage_2_buffer.front();
                frame = item.frame.clone();
                thread_3 = true;
                // 如果线程4也取过了
                if(thread_4)
                {
                    stage_2_buffer.pop();
                    // 通知上一个线程可以开始了
                    stage_2_not_full.notify_one();
                    if(do_stream == 0){
                        thread_4 = true;
                    }
                    else
                    {
                        thread_4 = false;
                    }
                }
            }

            // step3 start
            auto start_3 = std::chrono::high_resolution_clock::now();

            // 遍历检测结果
            for (size_t j = 0; j < item.bboxs.size(); j++)
            {
                cv::Rect r = get_rect(frame, item.bboxs[j].bbox);
                cv::rectangle(frame, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
                // 绘制labelid
                cv::putText(frame, std::to_string((int)item.bboxs[j].class_id), cv::Point(r.x, r.y - 10), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0x27, 0xC1, 0x36), 2);
            }

            // step3 end
            auto end_3 = std::chrono::high_resolution_clock::now();
            auto elapsed_3 = std::chrono::duration_cast<std::chrono::microseconds>(end_3 - start_3).count() / 1000.f;

#if PRINT_STEP_TIME
            std::cout << "step3 time: " << elapsed_3 << "ms"
                      << ", fps: " << 1000.f / elapsed_3 << std::endl;
#endif

            // 绘制时间和帧率
            std::string time_str = "time: " + std::to_string(elapsed_2);
            std::string fps_str = "fps: " + std::to_string(1000.f / elapsed_2);
            cv::putText(frame, time_str, cv::Point(50, 50), cv::FONT_HERSHEY_PLAIN, 3, cv::Scalar(255, 255, 255), 2);
            cv::putText(frame, fps_str, cv::Point(50, 100), cv::FONT_HERSHEY_PLAIN, 3, cv::Scalar(255, 255, 255), 2);


            // 有目标
            if (item.bboxs.size() > 0) {
			
                if (!flag) {
                    flag = true;

                    time_t nowtime;
                    time(&nowtime); //获取1970年1月1日0点0分0秒到现在经过的秒数
                    tm *p;
                    p = localtime(&nowtime); //将秒数./modules/core/src/count_non_zero.dispatch.cpp:12转换为本地时间,年从1900算起,需要+1900,月为0-11,所以要+1

                    std::string video_name = std::to_string(p->tm_year + 1900) + ":" + std::to_string(p->tm_mon + 1) + ":" +
                                    std::to_string(p->tm_mday) + " " + std::to_string(p->tm_hour) + ":" +
                                    std::to_string(p->tm_min) + ":" + std::to_string(p->tm_sec);
                    std::string video_path = "./output/" + video_name + ".mp4";

             
		    writer.open(video_path, cv::VideoWriter::fourcc('A', 'V', 'C', '1'), video_fps_, frameSize_);
                }
		cv::Mat temp;
                // 写入视频文件
		writer.write(frame);
                c_frame_count++;

		std::cout << c_frame_count <<std::endl;
                if (c_frame_count >= int(video_fps_)*10) {
                    // 完成写入后释放资源
		    writer.release();
		    c_frame_count = 0;

                    flag = false;
                }
            }
            // 没有目标
            else {
                if (flag) {
                    if (c_frame_count >=int(video_fps_)*10) {
                        // 完成写入后释放资源
			writer.release();
                        c_frame_count = 0;

                        flag = false;
                    }
                    else {
                        // 写入视频文件
			writer.write(frame);
                        c_frame_count++;
                    }
                }
            }


        }
    }
    // 线程4： streamer
    void streamer()
    {
        std::cout << "推流线程启动" << std::endl;
        // 实例化推流器
        streamer::Streamer streamer;
        streamer::StreamerConfig streamer_config(frameSize_.width, frameSize_.height,
                                                 frameSize_.width, frameSize_.height,
                                                 video_fps_, bitrate, "main", "rtmp://192.168.122.6:1935/c++/live");
        streamer.init(streamer_config);

        // 记录开始时间
        auto start_all = std::chrono::high_resolution_clock::now();
        int frame_count = 0;
        cv::Mat frame;
        while (true)
        {
            // 检查是否退出
            if (file_processed_done && stage_2_buffer.empty())
            {
                std::cout << "推流线程退出" << std::endl;
                break;
            }
            {
                // 使用互斥锁
                std::unique_lock<std::mutex> lock(stage_2_mutex);
                // 如果缓存为空，就等待
                stage_2_not_empty.wait(lock, []
                                       { return !stage_2_buffer.empty(); });
                // 取出一个元素
                frame = stage_2_buffer.front().frame.clone();
                thread_4 = true;
                // 如果线程3也取过了
                if(thread_3)
                {
                    stage_2_buffer.pop();
                    // 通知上一个线程可以开始了
                    stage_2_not_full.notify_one();
                    thread_3 = false;
                }
            }

            // step4 start
            auto start_4 = std::chrono::high_resolution_clock::now();
            // 推流
            streamer.stream_frame(frame.data);
            // step4 end
            auto end_4 = std::chrono::high_resolution_clock::now();
            auto elapsed_4 = std::chrono::duration_cast<std::chrono::microseconds>(end_4 - start_4).count() / 1000.f;

#if PRINT_STEP_TIME
            std::cout << "step4 time: " << elapsed_4 << "ms"
                      << ", fps: " << 1000.f / elapsed_4 << std::endl;
#endif

#if PRINT_ALL_TIME
            // 算法2：计算超过 1s 一共处理了多少张图片
            frame_count++;
            // all end
            auto end_all = std::chrono::high_resolution_clock::now();
            auto elapsed_all_2 = std::chrono::duration_cast<std::chrono::microseconds>(end_all - start_all).count() / 1000.f;
            // 每隔1秒打印一次
            if (elapsed_all_2 > 1000)
            {
                std::cout << "method 2 all steps time(ms): " << elapsed_all_2 << ", fps: " << frame_count / (elapsed_all_2 / 1000.0f) << ",frame count: " << frame_count << std::endl;
                frame_count = 0;
                start_all = std::chrono::high_resolution_clock::now();
            }
#endif
        }
    }

private:
    std::string input_video_path = "rtsp"; // 输入源，文件或者rtsp流
    int do_stream = 0;                     // 是否推流
    int bitrate = 4000000;                 // 推流码率
    cv::VideoCapture cap;                  // 视频流
    float video_fps_;                      // 视频帧率
    cv::Size frameSize_;                   // 视频帧大小
    
    int c_frame_count = 0;                   // 当前帧数
    bool flag = false;                      // 是否保存视频
    bool poped = false;
    bool thread_3 = false;
    bool thread_4 = false;


    // std::unique_ptr<nvinfer1::IRuntime> runtime;           // 运行时
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine_;       // 模型引擎
    std::shared_ptr<nvinfer1::IExecutionContext> context_; // 执行上下文

    bool file_processed_done = false; // 文件处理完成标志
    float elapsed_2 = 0;              // inference time

    // 加载模型文件
    std::vector<unsigned char> load_engine_file(const std::string &file_name)
    {
        std::vector<unsigned char> engine_data;
        std::ifstream engine_file(file_name, std::ios::binary);
        assert(engine_file.is_open() && "Unable to load engine file.");
        engine_file.seekg(0, engine_file.end);
        int length = engine_file.tellg();
        engine_data.resize(length);
        engine_file.seekg(0, engine_file.beg);
        engine_file.read(reinterpret_cast<char *>(engine_data.data()), length);
        return engine_data;
    }
};

int main(int argc, char **argv)
{
    if (argc < 5)
    {
        std::cerr << "用法: " << argv[0] << " <模型文件> <视频流路径> <是否推流> <码率> " << std::endl;
        return -1;
    }

    auto engine_file = argv[1];                // 模型文件
    std::string input_video_path = argv[2];    // 输入视频文件
    auto do_stream = std::stoi(argv[3]);       // 是否推流
    auto bitrate = std::stoi(argv[4]);         // 码率
    // initialize class
    auto app = BirdApp(engine_file, input_video_path, do_stream, bitrate);

    // thread 1 : read video stream
    std::thread T_readFrame(&BirdApp::readFrame, &app);
    // thread 2: inference
    std::thread T_inference(&BirdApp::inference, &app);
    // thread 3: writeAndProcessVideo
    std::thread T_process(&BirdApp::writeAndProcessVideo, &app);
    // thread 4: streamer
    std::thread T_streamer(&BirdApp::streamer, &app);

    // 等待线程结束
    T_readFrame.join();
    T_inference.join();
    T_process.join();
    T_streamer.join();

    return 0;
}
