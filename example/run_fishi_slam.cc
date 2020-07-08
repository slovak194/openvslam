#ifdef USE_PANGOLIN_VIEWER
#include "pangolin_viewer/viewer.h"
#elif USE_SOCKET_PUBLISHER
#include "socket_publisher/publisher.h"
#endif

#include "openvslam/system.h"
#include "openvslam/config.h"
#include "openvslam/util/stereo_rectifier.h"

#include <iostream>
#include <chrono>
#include <numeric>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <spdlog/spdlog.h>
#include <popl.hpp>

#ifdef USE_STACK_TRACE_LOGGER
#include <glog/logging.h>
#endif

#ifdef USE_GOOGLE_PERFTOOLS
#include <gperftools/profiler.h>
#endif

void mono_tracking(const std::shared_ptr<openvslam::config>& cfg,
                   const std::string& vocab_file_path, const std::string& video_file_path, const std::string& mask_img_path,
                   const unsigned int frame_skip, const bool no_sleep, const bool auto_term,
                   const bool eval_log, const std::string& map_db_path) {
  // load the mask image
  const cv::Mat mask = mask_img_path.empty() ? cv::Mat{} : cv::imread(mask_img_path, cv::IMREAD_GRAYSCALE);

  // build a SLAM system
  openvslam::system SLAM(cfg, vocab_file_path);
  // startup the SLAM process
  SLAM.startup();

  // create a viewer object
  // and pass the frame_publisher and the map_publisher
#ifdef USE_PANGOLIN_VIEWER
  pangolin_viewer::viewer viewer(cfg, &SLAM, SLAM.get_frame_publisher(), SLAM.get_map_publisher());
#elif USE_SOCKET_PUBLISHER
  socket_publisher::publisher publisher(cfg, &SLAM, SLAM.get_frame_publisher(), SLAM.get_map_publisher());
#endif

  auto video = cv::VideoCapture(video_file_path, cv::CAP_FFMPEG);
  std::vector<double> track_times;

  cv::Mat frame;
  double timestamp = 0.0;

  unsigned int num_frame = 0;

  bool is_not_end = true;
  // run the SLAM in another thread
  std::thread thread([&]() {
    while (is_not_end) {
      is_not_end = video.read(frame);

      const auto tp_1 = std::chrono::steady_clock::now();

      if (!frame.empty() && (num_frame % frame_skip == 0)) {
        // input the current frame and estimate the camera pose
        SLAM.feed_monocular_frame(frame, timestamp, mask);
      }

      const auto tp_2 = std::chrono::steady_clock::now();

      const auto track_time = std::chrono::duration_cast<std::chrono::duration<double>>(tp_2 - tp_1).count();
      if (num_frame % frame_skip == 0) {
        track_times.push_back(track_time);
      }

      // wait until the timestamp of the next frame
      if (!no_sleep) {
        const auto wait_time = 1.0 / cfg->camera_->fps_ - track_time;
        if (0.0 < wait_time) {
          std::this_thread::sleep_for(std::chrono::microseconds(static_cast<unsigned int>(wait_time * 1e6)));
        }
      }

      timestamp += 1.0 / cfg->camera_->fps_;
      ++num_frame;

      // check if the termination of SLAM system is requested or not
      if (SLAM.terminate_is_requested()) {
        break;
      }
    }

    // wait until the loop BA is finished
    while (SLAM.loop_BA_is_running()) {
      std::this_thread::sleep_for(std::chrono::microseconds(5000));
    }

    // automatically close the viewer
#ifdef USE_PANGOLIN_VIEWER
    if (auto_term) {
      viewer.request_terminate();
    }
#elif USE_SOCKET_PUBLISHER
    if (auto_term) {
            publisher.request_terminate();
        }
#endif
  });

  // run the viewer in the current thread
#ifdef USE_PANGOLIN_VIEWER
  viewer.run();
#elif USE_SOCKET_PUBLISHER
  publisher.run();
#endif

  thread.join();

  // shutdown the SLAM process
  SLAM.shutdown();

  if (eval_log) {
    // output the trajectories for evaluation
    SLAM.save_frame_trajectory("frame_trajectory.txt", "TUM");
    SLAM.save_keyframe_trajectory("keyframe_trajectory.txt", "TUM");
    // output the tracking times for evaluation
    std::ofstream ofs("track_times.txt", std::ios::out);
    if (ofs.is_open()) {
      for (const auto track_time : track_times) {
        ofs << track_time << std::endl;
      }
      ofs.close();
    }
  }

  if (!map_db_path.empty()) {
    // output the map database
    SLAM.save_map_database(map_db_path);
  }

  std::sort(track_times.begin(), track_times.end());
  const auto total_track_time = std::accumulate(track_times.begin(), track_times.end(), 0.0);
  std::cout << "median tracking time: " << track_times.at(track_times.size() / 2) << "[s]" << std::endl;
  std::cout << "mean tracking time: " << total_track_time / track_times.size() << "[s]" << std::endl;
}


void stereo_tracking(const std::shared_ptr<openvslam::config>& cfg,
                   const std::string& vocab_file_path, const std::string& video_file_path_1, const std::string& video_file_path_2, const std::string& mask_img_path,
                   const unsigned int frame_skip, const bool no_sleep, const bool auto_term,
                   const bool eval_log, const std::string& map_db_path) {
  // load the mask image
  const cv::Mat mask = mask_img_path.empty() ? cv::Mat{} : cv::imread(mask_img_path, cv::IMREAD_GRAYSCALE);

  // build a SLAM system
  openvslam::system SLAM(cfg, vocab_file_path);
  // startup the SLAM process
  SLAM.startup();

  // create a viewer object
  // and pass the frame_publisher and the map_publisher
#ifdef USE_PANGOLIN_VIEWER
  pangolin_viewer::viewer viewer(cfg, &SLAM, SLAM.get_frame_publisher(), SLAM.get_map_publisher());
#elif USE_SOCKET_PUBLISHER
  socket_publisher::publisher publisher(cfg, &SLAM, SLAM.get_frame_publisher(), SLAM.get_map_publisher());
#endif

  auto video_1 = cv::VideoCapture(video_file_path_1, cv::CAP_FFMPEG);
  auto video_2 = cv::VideoCapture(video_file_path_2, cv::CAP_FFMPEG);
  std::vector<double> track_times;



  cv::Mat frames[2];
  cv::Mat frames_rectified[2];

  double timestamp = 0.0;

  const openvslam::util::stereo_rectifier rectifier(cfg);

  unsigned int num_frame = 0;

  bool is_not_end = true;
  // run the SLAM in another thread
  std::thread thread([&]() {
    while (is_not_end) {
      is_not_end = video_1.read(frames[0]);
      is_not_end = video_2.read(frames[1]);

      const auto tp_1 = std::chrono::steady_clock::now();

      if (!frames[0].empty() && (num_frame % frame_skip == 0)) {
        // input the current frame and estimate the camera pose
//        SLAM.feed_monocular_frame(frame, timestamp, mask);

        rectifier.rectify(frames[0], frames[1], frames_rectified[0], frames_rectified[1]);

        auto now = std::chrono::system_clock::now();
        auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(now);
        auto epoch = now_ms.time_since_epoch();
        auto value = std::chrono::duration_cast<std::chrono::milliseconds>(epoch);
        double dnow = static_cast<double>(value.count());

        SLAM.feed_stereo_frame(frames_rectified[0], frames_rectified[1], dnow);
      }

      const auto tp_2 = std::chrono::steady_clock::now();

      const auto track_time = std::chrono::duration_cast<std::chrono::duration<double>>(tp_2 - tp_1).count();
      if (num_frame % frame_skip == 0) {
        track_times.push_back(track_time);
      }

      // wait until the timestamp of the next frame
      if (!no_sleep) {
        const auto wait_time = 1.0 / cfg->camera_->fps_ - track_time;
        if (0.0 < wait_time) {
          std::this_thread::sleep_for(std::chrono::microseconds(static_cast<unsigned int>(wait_time * 1e6)));
        }
      }

      timestamp += 1.0 / cfg->camera_->fps_;
      ++num_frame;

      // check if the termination of SLAM system is requested or not
      if (SLAM.terminate_is_requested()) {
        break;
      }
    }

    // wait until the loop BA is finished
    while (SLAM.loop_BA_is_running()) {
      std::this_thread::sleep_for(std::chrono::microseconds(5000));
    }

    // automatically close the viewer
#ifdef USE_PANGOLIN_VIEWER
    if (auto_term) {
      viewer.request_terminate();
    }
#elif USE_SOCKET_PUBLISHER
    if (auto_term) {
            publisher.request_terminate();
        }
#endif
  });

  // run the viewer in the current thread
#ifdef USE_PANGOLIN_VIEWER
  viewer.run();
#elif USE_SOCKET_PUBLISHER
  publisher.run();
#endif

  thread.join();

  // shutdown the SLAM process
  SLAM.shutdown();

  if (eval_log) {
    // output the trajectories for evaluation
    SLAM.save_frame_trajectory("frame_trajectory.txt", "TUM");
    SLAM.save_keyframe_trajectory("keyframe_trajectory.txt", "TUM");
    // output the tracking times for evaluation
    std::ofstream ofs("track_times.txt", std::ios::out);
    if (ofs.is_open()) {
      for (const auto track_time : track_times) {
        ofs << track_time << std::endl;
      }
      ofs.close();
    }
  }

  if (!map_db_path.empty()) {
    // output the map database
    SLAM.save_map_database(map_db_path);
  }

  std::sort(track_times.begin(), track_times.end());
  const auto total_track_time = std::accumulate(track_times.begin(), track_times.end(), 0.0);
  std::cout << "median tracking time: " << track_times.at(track_times.size() / 2) << "[s]" << std::endl;
  std::cout << "mean tracking time: " << total_track_time / track_times.size() << "[s]" << std::endl;
}



int main(int argc, char* argv[]) {
#ifdef USE_STACK_TRACE_LOGGER
  google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();
#endif

  // create options
  popl::OptionParser op("Allowed options");
  auto help = op.add<popl::Switch>("h", "help", "produce help message");
  auto vocab_file_path = op.add<popl::Value<std::string>>("v", "vocab", "vocabulary file path");
  auto video_file_path_1 = op.add<popl::Value<std::string>>("L", "video1", "video file path 1");
  auto video_file_path_2 = op.add<popl::Value<std::string>>("R", "video2", "video file path 2");
  auto config_file_path = op.add<popl::Value<std::string>>("c", "config", "config file path");
  auto mask_img_path = op.add<popl::Value<std::string>>("", "mask", "mask image path", "");
  auto frame_skip = op.add<popl::Value<unsigned int>>("", "frame-skip", "interval of frame skip", 1);
  auto no_sleep = op.add<popl::Switch>("", "no-sleep", "not wait for next frame in real time");
  auto auto_term = op.add<popl::Switch>("", "auto-term", "automatically terminate the viewer");
  auto debug_mode = op.add<popl::Switch>("", "debug", "debug mode");
  auto eval_log = op.add<popl::Switch>("", "eval-log", "store trajectory and tracking times for evaluation");
  auto map_db_path = op.add<popl::Value<std::string>>("p", "map-db", "store a map database at this path after SLAM", "");
  try {
    op.parse(argc, argv);
  }
  catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    std::cerr << std::endl;
    std::cerr << op << std::endl;
    return EXIT_FAILURE;
  }

  // check validness of options
  if (help->is_set()) {
    std::cerr << op << std::endl;
    return EXIT_FAILURE;
  }
  if (!vocab_file_path->is_set() || !video_file_path_1->is_set() || !video_file_path_2->is_set() || !config_file_path->is_set()) {
    std::cerr << "invalid arguments" << std::endl;
    std::cerr << std::endl;
    std::cerr << op << std::endl;
    return EXIT_FAILURE;
  }

  // setup logger
  spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] %^[%L] %v%$");
  if (debug_mode->is_set()) {
    spdlog::set_level(spdlog::level::debug);
  }
  else {
    spdlog::set_level(spdlog::level::info);
  }

  // load configuration
  std::shared_ptr<openvslam::config> cfg;
  try {
    cfg = std::make_shared<openvslam::config>(config_file_path->value());
  }
  catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

#ifdef USE_GOOGLE_PERFTOOLS
  ProfilerStart("slam.prof");
#endif


  if (cfg->camera_->setup_type_ == openvslam::camera::setup_type_t::Monocular) {
    mono_tracking(cfg, vocab_file_path->value(), video_file_path_1->value(), mask_img_path->value(),
                  frame_skip->value(), no_sleep->is_set(), auto_term->is_set(),
                  eval_log->is_set(), map_db_path->value());
  }
  else if (cfg->camera_->setup_type_ == openvslam::camera::setup_type_t::Stereo) {
    stereo_tracking(cfg, vocab_file_path->value(), video_file_path_1->value(), video_file_path_2->value(), mask_img_path->value(),
                    frame_skip->value(), no_sleep->is_set(), auto_term->is_set(),
                    eval_log->is_set(), map_db_path->value());
  }
  else {
    throw std::runtime_error("Invalid setup type: " + cfg->camera_->get_setup_type_string());
  }

#ifdef USE_GOOGLE_PERFTOOLS
  ProfilerStop();
#endif

  return EXIT_SUCCESS;
}