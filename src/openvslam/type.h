#ifndef OPENVSLAM_TYPE_H
#define OPENVSLAM_TYPE_H

#include <vector>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <type_traits>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core/types.hpp>


// TODO, OLSLO, find better place for eiget2json converison.
#include <nlohmann/json.hpp>

struct type {
    template<typename T>
    using base_type = typename std::remove_cv<typename std::remove_reference<T>::type>::type;
    template<typename T>
    static constexpr const char *dtype() {
        if constexpr(std::is_same<base_type<T>, char>::value) { return "u1"; }
        if constexpr(std::is_same<base_type<T>, std::uint8_t>::value) { return "u1"; }
        else if constexpr(std::is_same<base_type<T>, std::uint16_t>::value) { return "u2"; }
        else if constexpr(std::is_same<base_type<T>, std::uint32_t>::value) { return "u4"; }
        else if constexpr(std::is_same<base_type<T>, std::uint64_t>::value) { return "u8"; }
        else if constexpr(std::is_same<base_type<T>, std::int8_t>::value) { return "i1"; }
        else if constexpr(std::is_same<base_type<T>, std::int16_t>::value) { return "i2"; }
        else if constexpr(std::is_same<base_type<T>, std::int32_t>::value) { return "i4"; }
        else if constexpr(std::is_same<base_type<T>, std::int64_t>::value) { return "i8"; }
        else if constexpr(std::is_same<base_type<T>, float>::value) { return "f4"; }
        else if constexpr(std::is_same<base_type<T>, double>::value) { return "f8"; }
        else if constexpr(std::is_same<base_type<T>, long double>::value) { return "f8"; }
        else { return "unknown type"; }
    }
};

namespace Eigen {
template<typename T>
void to_json(nlohmann::json &j, const T &matrix) {
    typename T::Index rows = matrix.rows(), cols = matrix.cols();
    auto data_begin = const_cast<typename T::Scalar *>(matrix.data());
    std::vector<typename T::Scalar> v(data_begin, data_begin + rows * cols);

    bool is_row_major = matrix.IsRowMajor;

    j = {
        {"dtype", type::dtype<typename T::Scalar>()},
        {"shape", {rows, cols}},
        {"data", v},
        {"order", is_row_major ? "C" : "F"}
    };
}
}


namespace openvslam {

// floating point type

typedef float real_t;

// Eigen matrix types

template<size_t R, size_t C>
using MatRC_t = Eigen::Matrix<double, R, C>;

using Mat22_t = Eigen::Matrix2d;

using Mat33_t = Eigen::Matrix3d;

using Mat44_t = Eigen::Matrix4d;

using Mat55_t = MatRC_t<5, 5>;

using Mat66_t = MatRC_t<6, 6>;

using Mat77_t = MatRC_t<7, 7>;

using Mat34_t = MatRC_t<3, 4>;

using MatX_t = Eigen::MatrixXd;

// Eigen vector types

template<size_t R>
using VecR_t = Eigen::Matrix<double, R, 1>;

using Vec2_t = Eigen::Vector2d;

using Vec3_t = Eigen::Vector3d;

using Vec4_t = Eigen::Vector4d;

using Vec5_t = VecR_t<5>;

using Vec6_t = VecR_t<6>;

using Vec7_t = VecR_t<7>;

using VecX_t = Eigen::VectorXd;

// Eigen Quaternion type

using Quat_t = Eigen::Quaterniond;

// STL with Eigen custom allocator

template<typename T>
using eigen_alloc_vector = std::vector<T, Eigen::aligned_allocator<T>>;

template<typename T, typename U>
using eigen_alloc_map = std::map<T, U, std::less<T>, Eigen::aligned_allocator<std::pair<const T, U>>>;

template<typename T>
using eigen_alloc_set = std::set<T, std::less<T>, Eigen::aligned_allocator<const T>>;

template<typename T, typename U>
using eigen_alloc_unord_map = std::unordered_map<T, U, std::hash<T>, std::equal_to<T>, Eigen::aligned_allocator<std::pair<const T, U>>>;

template<typename T>
using eigen_alloc_unord_set = std::unordered_set<T, std::hash<T>, std::equal_to<T>, Eigen::aligned_allocator<const T>>;

// vector operators

template<typename T>
inline Vec2_t operator+(const Vec2_t& v1, const cv::Point_<T>& v2) {
    return {v1(0) + v2.x, v1(1) + v2.y};
}

template<typename T>
inline Vec2_t operator+(const cv::Point_<T>& v1, const Vec2_t& v2) {
    return v2 + v1;
}

template<typename T>
inline Vec2_t operator-(const Vec2_t& v1, const cv::Point_<T>& v2) {
    return v1 + (-v2);
}

template<typename T>
inline Vec2_t operator-(const cv::Point_<T>& v1, const Vec2_t& v2) {
    return v1 + (-v2);
}

// tracker state
enum class tracker_state_t : unsigned int {
  NotInitialized = 0,
  Initializing = 1,
  Tracking = 2,
  Lost = 3
};

} // namespace openvslam

#endif // OPENVSLAM_TYPE_H
