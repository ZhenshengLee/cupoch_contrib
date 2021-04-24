#pragma once

#include "cupoch/cupoch.h"

namespace cupoch
{
namespace geometry
{
// 扩充功能的boundingbox 类, 增加
template <int Dim>
class AxisAlignedBoundingBoxEx : public AxisAlignedBoundingBox<Dim>
{
  public:
    __host__ __device__ AxisAlignedBoundingBoxEx() : AxisAlignedBoundingBox<Dim>()
    {
    }
    __host__ __device__ AxisAlignedBoundingBoxEx(const Eigen::Matrix<float, Dim, 1>& min_bound,
                                                 const Eigen::Matrix<float, Dim, 1>& max_bound)
      : AxisAlignedBoundingBox<Dim>(min_bound, max_bound)
    {
    }
    __host__ __device__ ~AxisAlignedBoundingBoxEx()
    {
    }

  public:
    /// Return indices to points that are without the bounding box.
    ///
    /// \param points A list of points.
    utility::device_vector<size_t>
    GetPointIndicesWithoutBoundingBox(const utility::device_vector<Eigen::Matrix<float, Dim, 1>>& points) const;
};

}  // namespace geometry
}  // namespace cupoch