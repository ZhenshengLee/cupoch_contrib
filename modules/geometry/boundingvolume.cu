#include "modules/geometry/boundingvolume.h"

namespace cupoch
{
namespace geometry
{
// namespace
// {

template <int Dim>
struct check_without_axis_aligned_bounding_box_functor
{
    check_without_axis_aligned_bounding_box_functor(const Eigen::Matrix<float, Dim, 1>* points,
                                                    const Eigen::Matrix<float, Dim, 1>& min_bound,
                                                    const Eigen::Matrix<float, Dim, 1>& max_bound)
      : points_(points), min_bound_(min_bound), max_bound_(max_bound){};
    const Eigen::Matrix<float, Dim, 1>* points_;
    const Eigen::Matrix<float, Dim, 1> min_bound_;
    const Eigen::Matrix<float, Dim, 1> max_bound_;
    __device__ bool operator()(size_t idx) const
    {
        const Eigen::Matrix<float, Dim, 1>& point = points_[idx];
#pragma unroll
        for (int i = 0; i < Dim; ++i)
        {
            if (point(i) < min_bound_(i) || point(i) > max_bound_(i))
            {
                return true;
            }
        }
        return false;
    }
};

// } // namespace

template <int Dim>
utility::device_vector<size_t> AxisAlignedBoundingBoxEx<Dim>::GetPointIndicesWithoutBoundingBox(
    const utility::device_vector<Eigen::Matrix<float, Dim, 1>>& points) const
{
    utility::device_vector<size_t> indices(points.size());
    check_without_axis_aligned_bounding_box_functor<Dim> func(thrust::raw_pointer_cast(points.data()), this->min_bound_,
                                                              this->max_bound_);
    auto end = thrust::copy_if(thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator(points.size()),
                               indices.begin(), func);
    indices.resize(thrust::distance(indices.begin(), end));
    return indices;
}

template class AxisAlignedBoundingBoxEx<2>;
template class AxisAlignedBoundingBoxEx<3>;

}  // namespace geometry
}  // namespace cupoch
