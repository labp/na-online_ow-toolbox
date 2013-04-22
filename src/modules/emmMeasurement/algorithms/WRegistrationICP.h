/**
 * TODO license and documentation
 */

#ifndef WREGISTRATIONICP_H_
#define WREGISTRATIONICP_H_

#include <string>

#include <boost/shared_ptr.hpp>
#include <Eigen/Dense>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>

#include "WRegistration.h"

class WRegistrationICP: public WRegistration
{
public:
    /**
     * Abbreviation for a shared pointer.
     */
    typedef boost::shared_ptr< WRegistrationICP > SPtr;

    /**
     * Abbreviation for const shared pointer.
     */
    typedef boost::shared_ptr< const WRegistrationICP > ConstSPtr;

    typedef pcl::PointXYZ PointT;

    static const std::string CLASS;

    WRegistrationICP();
    virtual ~WRegistrationICP();

    virtual WRegistration::MatrixTransformation getTransformationMatrix() const;

    virtual double compute( const WRegistration::PointCloud& from, const WRegistration::PointCloud& to );

    double compute( const WRegistration::PointCloud& from, const WRegistration::PointCloud& to,
                    WRegistration::MatrixTransformation initial );

private:
    bool computeICP( const pcl::PointCloud< PointT >::Ptr from, const pcl::PointCloud< PointT >::Ptr to,
                    Eigen::Matrix4f initial );

    pcl::IterativeClosestPoint< PointT, PointT >* m_icp;
};

#endif /* WREGISTRATIONICP_H_ */
