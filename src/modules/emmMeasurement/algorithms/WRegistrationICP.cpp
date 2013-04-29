/**
 * TODO license and documentation
 */

#include <string>
#include <vector>

#include <Eigen/Dense>

#include <core/common/WLogger.h>

#include "core/util/WLGeometry.h"
#include "WRegistration.h"
#include "WRegistrationICP.h"

const std::string WRegistrationICP::CLASS = "WRegistrationICP";

WRegistrationICP::WRegistrationICP()
{
    m_icp = new pcl::IterativeClosestPoint< PointT, PointT >();
}

WRegistrationICP::~WRegistrationICP()
{
    delete m_icp;
}

WRegistration::MatrixTransformation WRegistrationICP::getTransformationMatrix() const
{
    WRegistration::MatrixTransformation transformation;

    Eigen::Matrix4f icpTransformation = m_icp->getFinalTransformation();
    transformation( 0, 0 ) = icpTransformation( 0, 0 );
    transformation( 0, 1 ) = icpTransformation( 0, 1 );
    transformation( 0, 2 ) = icpTransformation( 0, 2 );
    transformation( 0, 3 ) = icpTransformation( 0, 3 );

    transformation( 1, 0 ) = icpTransformation( 1, 0 );
    transformation( 1, 1 ) = icpTransformation( 1, 1 );
    transformation( 1, 2 ) = icpTransformation( 1, 2 );
    transformation( 1, 3 ) = icpTransformation( 1, 3 );

    transformation( 2, 0 ) = icpTransformation( 2, 0 );
    transformation( 2, 1 ) = icpTransformation( 2, 1 );
    transformation( 2, 2 ) = icpTransformation( 2, 2 );
    transformation( 2, 3 ) = icpTransformation( 2, 3 );

    transformation( 3, 0 ) = icpTransformation( 3, 0 );
    transformation( 3, 1 ) = icpTransformation( 3, 1 );
    transformation( 3, 2 ) = icpTransformation( 3, 2 );
    transformation( 3, 3 ) = icpTransformation( 3, 3 );

    return transformation;
}

double WRegistrationICP::compute( const WRegistration::PointCloud& from, const WRegistration::PointCloud& to )
{
    WRegistration::MatrixTransformation initial = WRegistration::MatrixTransformation::identity();
    return compute( from, to, initial );
}

double WRegistrationICP::compute( const WRegistration::PointCloud& from, const WRegistration::PointCloud& to,
                WRegistration::MatrixTransformation initial )
{
    wlog::debug( CLASS ) << "Converting points to PCL type.";
    pcl::PointCloud< PointT >::Ptr pclFrom( new pcl::PointCloud< PointT > );
    for( WRegistration::PointCloud::size_type i = 0; i < from.size(); ++i )
    {
        Point p = from.at( i );
        pclFrom->push_back( PointT( p.x(), p.y(), p.z() ) );
    }

    pcl::PointCloud< PointT >::Ptr pclTo( new pcl::PointCloud< PointT > );
    for( WRegistration::PointCloud::size_type i = 0; i < to.size(); ++i )
    {
        Point p = to.at( i );
        pclTo->push_back( PointT( p.x(), p.y(), p.z() ) );
    }

    bool converged = computeICP( pclFrom, pclTo, initial );
    wlog::debug( CLASS ) << "compute() hasConverged: " << converged;

    // NOTE: m_icp->getFitnessScore() != WRegistration::meanSquareError
    // -> calculate MSE to compare RegistrationNaive and RegistrationICP
    Eigen::Matrix4f icpMat = m_icp->getFinalTransformation();

    Vector translation;
    translation.x() = icpMat( 0, 3 );
    translation.y() = icpMat( 1, 3 );
    translation.z() = icpMat( 2, 3 );

    MatrixRotation rotation;
    rotation( 0, 0 ) = icpMat( 0, 0 );
    rotation( 0, 1 ) = icpMat( 0, 1 );
    rotation( 0, 2 ) = icpMat( 0, 2 );

    rotation( 1, 0 ) = icpMat( 1, 0 );
    rotation( 1, 1 ) = icpMat( 1, 1 );
    rotation( 1, 2 ) = icpMat( 1, 2 );

    rotation( 2, 0 ) = icpMat( 2, 0 );
    rotation( 2, 1 ) = icpMat( 2, 1 );
    rotation( 2, 2 ) = icpMat( 2, 2 );

    std::vector< Point > rotated;
    rotated.resize( from.size() );
    std::transform( from.begin(), from.end(), rotated.begin(), boost::bind( WLGeometry::rotate, rotation, _1 ) );

    std::vector< Point > Pt;
    Pt.resize( from.size() );
    std::transform( rotated.begin(), rotated.end(), Pt.begin(), boost::bind( WLGeometry::tranlate, translation, _1 ) );

    double error = WRegistration::meanSquareError( Pt, WRegistration::closestPointCorresponces( Pt, to ) );

    return error;
}

bool WRegistrationICP::computeICP( const pcl::PointCloud< PointT >::Ptr from, const pcl::PointCloud< PointT >::Ptr to,
                Eigen::Matrix4f initial )
{
    m_icp->setInputCloud( from );
    m_icp->setInputTarget( to );

    m_icp->align( *from, initial );

    return m_icp->hasConverged();
}

