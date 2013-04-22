/**
 * TODO license and documentation
 */

#include <algorithm>
#include <cmath>

#include <boost/bind.hpp>

#include "core/common/WAssert.h"

#include "WGeometry.h"
#include "WRegistration.h"

WRegistration::WRegistration()
{
    // TODO Auto-generated constructor stub

}

WRegistration::~WRegistration()
{
    // TODO Auto-generated destructor stub
}

WRegistration::PointCloud WRegistration::closestPointCorresponces( const PointCloud& P, const PointCloud& X )
{
    PointCloud corresponces;
    corresponces.reserve( P.size() );
    corresponces.resize( P.size() );

    std::transform( P.begin(), P.end(), corresponces.begin(), boost::bind( WGeometry::minDistance, _1, X ) );

    return corresponces;
}

double WRegistration::meanSquareError( const PointCloud& P, const PointCloud& X )
{
    double mse = 0;

    WAssertDebug( P.size() == X.size(), "Point clouds have different sizes!" );
    for( PointCloud::size_type i = 0; i < P.size(); ++i )
    {
        mse += pow( WGeometry::distance( P.at( i ), X.at( i ) ), 2 );
    }

    mse = mse / P.size();
    return mse;
}
