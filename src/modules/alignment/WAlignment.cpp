//---------------------------------------------------------------------------
//
// Project: OpenWalnut ( http://www.openwalnut.org )
//
// Copyright 2009 OpenWalnut Community, BSV@Uni-Leipzig and CNCF@MPI-CBS
// For more information see http://www.openwalnut.org/copying
//
// This file is part of OpenWalnut.
//
// OpenWalnut is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// OpenWalnut is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with OpenWalnut. If not, see <http://www.gnu.org/licenses/>.
//
//---------------------------------------------------------------------------

#include <limits>
#include <vector>

#include <pcl/correspondence.h>
#include <pcl/pcl_base.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/registration/icp.h>

#include <core/common/WLogger.h>

#include "core/util/profiler/WLTimeProfiler.h"
#include "WAlignment.h"

using pcl::Correspondence;
using pcl::Correspondences;
using pcl::IterativeClosestPoint;
using pcl::PointXYZ;
using pcl::PointCloud;
using pcl::registration::TransformationEstimationSVD;

const std::string WAlignment::CLASS = "WAlignment";

const double WAlignment::NOT_CONVERGED = -1.0;

WAlignment::WAlignment( int maxInterations ) :
                m_maxIterations( maxInterations )
{
}

WAlignment::~WAlignment()
{
}

void WAlignment::addCorrespondence( const CorrespondenceT& cor )
{
    m_correspondences.push_back( cor );
}
void WAlignment::clearCorrespondences()
{
    m_correspondences.clear();
}

double WAlignment::align( TransformationT* const matrix, const PointsT& from, const PointsT& to )
{
    WLTimeProfiler tp( CLASS, "align" );

    PCLMatrixT pclMatrix = PCLMatrixT::Identity();
    if( !m_correspondences.empty() )
    {
        estimateTransformation( &pclMatrix );
    }
    else
    {
        if( !matrix->isZero() && !matrix->isIdentity() )
        {
            wlog::info( CLASS ) << "Using matrix as initial transformation.";
#ifndef LABP_FLOAT_COMPUTATION
            pclMatrix = matrix->cast< PCLMatrixT::Scalar >();
#else
            pclMatrix = *matrix;
#endif
        }
    }

    const double score = icpAlign( &pclMatrix, from, to );
#ifndef LABP_FLOAT_COMPUTATION
    *matrix = pclMatrix.cast< TransformationT::Scalar >();
#else
    *matrix = pclMatrix;
#endif

    return score;
}

bool WAlignment::estimateTransformation( PCLMatrixT* const matrix )
{
    WLTimeProfiler tp( CLASS, "estimateTransformation" );

    PointCloud< PointXYZ > src, trg;
    Correspondences corrs;
    int count = 0;

    std::list< CorrespondenceT >::const_iterator it;
    for( it = m_correspondences.begin(); it != m_correspondences.end(); ++it )
    {
        src.push_back( PointXYZ( it->first.x(), it->first.y(), it->first.z() ) );
        trg.push_back( PointXYZ( it->second.x(), it->second.y(), it->second.z() ) );

        corrs.push_back( Correspondence( count, count, std::numeric_limits< float >::max() ) );

        ++count;
    }

    TransformationEstimationSVD< PointXYZ, PointXYZ > te;
    te.estimateRigidTransformation( src, trg, corrs, *matrix );
    wlog::debug( CLASS ) << "Estimate transformation:\n" << *matrix;

    return !matrix->isIdentity();
}

double WAlignment::icpAlign( PCLMatrixT* const trans, const PointsT& from, const PointsT& to )
{
    WLTimeProfiler tp( CLASS, "icpAlign" );

    wlog::debug( CLASS ) << "icpAlign: Transforming WPosition to PCL::PointXYZ";
    PointCloud< PointXYZ > src;
    PointsT::const_iterator itPos;
    for( itPos = from.begin(); itPos != from.end(); ++itPos )
    {
        src.push_back( PointXYZ( itPos->x(), itPos->y(), itPos->z() ) );
    }

    PointCloud< PointXYZ > trg;
    for( itPos = to.begin(); itPos != to.end(); ++itPos )
    {
        trg.push_back( PointXYZ( itPos->x(), itPos->y(), itPos->z() ) );
    }

    wlog::debug( CLASS ) << "icpAlign: Run ICP";
    IterativeClosestPoint< PointXYZ, PointXYZ > icp;
    icp.setMaximumIterations( m_maxIterations );
    icp.setInputCloud( src.makeShared() );
    icp.setInputTarget( trg.makeShared() );

    icp.align( src, *trans );
    double score = NOT_CONVERGED;
    if( icp.hasConverged() )
    {
        score = icp.getFitnessScore();
        *trans = icp.getFinalTransformation();
        wlog::debug( CLASS ) << "ICP score: " << score;
        wlog::debug( CLASS ) << "ICP transformation:\n" << *trans;
        return score;
    }
    else
    {
        wlog::error( CLASS ) << "icpAlign: ICP not converged!";
        return score;
    }
}

