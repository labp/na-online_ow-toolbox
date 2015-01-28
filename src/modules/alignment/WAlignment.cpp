//---------------------------------------------------------------------------
//
// Project: NA-Online ( http://www.labp.htwk-leipzig.de )
//
// Copyright 2010 Laboratory for Biosignal Processing, HTWK Leipzig, Germany
//
// This file is part of NA-Online.
//
// NA-Online is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// NA-Online is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with NA-Online. If not, see <http://www.gnu.org/licenses/>.
//
//---------------------------------------------------------------------------

#include <limits>
#include <list>
#include <string>

#include <pcl/correspondence.h>
#include <pcl/pcl_base.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/registration/icp.h>

#include <core/common/WLogger.h>
#include <core/common/exceptions/WPreconditionNotMet.h>

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

double WAlignment::align( TransformationT* const transformation, const PointsT& from, const PointsT& to )
{
    WLTimeProfiler tp( CLASS, __func__ );

    // Error checking
    // --------------
    if( transformation == NULL )
    {
        wlog::error( CLASS ) << "Matrix pointer is null!";
        return NOT_CONVERGED;
    }

    if( from.empty() || to.empty() )
    {
        wlog::error( CLASS ) << "FROM or TO points are empty!";
        return NOT_CONVERGED;
    }

    if( from.unit() != WLEUnit::UNKNOWN && to.unit() != WLEUnit::UNKNOWN )
    {
        if( from.unit() != to.unit() )
        {
            throw WPreconditionNotMet( "Units are not equals!" );
        }
        if( from.exponent() != to.exponent() )
        {
            throw WPreconditionNotMet( "Exponents are not equals!" );
        }
    }
    else
    {
        wlog::warn( CLASS ) << "Translation unit is not set and could not be checked: " << from.unit() << "/" << to.unit();
    }

    // Align
    // -----
    PCLMatrixT pclMatrix = PCLMatrixT::Identity();
    if( !m_correspondences.empty() )
    {
        estimateTransformation( &pclMatrix );
    }
    else
    {
        if( !transformation->empty() )
        {
            wlog::info( CLASS ) << "Using matrix as initial transformation.";
            pclMatrix = transformation->data().cast< PCLMatrixT::Scalar >();
        }
    }

    const double score = icpAlign( &pclMatrix, from, to );

    // Prepare output
    // --------------
    transformation->from( from.coordSystem() );
    transformation->to( to.coordSystem() );
    transformation->unit( from.unit() );
    transformation->exponent( from.exponent() );
    transformation->data() = pclMatrix.cast< TransformationT::ScalarT >();

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

    TransformationEstimationSVD < PointXYZ, PointXYZ > te;
    te.estimateRigidTransformation( src, trg, corrs, *matrix );
    wlog::debug( CLASS ) << "Estimate transformation:\n" << *matrix;

    return !matrix->isIdentity();
}

double WAlignment::icpAlign( PCLMatrixT* const trans, const PointsT& from, const PointsT& to )
{
    WLTimeProfiler tp( CLASS, "icpAlign" );

    wlog::debug( CLASS ) << "icpAlign: Transforming WPosition to PCL::PointXYZ";
    PointCloud < PointXYZ > src;
    for( PointsT::IndexT i = 0; i < from.size(); ++i )
    {
        const PointsT::PositionT tmp = from.at( i );
        src.push_back( PointXYZ( tmp.x(), tmp.y(), tmp.z() ) );
    }

    PointCloud < PointXYZ > trg;
    for( PointsT::IndexT i = 0; i < to.size(); ++i )
    {
        const PointsT::PositionT tmp = to.at( i );
        trg.push_back( PointXYZ( tmp.x(), tmp.y(), tmp.z() ) );
    }

    wlog::debug( CLASS ) << "icpAlign: Run ICP";
    IterativeClosestPoint < PointXYZ, PointXYZ > icp;
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

