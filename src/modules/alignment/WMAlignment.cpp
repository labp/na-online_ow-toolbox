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
#include <pcl/point_types.h>
#include <pcl/pcl_base.h>
#include <pcl/common/common.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/registration/icp.h>

#include <core/graphicsEngine/WGEZoomTrackballManipulator.h>
#include <core/kernel/WKernel.h>

#include "core/data/WLDigPoint.h"
#include "core/data/emd/WLEMDEEG.h"
#include "core/module/WLModuleInputDataRingBuffer.h"
#include "core/module/WLModuleOutputDataCollectionable.h"
#include "core/util/profiler/WLTimeProfiler.h"
#include "WMAlignment.h"
#include "WMAlignment.xpm"

using namespace LaBP;

using pcl::Correspondence;
using pcl::Correspondences;
using pcl::IterativeClosestPoint;
using pcl::PointXYZ;
using pcl::PointCloud;
using pcl::registration::TransformationEstimationSVD;

// This line is needed by the module loader to actually find your module.
W_LOADABLE_MODULE( WMAlignment )

WMAlignment::WMAlignment()
{
}

WMAlignment::~WMAlignment()
{
    WKernel::getRunningKernel()->getGui()->closeCustomWidget( m_widget );
}

const std::string WMAlignment::getName() const
{
    return "Alignment";
}

const std::string WMAlignment::getDescription() const
{
    return "Alignment and registration for Fiducial and AC-PC coordinate system.";
}

WModule::SPtr WMAlignment::factory() const
{
    return WModule::SPtr( new WMAlignment() );
}

const char** WMAlignment::getXPMIcon() const
{
    return module_xpm;
}

void WMAlignment::connectors()
{
    m_input = LaBP::WLModuleInputDataRingBuffer< WLEMMCommand >::SPtr(
                    new LaBP::WLModuleInputDataRingBuffer< WLEMMCommand >( 8, shared_from_this(), "in",
                                    "Expects a EMM-DataSet for filtering." ) );
    addConnector( m_input );

    m_output = LaBP::WLModuleOutputDataCollectionable< WLEMMCommand >::SPtr(
                    new LaBP::WLModuleOutputDataCollectionable< WLEMMCommand >( shared_from_this(), "out",
                                    "Provides a filtered EMM-DataSet" ) );
    addConnector( m_output );
}

void WMAlignment::properties()
{
    WModule::properties();

    m_propCondition = WCondition::SPtr( new WCondition() );
    m_trgReset = m_properties->addProperty( "Reset:", "Reset", WPVBaseTypes::PV_TRIGGER_READY, m_propCondition );

    m_propEstGroup = m_properties->addPropertyGroup( "Transformation Estimation",
                    "Contains properties an initial transformation estimation.", false );
    // Defaults for intershift is05
    const WPosition LAP( -0.0754, -0.0131, -0.0520 );
    const WPosition NASION( -0.0012, 0.0836, -0.0526 );
    const WPosition RAP( 0.0706, -0.0140, -0.0613 );
    m_propEstLPA = m_propEstGroup->addProperty( "LPA (AC-PC):", "Left pre-auricular in AC-PC coordinate system.", LAP, false );
    m_propEstNasion = m_propEstGroup->addProperty( "Nasion (AC-PC):", "Nasion in AC-PC coordinate system.", NASION, false );
    m_propEstRPA = m_propEstGroup->addProperty( "RPA (AC-PC):", "Right pre-auricular in AC-PC coordinate system.", RAP, false );

    m_propIcpGroup = m_properties->addPropertyGroup( "ICP properties", "Contains properties for ICP.", false );
    m_propIcpIterations = m_propIcpGroup->addProperty( "Iterations:", "Maximum iterations for ICP algorithm.", 10, false );
    m_propIcpConverged = m_propIcpGroup->addProperty( "Converged:", "Indicates if ICP has converged.", false, false );
    m_propIcpConverged->setPurpose( PV_PURPOSE_INFORMATION );
    m_propIcpScore = m_propIcpGroup->addProperty( "Score:", "Fitness score of converged ICP.", -1.0, false );
    m_propIcpScore->setPurpose( PV_PURPOSE_INFORMATION );
}

void WMAlignment::viewInit()
{
    m_widget = WKernel::getRunningKernel()->getGui()->openCustomWidget( getName(), WGECamera::ORTHOGRAPHIC,
                    m_shutdownFlag.getCondition() );
    m_widget->getViewer()->setCameraManipulator( new WGEZoomTrackballManipulator() );

    m_drawable = WLEMDDrawable3DEEGBEM::SPtr( new WLEMDDrawable3DEEGBEM( m_widget ) );
}

void WMAlignment::viewUpdate( WLEMMeasurement::SPtr emm )
{
    if( m_widget->getViewer()->isClosed() )
    {
        return;
    }
    m_drawable->draw( emm );
}

void WMAlignment::moduleInit()
{
    infoLog() << "Initializing module ...";
    waitRestored();

    viewInit();
}

void WMAlignment::moduleMain()
{
    m_moduleState.setResetable( true, true ); // resetable, autoreset
    m_moduleState.add( m_input->getDataChangedCondition() ); // when inputdata changed
    m_moduleState.add( m_propCondition ); // when properties changed

    ready(); // signal ready state

    moduleInit();

    WLEMMCommand::SPtr cmdIn;
    while( !m_shutdownFlag() )
    {
        if( m_input->isEmpty() ) // continue processing if data is available
        {
            debugLog() << "Waiting for Events";
            m_moduleState.wait(); // wait for events like inputdata or properties changed
        }

        // ---------- SHUTDOWNEVENT ----------
        if( m_shutdownFlag() )
        {
            break; // break mainLoop on shutdown
        }

        cmdIn.reset();
        if( !m_input->isEmpty() )
        {
            cmdIn = m_input->getData();
        }
        const bool dataValid = ( cmdIn );

        // ---------- INPUTDATAUPDATEEVENT ----------
        if( dataValid ) // If there was an update on the inputconnector
        {
            process( cmdIn );
        }
    }
}

bool WMAlignment::processCompute( WLEMMeasurement::SPtr emm )
{
    WLEMMCommand::SPtr cmd( new WLEMMCommand( WLEMMCommand::Command::COMPUTE ) );
    cmd->setEmm( emm );
    WLEMMeasurement& emmRef = *emm;

    // TODO(pieloth): check if computation is needed!

    Fiducial eegPoints;
    if( !extractFiducialPoints( &eegPoints, emmRef ) )
    {
        m_output->updateData( cmd );
        return false;
    }

    Fiducial skinPoints;
    skinPoints.lpa = m_propEstLPA->get( false );
    skinPoints.nasion = m_propEstNasion->get( false );
    skinPoints.rpa = m_propEstRPA->get( false );

    PCLMatrixT trans;
    if( !estimateTransformation( &trans, eegPoints, skinPoints, emmRef ) )
    {
        m_output->updateData( cmd );
        return false;
    }

    double score = -1;
    if( !icpAlign( &trans, &score, emmRef, m_propIcpIterations->get( false ) ) )
    {
        m_output->updateData( cmd );
        m_propIcpConverged->set( false, false );
        return false;
    }
    m_propIcpConverged->set( true, false );
    m_propIcpScore->set( score, false );
#ifndef LABP_FLOAT_COMPUTATION
    emmRef.setFidToACPCTransformation( trans.cast< WLMatrix4::ScalarT >() );
#else
    emmRef.setFidToACPCTransformation( trans );
#endif
    viewUpdate( emm );

    m_output->updateData( cmd );
    return true;
}

bool WMAlignment::processInit( WLEMMCommand::SPtr cmd )
{
    // TODO
    m_output->updateData( cmd );
    return true;
}

bool WMAlignment::processReset( WLEMMCommand::SPtr cmd )
{
    // TODO
    m_output->updateData( cmd );
    return true;
}

bool WMAlignment::processTime( WLEMMCommand::SPtr cmd )
{
    // TODO
    m_output->updateData( cmd );
    return true;
}

bool WMAlignment::processMisc( WLEMMCommand::SPtr cmd )
{
    // TODO
    m_output->updateData( cmd );
    return true;
}

bool WMAlignment::extractFiducialPoints( Fiducial* const eegPoints, const WLEMMeasurement& emm )
{
    std::vector< WLDigPoint > digPoints = emm.getDigPoints( WLDigPoint::PointType::CARDINAL );
    char count = 0;
    std::vector< WLDigPoint >::const_iterator cit;
    for( cit = digPoints.begin(); cit != digPoints.end() && count < 3; ++cit )
    {
        if( cit->checkCardinal( WLDigPoint::CardinalPoints::LPA ) )
        {
            eegPoints->lpa = cit->getPoint();
            ++count;
        }
        if( cit->checkCardinal( WLDigPoint::CardinalPoints::NASION ) )
        {
            eegPoints->nasion = cit->getPoint();
            ++count;
        }
        if( cit->checkCardinal( WLDigPoint::CardinalPoints::RPA ) )
        {
            eegPoints->rpa = cit->getPoint();
            ++count;
        }
    }
    if( count > 2 )
    {
        return true;

    }
    else
    {
        warnLog() << "Could not found fiducial points: " << ( short )count;
        return false;
    }
}

bool WMAlignment::estimateTransformation( PCLMatrixT* const trans, const Fiducial& eegPoints, const Fiducial& skinPoints,
                const WLEMMeasurement& emm )
{
    WLTimeProfiler tp( getName(), "estimateTransformation" );

    PointCloud< PointXYZ > src, trg;

    src.push_back( PointXYZ( eegPoints.lpa.x(), eegPoints.lpa.y(), eegPoints.lpa.z() ) );
    trg.push_back( PointXYZ( skinPoints.lpa.x(), skinPoints.lpa.y(), skinPoints.lpa.z() ) );

    src.push_back( PointXYZ( eegPoints.nasion.x(), eegPoints.nasion.y(), eegPoints.nasion.z() ) );
    trg.push_back( PointXYZ( skinPoints.nasion.x(), skinPoints.nasion.y(), skinPoints.nasion.z() ) );

    src.push_back( PointXYZ( eegPoints.rpa.x(), eegPoints.rpa.y(), eegPoints.rpa.z() ) );
    trg.push_back( PointXYZ( skinPoints.rpa.x(), skinPoints.rpa.y(), skinPoints.rpa.z() ) );

    Correspondences corrs;
    corrs.push_back( Correspondence( 0, 0, std::numeric_limits< float >::max() ) );
    corrs.push_back( Correspondence( 1, 1, std::numeric_limits< float >::max() ) );
    corrs.push_back( Correspondence( 2, 2, std::numeric_limits< float >::max() ) );

    TransformationEstimationSVD< PointXYZ, PointXYZ > te;
    Eigen::Matrix< float, 4, 4 > guess;
    te.estimateRigidTransformation( src, trg, corrs, *trans );
    debugLog() << "Estimate transformation:\n" << *trans;

    return true;
}

bool WMAlignment::icpAlign( PCLMatrixT* const trans, double* const score, const WLEMMeasurement& emm, int maxIterations )
{
    WLTimeProfiler tp( getName(), "icpAlign" );

    debugLog() << "icpAlign: Get EEG sensor positions.";
    WLEMDEEG::ConstSPtr eeg;
    if( emm.hasModality( WEModalityType::EEG ) )
    {
        eeg = emm.getModality< const WLEMDEEG >( WEModalityType::EEG );
    }
    else
    {
        errorLog() << "icpAlign: No EEG data!";
        return false;
    }
    boost::shared_ptr< std::vector< WPosition > > eegPositions = eeg->getChannelPositions3d();

    debugLog() << "icpAlign: Get BEM skin layer.";
    WLEMMSubject::ConstSPtr subject = emm.getSubject();
    const std::vector< WLEMMBemBoundary::SPtr >& bems = subject->getBemBoundaries();
    std::vector< WLEMMBemBoundary::SPtr >::const_iterator itBem;
    WLEMMBemBoundary::ConstSPtr bemSkin;
    for( itBem = bems.begin(); itBem != bems.end(); ++itBem )
    {
        if( ( *itBem )->getBemType() == WEBemType::OUTER_SKIN )
        {
            bemSkin = *itBem;
            break;
        }
    }
    if( !bemSkin )
    {
        errorLog() << "icpAlign: No BEM skin layer available!";
        return false;
    }

    debugLog() << "icpAlign: Collect bottom sphere parameters.";
    const std::vector< WPosition >& bemPosition = bemSkin->getVertex();
    WPosition::ValueType min = std::numeric_limits< WPosition::ValueType >::max();
    WPosition::ValueType max = std::numeric_limits< WPosition::ValueType >::min();
    std::vector< WPosition >::const_iterator itPos;
    for( itPos = bemPosition.begin(); itPos != bemPosition.end(); ++itPos )
    {
        const WPosition::ValueType z = itPos->z();
        if( z < min )
        {
            min = z;
        }
        if( z > max )
        {
            max = z;
        }
    }
    const WPosition::ValueType z_threashold = min + ( max - min ) * 0.25;
    debugLog() << "icpAlign: BEM z_threashold: " << z_threashold;

    debugLog() << "icpAlign: Transforming WPosition to PCL::PointXYZ";
    PointCloud< PointXYZ > src;
    for( itPos = eegPositions->begin(); itPos != eegPositions->end(); ++itPos )
    {
        src.push_back( PointXYZ( itPos->x(), itPos->y(), itPos->z() ) );
    }

    PointCloud< PointXYZ > trg;
    WEExponent::Enum exp = bemSkin->getVertexExponent();
    float factor = WEExponent::factor( exp );
    size_t removed = 0;
    for( itPos = bemPosition.begin(); itPos != bemPosition.end(); ++itPos )
    {
        if( itPos->z() > z_threashold )
        {
            trg.push_back( PointXYZ( itPos->x() * factor, itPos->y() * factor, itPos->z() * factor ) );
        }
        else
        {
            ++removed;
        }
    }
    debugLog() << "Removed points from BEM: " << removed;

    debugLog() << "icpAlign: Run ICP";
    IterativeClosestPoint< PointXYZ, PointXYZ > icp;
    icp.setMaximumIterations( maxIterations );
    icp.setInputCloud( src.makeShared() );
    icp.setInputTarget( trg.makeShared() );

    icp.align( src, *trans );
    if( icp.hasConverged() )
    {
        *score = icp.getFitnessScore();
        *trans = icp.getFinalTransformation();
        debugLog() << "ICP score: " << *score;
        debugLog() << "ICP transformation:\n" << *trans;
        return true;
    }
    else
    {
        errorLog() << "icpAlign: ICP not converged!";
        return false;
    }
}
