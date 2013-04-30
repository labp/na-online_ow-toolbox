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

#include <algorithm> // min(), max()
#include <cmath>
#include <cstddef>
#include <string>

#include <core/common/WItemSelectionItemTyped.h>
#include <core/dataHandler/WDataSet.h>
#include <core/gui/WCustomWidget.h>
#include <core/gui/WGUI.h>
#include <core/kernel/WKernel.h>

#include "core/common/math/WLMatrixTypes.h"

#include "core/dataHandler/WDataSetEMM.h"
#include "core/dataHandler/WDataSetEMMEnumTypes.h"

#include "core/gui/drawable/WLEMDDrawable.h"
#include "core/gui/drawable/WLEMDDrawable2D.h"
#include "core/gui/drawable/WLEMDDrawable2DMultiChannel.h"
#include "core/gui/drawable/WLEMDDrawable3D.h"
#include "core/gui/events/WLMarkTimePositionHandler.h"
#include "core/graphicsEngine/WLColorMap.h"

#include "WLModuleDrawable.h"

using std::min;
using std::max;

const int LaBP::WLModuleDrawable::AUTO_SCALE_PACKETS = 8;

LaBP::WLModuleDrawable::WLModuleDrawable()
{
    m_autoScaleCounter = AUTO_SCALE_PACKETS;
    m_positions_changed = true;
    m_graphType = WLEMDDrawable2D::WEGraphType::MULTI;
    m_calculator = boost::shared_ptr< WLBoundCalculator >( new WLBoundCalculator() );
}

LaBP::WLModuleDrawable::~WLModuleDrawable()
{
    WKernel::getRunningKernel()->getGui()->closeCustomWidget( m_widget->getTitle() );
}

void LaBP::WLModuleDrawable::properties()
{
    WModule::properties();
    m_runtimeName->setPurpose( PV_PURPOSE_INFORMATION );
    m_propView = m_properties->addPropertyGroup( "View Properties", "Contains properties for the display module", false );

    // VIEWPROPERTIES ---------------------------------------------------------------------------------------
    m_channelHeight = m_propView->addProperty( "Channel height", "The distance between two curves of the graph in pixel.", 64.0,
                    boost::bind( &LaBP::WLModuleDrawable::handleChannelHeightChanged, this ), false );
    m_channelHeight->setMin( 4.0 );
    m_channelHeight->setMax( 512.0 );

    WItemSelection::SPtr colorModeSelection( new WItemSelection() );
    std::vector< WEColorMapMode::Enum > colorModes = WEColorMapMode::values();
    for( std::vector< WEColorMapMode::Enum >::iterator it = colorModes.begin(); it != colorModes.end(); ++it )
    {
        colorModeSelection->addItem(
                        WItemSelectionItemTyped< WEColorMapMode::Enum >::SPtr(
                                        new WItemSelectionItemTyped< WEColorMapMode::Enum >( *it, WEColorMapMode::name( *it ),
                                                        WEColorMapMode::name( *it ) ) ) );
    }

    m_selectionColorMode = m_propView->addProperty( "Color mode", "Select a mode", colorModeSelection->getSelectorFirst(),
                    boost::bind( &LaBP::WLModuleDrawable::handleColorModeChanged, this ) );

    WPropertyHelper::PC_SELECTONLYONE::addTo( m_selectionColorMode );
    WPropertyHelper::PC_NOTEMPTY::addTo( m_selectionColorMode );

    WItemSelection::SPtr colorMapSelection( new WItemSelection() );
    std::vector< WEColorMap::Enum > colors = WEColorMap::values();
    for( std::vector< WEColorMap::Enum >::iterator it = colors.begin(); it != colors.end(); ++it )
    {
        colorMapSelection->addItem(
                        WItemSelectionItemTyped< WEColorMap::Enum >::SPtr(
                                        new WItemSelectionItemTyped< WEColorMap::Enum >( *it, WEColorMap::name( *it ),
                                                        WEColorMap::name( *it ) ) ) );
    }

    m_selectionColor = m_propView->addProperty( "Color map", "Select a color", colorMapSelection->getSelector( 2 ),
                    boost::bind( &LaBP::WLModuleDrawable::handleColorChanged, this ) );

    WPropertyHelper::PC_SELECTONLYONE::addTo( m_selectionColor );
    WPropertyHelper::PC_NOTEMPTY::addTo( m_selectionColor );

    WItemSelection::SPtr calculateSelection( new WItemSelection() );
    std::vector< LaBP::WEModalityType::Enum > modalities_m = LaBP::WEModalityType::values();
    for( std::vector< LaBP::WEModalityType::Enum >::iterator it = modalities_m.begin(); it != modalities_m.end(); ++it )
    {
        calculateSelection->addItem(
                        WItemSelectionItemTyped< LaBP::WEModalityType::Enum >::SPtr(
                                        new WItemSelectionItemTyped< LaBP::WEModalityType::Enum >( *it,
                                                        LaBP::WEModalityType::name( *it ),
                                                        LaBP::WEModalityType::description( *it ) ) ) );
    }

    m_selectionCalculate = m_propView->addProperty( "Compute modality", "Select a modality to compute.",
                    calculateSelection->getSelectorFirst() );

    WPropertyHelper::PC_SELECTONLYONE::addTo( m_selectionCalculate );
    WPropertyHelper::PC_NOTEMPTY::addTo( m_selectionCalculate );

    // TODO(pieloth): unit = actually seconds, but we are currently ignoring it, cause we take the whole width to draw the channels
    // we are not using it, also it will be hidden until a solution is found. Dynamic view divide the width into the amount
    // of blocks ( default 2 ), also timeRange is not longer required.
    m_timeRange = m_propView->addProperty( "Time Range", "Size of time windows in ???.", 1.0,
                    boost::bind( &LaBP::WLModuleDrawable::handleTimeRangeChanged, this ), true );
    m_timeRange->setMin( 0.100 );
    m_timeRange->setMax( 4.0 );

    WItemSelection::SPtr viewSelection( new WItemSelection() );
    std::vector< LaBP::WEModalityType::Enum > modalities = LaBP::WEModalityType::values();
    for( std::vector< LaBP::WEModalityType::Enum >::iterator it = modalities.begin(); it != modalities.end(); ++it )
    {
        viewSelection->addItem(
                        WItemSelectionItemTyped< LaBP::WEModalityType::Enum >::SPtr(
                                        new WItemSelectionItemTyped< LaBP::WEModalityType::Enum >( *it,
                                                        LaBP::WEModalityType::name( *it ),
                                                        LaBP::WEModalityType::description( *it ) ) ) );
    }

    m_selectionView = m_propView->addProperty( "View modality", "Select a to visualize", viewSelection->getSelectorFirst(),
                    boost::bind( &LaBP::WLModuleDrawable::handleViewModalityChanged, this ) );

    WPropertyHelper::PC_SELECTONLYONE::addTo( m_selectionView );
    WPropertyHelper::PC_NOTEMPTY::addTo( m_selectionView );

    m_autoSensitivity = m_propView->addProperty( "Sensitivity automatic", "Sensitivity automatic calculate.", true,
                    boost::bind( &LaBP::WLModuleDrawable::handleAutoSensitivityChanged, this ), false );

    m_amplitudeScale = m_propView->addProperty( "Amplitude scale", "Scale of the amplitude / y-axis", 1.5e-9,
                    boost::bind( &LaBP::WLModuleDrawable::handleAmplitudeScaleChanged, this ), true );
    m_minSensitity3D = m_propView->addProperty( "Min 3D scale", "Minimum data value for color map.", -1.5e-9,
                    boost::bind( &LaBP::WLModuleDrawable::handleMin3DChanged, this ), true );
    m_maxSensitity3D = m_propView->addProperty( "Max 3D scale", "Maximum data value for color map.", 1.5e-9,
                    boost::bind( &LaBP::WLModuleDrawable::handleMax3DChanged, this ), true );
}

LaBP::WEModalityType::Enum LaBP::WLModuleDrawable::getViewModality()
{
    return m_selectionView->get().at( 0 )->getAs< WItemSelectionItemTyped< LaBP::WEModalityType::Enum > >()->getValue();
}

LaBP::WEModalityType::Enum LaBP::WLModuleDrawable::getCalculateModality()
{
    LaBP::WEModalityType::Enum modality;
    modality = m_selectionCalculate->get().at( 0 )->getAs< WItemSelectionItemTyped< LaBP::WEModalityType::Enum > >()->getValue();
    return modality;
}

void LaBP::WLModuleDrawable::hideComputeModalitySelection( bool enable )
{
    m_selectionCalculate->setHidden( enable );
}

void LaBP::WLModuleDrawable::handleAutoSensitivityChanged()
{
    m_autoScaleCounter = AUTO_SCALE_PACKETS;
    m_amplitudeScale->setHidden( m_autoSensitivity->get() );
    m_minSensitity3D->setHidden( m_autoSensitivity->get() );
    m_maxSensitity3D->setHidden( m_autoSensitivity->get() );
}

void LaBP::WLModuleDrawable::handleColorChanged()
{
    createColorMap();
    m_drawable3D->setColorMap( m_colorMap );
    m_drawable3D->redraw();
}

void LaBP::WLModuleDrawable::handleColorModeChanged()
{
    createColorMap();
    m_drawable3D->setColorMap( m_colorMap );
    m_drawable3D->redraw();
}

void LaBP::WLModuleDrawable::handleViewModalityChanged()
{
    m_drawable2D->clearWidget();
    m_drawable3D->clearWidget( true );

    resetView();

    m_drawable2D->redraw();
    m_drawable3D->redraw();
}

void LaBP::WLModuleDrawable::handleMin3DChanged()
{
    createColorMap();
    m_drawable3D->setColorMap( m_colorMap );
    m_drawable3D->redraw();
}

void LaBP::WLModuleDrawable::handleMax3DChanged()
{
    createColorMap();
    m_drawable3D->setColorMap( m_colorMap );
    m_drawable3D->redraw();
}

void LaBP::WLModuleDrawable::handleAmplitudeScaleChanged()
{
    m_drawable2D->setAmplitudeScale( m_amplitudeScale->get() );
    m_drawable2D->redraw();
}

void LaBP::WLModuleDrawable::handleTimeRangeChanged()
{
    m_drawable2D->setTimeRange( m_timeRange->get() );
// TODO: Code definition
}

void LaBP::WLModuleDrawable::handleChannelHeightChanged()
{
    WLEMDDrawable2DMultiChannel::SPtr drawable = m_drawable2D->getAs< WLEMDDrawable2DMultiChannel >();
    if( drawable )
    {
        drawable->setChannelHeight( static_cast< WLEMDDrawable::ValueT >( m_channelHeight->get() ) );
        drawable->redraw();
    }
}

void LaBP::WLModuleDrawable::initView( LaBP::WLEMDDrawable2D::WEGraphType::Enum graphType )
{
    waitRestored();

    m_graphType = graphType;

    m_widget = WKernel::getRunningKernel()->getGui()->openCustomEMDWidget( getName(), WGECamera::TWO_D,
                    m_shutdownFlag.getCondition() );

// TODO: Skizze von Mirco ? Sollte er raus oder nicht?
//    m_widget->setVisibility( showChannels, 0 );
//    m_widget->setVisibility( showHead, 1 );
//    m_widget->setVisibility( showInfo, 2 );
//    m_widget->setVisibility( showInfo, 3 );
//    m_widget->setVisibility( showInfo, 4 );
//    m_widget->setVisibility( showInfo, 5 );
//    m_widget->setVisibility( showInfo, 6 );

    createColorMap();

    resetView();
}

void LaBP::WLModuleDrawable::updateView( boost::shared_ptr< LaBP::WDataSetEMM > emm )
{
    if( m_widget->getViewer()->isClosed() )
    {
        return;
    }

    bool autoScale = m_autoSensitivity->get() && m_autoScaleCounter > 0;

    if( autoScale )
    {
        --m_autoScaleCounter;
        // Scale 2D
        const LaBP::WDataSetEMMEMD::SampleT amplitudeScale = m_calculator->getMax2D( emm, getViewModality() );
        m_amplitudeScale->set( amplitudeScale );
        // Scale 3D
        const LaBP::WDataSetEMMEMD::SampleT sens3dScale = m_calculator->getMax3D( emm, getViewModality() );
        m_maxSensitity3D->set( sens3dScale );
        m_minSensitity3D->set( -sens3dScale );
    }
    if( m_autoScaleCounter == 0 )
    {
        m_autoSensitivity->set( false, true );
    }

    m_drawable2D->draw( emm );
    m_drawable3D->draw( emm );
}

void LaBP::WLModuleDrawable::resetView()
{
    debugLog() << "reset() called!";
    m_eventHandler.clear();

    WCustomWidget::SPtr widget2D = m_widget->getSubWidget( LaBP::WLEMDWidget::WEWidgetType::EMD_2D );
    m_drawable2D = WLEMDDrawable2D::getInstance( widget2D, getViewModality(), m_graphType );
    WLEMDDrawable2DMultiChannel::SPtr drawable = m_drawable2D->getAs< WLEMDDrawable2DMultiChannel >();
    if( drawable )
    {
        drawable->setChannelHeight( static_cast< WLEMDDrawable::ValueT >( m_channelHeight->get() ) );
    }
    m_drawable2D->setTimeRange( m_timeRange->get() );
    // TODO (pieloth): m_drawable2D->setLabelWidth( m_labelsWidth->get() );
    m_drawable2D->setAmplitudeScale( m_amplitudeScale->get() );
    //m_drawable2D->setTimePosition( 0.0 );

    WCustomWidget::SPtr widget3D = m_widget->getSubWidget( LaBP::WLEMDWidget::WEWidgetType::EMD_3D );
    m_drawable3D = WLEMDDrawable3D::getInstance( widget3D, getViewModality() );
    m_drawable3D->setColorMap( m_colorMap );
    //m_drawable3D->setTimePosition( 0.0 );

    WLMarkTimePositionHandler::SPtr handler( new WLMarkTimePositionHandler( m_drawable2D, m_drawable3D ) );
    m_eventHandler.push_back( handler );
}

double LaBP::WLModuleDrawable::getTimerange()
{
    return m_timeRange->get();
}

void LaBP::WLModuleDrawable::setTimerange( double value )
{
    value = max( value, m_timeRange->getMin()->getMin() );
    value = min( value, m_timeRange->getMax()->getMax() );
    m_timeRange->set( value );
}

void LaBP::WLModuleDrawable::setTimerangeInformationOnly( bool enable )
{
    if( enable == true )
    {
        m_timeRange->setPurpose( PV_PURPOSE_INFORMATION );
    }
    else
    {
        m_timeRange->setPurpose( PV_PURPOSE_PARAMETER );
    }
}

void LaBP::WLModuleDrawable::createColorMap()
{
// create color map
    const WEColorMap::Enum color_map =
                    m_selectionColor->get().at( 0 )->getAs< WItemSelectionItemTyped< WEColorMap::Enum > >()->getValue();
    const WEColorMapMode::Enum color_mode = m_selectionColorMode->get().at( 0 )->getAs<
                    WItemSelectionItemTyped< WEColorMapMode::Enum > >()->getValue();
    m_colorMap = WEColorMap::instance( color_map, m_minSensitity3D->get(), m_maxSensitity3D->get(), color_mode );
}
