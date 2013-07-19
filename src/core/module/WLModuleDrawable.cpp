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
#include <vector>

#include <boost/shared_ptr.hpp> // dynamic pointer cast
#include <core/common/WItemSelectionItemTyped.h>
#include <core/dataHandler/WDataSet.h>
#include <core/gui/WCustomWidget.h>
#include <core/gui/WGUI.h>
#include <core/kernel/WKernel.h>

#include "core/data/emd/WLEMData.h"
#include "core/data/WLMatrixTypes.h"
#include "core/data/WLEMMCommand.h"
#include "core/data/WLEMMeasurement.h"
#include "core/data/WLEMMEnumTypes.h"
#include "core/gui/drawable/WLEMDDrawable.h"
#include "core/gui/drawable/WLEMDDrawable2D.h"
#include "core/gui/drawable/WLEMDDrawable2DMultiChannel.h"
#include "core/gui/drawable/WLEMDDrawable3D.h"
#include "core/gui/drawable/WLEMDDrawable3DEEG.h"
#include "core/gui/events/WLGUIEvent.h"
#include "core/gui/events/WLMarkTimePositionHandler.h"
#include "core/gui/events/WL2DChannelScrollHandler.h"
#include "core/gui/colorMap/WLColorMap.h"
#include "core/util/WLBoundCalculator.h"

#include "WLModuleDrawable.h"

using std::min;
using std::max;

const int LaBP::WLModuleDrawable::AUTO_SCALE_PACKETS = 8;

LaBP::WLModuleDrawable::WLModuleDrawable()
{
    m_autoScaleCounter = AUTO_SCALE_PACKETS;
    m_graphType = WLEMDDrawable2D::WEGraphType::MULTI;
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
    m_channelHeight = m_propView->addProperty( "Channel height", "The distance between two curves of the graph in pixel.", 32.0,
                    boost::bind( &LaBP::WLModuleDrawable::callbackChannelHeightChanged, this ), false );
    m_channelHeight->setMin( 4.0 );
    m_channelHeight->setMax( 512.0 );

    m_labelsOn = m_propView->addProperty( "Labels on", "Switch channel labels on/off (3D).", true,
                    boost::bind( &LaBP::WLModuleDrawable::callbackLabelsChanged, this ), false );

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
                    boost::bind( &LaBP::WLModuleDrawable::callbackColorModeChanged, this ) );

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
                    boost::bind( &LaBP::WLModuleDrawable::callbackColorChanged, this ) );

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
                    boost::bind( &LaBP::WLModuleDrawable::callbackTimeRangeChanged, this ), true );
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
                    boost::bind( &LaBP::WLModuleDrawable::callbackViewModalityChanged, this ) );

    WPropertyHelper::PC_SELECTONLYONE::addTo( m_selectionView );
    WPropertyHelper::PC_NOTEMPTY::addTo( m_selectionView );

    m_autoSensitivity = m_propView->addProperty( "Sensitivity automatic", "Sensitivity automatic calculate.", true,
                    boost::bind( &LaBP::WLModuleDrawable::callbackAutoSensitivityChanged, this ), false );

    m_amplitudeScale = m_propView->addProperty( "Amplitude scale", "Scale of the amplitude / y-axis", 1.5e-9,
                    boost::bind( &LaBP::WLModuleDrawable::callbackAmplitudeScaleChanged, this ), true );
    m_minSensitity3D = m_propView->addProperty( "Min 3D scale", "Minimum data value for color map.", -1.5e-9,
                    boost::bind( &LaBP::WLModuleDrawable::callbackMin3DChanged, this ), true );
    m_maxSensitity3D = m_propView->addProperty( "Max 3D scale", "Maximum data value for color map.", 1.5e-9,
                    boost::bind( &LaBP::WLModuleDrawable::callbackMax3DChanged, this ), true );
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

void LaBP::WLModuleDrawable::callbackAutoSensitivityChanged()
{
    m_autoScaleCounter = AUTO_SCALE_PACKETS;
    m_amplitudeScale->setHidden( m_autoSensitivity->get() );
    m_minSensitity3D->setHidden( m_autoSensitivity->get() );
    m_maxSensitity3D->setHidden( m_autoSensitivity->get() );
}

void LaBP::WLModuleDrawable::callbackColorChanged()
{
    createColorMap();
    m_drawable3D->setColorMap( m_colorMap );
    m_drawable3D->redraw();
}

void LaBP::WLModuleDrawable::callbackColorModeChanged()
{
    createColorMap();
    m_drawable3D->setColorMap( m_colorMap );
    m_drawable3D->redraw();
}

void LaBP::WLModuleDrawable::callbackViewModalityChanged()
{
    viewReset();

    m_drawable2D->redraw();
    m_drawable3D->redraw();
}

void LaBP::WLModuleDrawable::callbackMin3DChanged()
{
    createColorMap();
    m_drawable3D->setColorMap( m_colorMap );
    m_drawable3D->redraw();
}

void LaBP::WLModuleDrawable::callbackMax3DChanged()
{
    createColorMap();
    m_drawable3D->setColorMap( m_colorMap );
    m_drawable3D->redraw();
}

void LaBP::WLModuleDrawable::callbackAmplitudeScaleChanged()
{
    m_drawable2D->setAmplitudeScale( m_amplitudeScale->get() );
    m_drawable2D->redraw();
}

void LaBP::WLModuleDrawable::callbackTimeRangeChanged()
{
    m_drawable2D->setTimeRange( m_timeRange->get() );
// TODO(pieloth): ?Code definition?
}

void LaBP::WLModuleDrawable::callbackChannelHeightChanged()
{
    WLEMDDrawable2DMultiChannel::SPtr drawable = m_drawable2D->getAs< WLEMDDrawable2DMultiChannel >();
    if( drawable )
    {
        drawable->setChannelHeight( static_cast< WLEMDDrawable::ValueT >( m_channelHeight->get() ) );
        drawable->redraw();
    }
}

void LaBP::WLModuleDrawable::callbackLabelsChanged()
{
    WLEMDDrawable3DEEG::SPtr drawable = m_drawable3D->getAs< WLEMDDrawable3DEEG >();
    if( drawable )
    {
        drawable->setLabels( m_labelsOn->get() );
        drawable->redraw();
    }
}

void LaBP::WLModuleDrawable::viewInit( LaBP::WLEMDDrawable2D::WEGraphType::Enum graphType )
{
    waitRestored();

    m_graphType = graphType;

    m_widget = WKernel::getRunningKernel()->getGui()->openCustomEMDWidget( getName(), WGECamera::TWO_D,
                    m_shutdownFlag.getCondition() );

    createColorMap();

    viewReset();
}

void LaBP::WLModuleDrawable::viewUpdate( WLEMMeasurement::SPtr emm )
{
    if( m_widget->getViewer()->isClosed() )
    {
        return;
    }

    bool autoScale = m_autoSensitivity->get() && m_autoScaleCounter > 0;

    if( autoScale )
    {
        WLBoundCalculator calculator;
        --m_autoScaleCounter;
        // Scale 2D
        const WLEMData::ScalarT amplitudeScale = calculator.getMax2D( emm, getViewModality() );
        m_amplitudeScale->set( amplitudeScale );
        // Scale 3D
        const WLEMData::ScalarT sens3dScale = calculator.getMax3D( emm, getViewModality() );
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

void LaBP::WLModuleDrawable::viewReset()
{
    debugLog() << "reset() called!";
    // Avoid memory leak due to circular references drawables <-> listener
    if( m_drawable2D )
    {
        m_drawable2D->clearListeners();
    }
    if( m_drawable3D )
    {
        m_drawable3D->clearListeners();
    }

    WCustomWidget::SPtr widget2D = m_widget->getSubWidget( LaBP::WLEMDWidget::WEWidgetType::EMD_2D );
    m_drawable2D = WLEMDDrawable2D::getInstance( widget2D, getViewModality(), m_graphType );
    WLEMDDrawable2DMultiChannel::SPtr drawable = m_drawable2D->getAs< WLEMDDrawable2DMultiChannel >();
    if( drawable )
    {
        drawable->setChannelHeight( static_cast< WLEMDDrawable::ValueT >( m_channelHeight->get() ) );
        WL2DChannelScrollHandler::SPtr handler( new WL2DChannelScrollHandler( drawable ) );
        drawable->addMouseEventListener( handler );
    }
    m_drawable2D->setTimeRange( m_timeRange->get() );
    m_drawable2D->setAmplitudeScale( m_amplitudeScale->get() );

    WCustomWidget::SPtr widget3D = m_widget->getSubWidget( LaBP::WLEMDWidget::WEWidgetType::EMD_3D );
    m_drawable3D = WLEMDDrawable3D::getInstance( widget3D, getViewModality() );
    m_drawable3D->setColorMap( m_colorMap );

    WLMarkTimePositionHandler::SPtr handler( new WLMarkTimePositionHandler( m_drawable2D, m_drawable3D, m_output ) );
    m_drawable2D->addMouseEventListener( handler );
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

bool LaBP::WLModuleDrawable::processTime( WLEMMCommand::SPtr labp )
{
    bool rc = true;
    const float relative = labp->getParameterAs< float >();
    rc &= m_drawable2D->setSelectedTime( relative );
    rc &= m_drawable3D->setSelectedTime( relative );
    m_output->updateData( labp );
    return rc;
}

bool LaBP::WLModuleDrawable::processMisc( WLEMMCommand::SPtr labp )
{
    m_output->updateData( labp );
    return false;
}
