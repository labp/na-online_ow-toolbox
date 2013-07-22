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

#ifndef WLMODULEDRAWABLE_H
#define WLMODULEDRAWABLE_H

#include <set>

#include <boost/shared_ptr.hpp>

#include <core/common/WProperties.h>
#include <core/gui/WLEMDWidget.h>
#include <core/kernel/WModule.h>

#include "core/data/WLEMMCommand.h"
#include "core/data/WLEMMeasurement.h"
#include "core/gui/colorMap/WLColorMap.h"
#include "core/gui/drawable/WLEMDDrawable2D.h"
#include "core/gui/drawable/WLEMDDrawable3D.h"
#include "core/module/WLModuleOutputDataCollectionable.h"
#include "core/module/WLEMMCommandProcessor.h"

using std::set;
using namespace LaBP;

/**
 * Virtual Implementation of WModule to let our modules use a VIEW including just 4 lines! of code
 */
class WLModuleDrawable: public WModule, public WLEMMCommandProcessor
{
public:
    /**
     * Abbreviation for a shared pointer.
     */
    typedef boost::shared_ptr< WLModuleDrawable > SPtr;

    /**
     * Abbreviation for const shared pointer.
     */
    typedef boost::shared_ptr< const WLModuleDrawable > ConstSPtr;

    /**
     * standard constructor
     */
    WLModuleDrawable();

    /**
     * destructor
     */
    virtual ~WLModuleDrawable();

protected:
    // ----------------------------
    // Methods from WLEMMCommandProcessor
    // ----------------------------
    virtual bool processTime( WLEMMCommand::SPtr labp );
    virtual bool processMisc( WLEMMCommand::SPtr labp );

    /**
     * Output connector for a filtered WEEG2 dataset
     */
    WLModuleOutputDataCollectionable< WLEMMCommand >::SPtr m_output;

    /**
     * Initialize the properties for this module and load the widget
     */
    virtual void properties();

    /**
     * Sets the new data to draw.
     */
    void viewUpdate( WLEMMeasurement::SPtr emm );

    /**
     * Set which elements of the view we want to see: info panels, channels and/or head. Called it after ready()!
     */
    void viewInit( WLEMDDrawable2D::WEGraphType::Enum graphType );

    void viewReset();

    /**
     * Initializes the underlying algorithm with the values of WProperties. Called it after ready()!
     */
    virtual void moduleInit() = 0;

    WEModalityType::Enum getViewModality();

    void setViewModality( WEModalityType::Enum mod );

    void hideViewModalitySelection( bool enable );

    WEModalityType::Enum getCalculateModality();

    double getTimerange();

    void setTimerange( double value );

    void setTimerangeInformationOnly( bool enable );

    void hideComputeModalitySelection( bool enable );

    void setComputeModalitySelection( const set< WEModalityType::Enum >& modalities );

    WLEMDDrawable2D::SPtr m_drawable2D;

    WLEMDDrawable3D::SPtr m_drawable3D;

private:
    void createColorMap();

    void callbackTimeRangeChanged();

    void callbackChannelHeightChanged();

    void callbackColorChanged();

    void callbackColorModeChanged();

    void callbackViewModalityChanged();

    void callbackMin3DChanged();

    void callbackMax3DChanged();

    void callbackAmplitudeScaleChanged();

    void callbackAutoSensitivityChanged();

    void callbackLabelsChanged();

    WLEMDDrawable2D::WEGraphType::Enum m_graphType;

    WPropGroup m_propView;

    WPropBool m_autoSensitivity;

    WPropBool m_labelsOn;

    /**
     * the width of the graph in seconds as property
     */
    WPropDouble m_timeRange;

    /**
     * the distance between two curves of the graph in pixel as property
     */
    WPropDouble m_channelHeight;

    WPropSelection m_selectionColor;

    WPropSelection m_selectionView;

    WPropSelection m_selectionCalculate;

    WPropSelection m_selectionColorMode;

    WLEMDWidget::SPtr m_widget;

    WPropDouble m_amplitudeScale;

    WPropDouble m_minSensitity3D;

    WPropDouble m_maxSensitity3D;

    int m_autoScaleCounter;

    WLColorMap::SPtr m_colorMap;

    static const int AUTO_SCALE_PACKETS;
};

#endif  // WLMODULEDRAWABLE_H
