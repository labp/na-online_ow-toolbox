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

#ifndef WLMODULEDRAWABLE_H
#define WLMODULEDRAWABLE_H

#include <list>
#include <set>

#include <boost/shared_ptr.hpp>

#include <core/common/WProperties.h>
#include <core/ui/WUIGridWidget.h>
#include <core/kernel/WModule.h>

#include "core/data/WLEMMCommand.h"
#include "core/data/WLEMMeasurement.h"
#include "core/data/enum/WLEModality.h"
#include "core/gui/colorMap/WLColorMap.h"
#include "core/gui/drawable/WLEMDDrawable2D.h"
#include "core/gui/drawable/WLEMDDrawable3D.h"
#include "core/gui/events/WL2DChannelScrollHandler.h"
#include "core/gui/events/WLMarkTimePositionHandler.h"
#include "core/gui/events/WLResizeHandler.h"
#include "core/module/WLModuleOutputDataCollectionable.h"
#include "core/module/WLEMMCommandProcessor.h"
#include "core/util/bounds/WLABoundCalculator.h"

/**
 * Virtual implementation of WModule to let NA-Online modules use a VIEW including just a few lines of code.
 *
 * \author pieloth
 * \ingroup module
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
    virtual bool processTime( WLEMMCommand::SPtr cmdIn );
    virtual bool processMisc( WLEMMCommand::SPtr cmdIn );

    WLModuleOutputDataCollectionable< WLEMMCommand >::SPtr m_output; /**< Output connector for buffered input connectors. */

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
     * Must be called from derived classes in moduleMain after receiving a shutdown.
     */
    void viewCleanup();

    /**
     * Initializes the underlying algorithm with the values of WProperties. Called it after ready()!
     */
    virtual void moduleInit() = 0;

    WLEModality::Enum getViewModality();

    void setViewModality( WLEModality::Enum mod );

    /**
     * Clears and sets the view modality selection.
     *
     * \param mods List of modalities to select.
     */
    void setViewModalitySelection( std::list< WLEModality::Enum > mods );

    void hideViewModalitySelection( bool enable );

    WLEModality::Enum getCalculateModality();

    double getTimerange();

    void setTimerange( double value );

    void setTimerangeInformationOnly( bool enable );

    void hideComputeModalitySelection( bool enable );

    void setComputeModalitySelection( const std::set< WLEModality::Enum >& modalities );

    void hideLabelChanged( bool enable );

    /**
     * Gets the current bound calculator.
     *
     * @return Returns a shared pinter on the bound calculator.
     */
    WLABoundCalculator::SPtr getBoundCalculator();

    /**
     * Sets the current bound calculator.
     *
     * @param calculator The bound calculator.
     */
    void setBoundCalculator( WLABoundCalculator::SPtr calculator );

    /**
     * Gets the view properties group.
     * This properties can be used to add some properties to the group from concrete modules.
     * The bound calculator uses this behaviour for example.
     *
     * @return Returns a shared pointer on the constant property group.
     */
    boost::shared_ptr< WPVGroup > getViewProperties();

    /**
     * Gets the last processed EMM object.
     *
     * @return Returns a shared pointer on a WLEMMeasurement.
     */
    WLEMMeasurement::SPtr getLastEMM();

    /**
     * Sets the last processed EMM object.
     *
     * @param emm The EMM obejct.
     */
    void setLastEMM( WLEMMeasurement::SPtr emm );

    /**
     * Uses the current bound calcluator to determine the bounds for 3D and 2D views.
     */
    void calcBounds();

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

    WUIGridWidget::SPtr m_widget;
    WUIViewWidget::SPtr m_widget2D;
    WUIViewWidget::SPtr m_widget3D;

    WL2DChannelScrollHandler::RefPtr m_scrollHandler;

    WLMarkTimePositionHandler::RefPtr m_clickHandler;

    WLResizeHandler::RefPtr m_resize2dHandler;

    WLResizeHandler::RefPtr m_resize3dHandler;

    WPropDouble m_amplitudeScale;

    WPropDouble m_minSensitity3D;

    WPropDouble m_maxSensitity3D;

    int m_autoScaleCounter;

    WLColorMap::SPtr m_colorMap;

    double m_range;

    /**
     * Calculates the boundaries for 2D and 3D views based on the signal data.
     */
    WLABoundCalculator::SPtr m_boundCalculator;

    /**
     * The last processed EMM object.
     */
    WLEMMeasurement::SPtr m_lastEmm;
};

#endif  // WLMODULEDRAWABLE_H
