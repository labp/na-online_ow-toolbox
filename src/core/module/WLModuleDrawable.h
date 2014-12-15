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
 * Extends WModule for NA-Online modules by a separated view, view properties and
 * an output connector for buffered input connectors. To provide a buffered input connector in each signal processing step,
 * NA-Online modules should use instances of WLModuleInputDataCollection and
 * WLModuleOutputDataCollectionable as connectors.
 *
 * \author pieloth
 * \ingroup module
 */
class WLModuleDrawable: public WModule, public WLEMMCommandProcessor
{
public:
    typedef boost::shared_ptr< WLModuleDrawable > SPtr; //!< Abbreviation for a shared pointer.

    typedef boost::shared_ptr< const WLModuleDrawable > ConstSPtr; //!< Abbreviation for const shared pointer.

    /**
     * Constructor.
     */
    WLModuleDrawable();

    /**
     * Destructor.
     */
    virtual ~WLModuleDrawable();

protected:
    // ----------------------------
    // Methods from WLEMMCommandProcessor
    // ----------------------------
    virtual bool processTime( WLEMMCommand::SPtr cmdIn );
    virtual bool processMisc( WLEMMCommand::SPtr cmdIn );

    WLModuleOutputDataCollectionable< WLEMMCommand >::SPtr m_output; //!< Output connector for buffered input connectors.

    virtual void properties();

    /**
     * Gets the view properties group.
     * This properties can be used to add some properties to the group from concrete modules.
     * The bound calculator uses this behaviour for example.
     *
     * \return Returns a shared pointer on the constant property group.
     */
    WPVGroup::SPtr getViewProperties();

    /**
     * Updates all view components with the new data. No action if widget is closed or not visible.
     *
     * \param emm New data.
     */
    void viewUpdate( WLEMMeasurement::SPtr emm );

    /**
     * Initializes the view for the requested graph type. 2D, 3D view and all necessary event handlers are created.
     * \note Must be called after WModule::ready()!
     *
     * \param graphType Graph type for this module.
     */
    void viewInit( WLEMDDrawable2D::WEGraphType::Enum graphType );

    /**
     * Resets all view components.
     */
    void viewReset();

    /**
     * Unloads all view components, especially the event handlers.
     * \note Must be called from in WModule::moduleMain() after receiving a shutdown.
     */
    void viewCleanup();

    /**
     * \brief Initializes the module/algorithm with the all necessary values.
     * Use it to initialized the module, to initialize the view, to initialize the algorithm, to load a stored module and
     * to set/create all initial values.
     *
     * \note Should be the first call in WModule::moduleMain().
     */
    virtual void moduleInit() = 0;

    /**
     * Returns the modality which is visualized.
     *
     * \return Modality type to visualize.
     */
    WLEModality::Enum getViewModality();

    /**
     * Sets the modality, which schoul be visualize.
     * \param mod Modality type to visualize.
     */
    void setViewModality( WLEModality::Enum mod );

    /**
     * Clears and sets the view modality selection.
     *
     * \param mods List of modalities to select.
     */
    void setViewModalitySelection( std::list< WLEModality::Enum > mods );

    /**
     * Hides the selection for view modality, e.g. the module belongs to one modality only.
     *
     * \param enable True to hide the selection.
     */
    void hideViewModalitySelection( bool enable );

    /**
     * Returns the modality which should be processed.
     *
     * \return Modality type to process.
     */
    WLEModality::Enum getComputeModality();

    /**
     * Clears and sets the compute modality selection.
     *
     * \param mods List of modalities to process.
     */
    void setComputeModalitySelection( const std::set< WLEModality::Enum >& modalities );

    /**
     * Hides the selection for compute modality, e.g. the module belongs to one modality only.
     *
     * \param enable True to hide the selection.
     */
    void hideComputeModalitySelection( bool enable );

    /**
     * Gets the time range to visualize. TODO(pieloth): seconds or milliseconds?
     *
     * \return Time range in ???.
     */
    double getTimerange();

    /**
     * Sets the time range in ??? to visualize. TODO(pieloth): seconds or milliseconds?
     *
     * \return Time range in ???.
     */
    void setTimerange( double value );

    /**
     * Deactivate user changes for time range in the GUI.
     *
     * \param enable True to deactivate user changes.
     */
    void setTimerangeInformationOnly( bool enable );

    /**
     * Hides the sensor labels/ channels names in 3D view.
     *
     * \param enable True to hide.
     */
    void hideLabelsOn( bool enable );

    /**
     * Gets the current bound calculator.
     *
     * \return Returns a shared pinter on the bound calculator.
     */
    WLABoundCalculator::SPtr getBoundCalculator();

    /**
     * Sets the current bound calculator.
     *
     * \param calculator The bound calculator.
     */
    void setBoundCalculator( WLABoundCalculator::SPtr calculator );

    /**
     * Uses the current bound calculator to determine the bounds/scale for 3D and 2D views.
     */
    void calcBounds();

    /**
     * Gets the last processed EMM object.
     *
     * \return Returns a shared pointer on a WLEMMeasurement.
     */
    WLEMMeasurement::SPtr getLastEMM();

    /**
     * Sets the last processed EMM object.
     *
     * \param emm The EMM obejct.
     */
    void setLastEMM( WLEMMeasurement::SPtr emm );

    WLEMDDrawable2D::SPtr m_drawable2D; //!< 2D view.

    WLEMDDrawable3D::SPtr m_drawable3D; //!< 3D view.

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

    WPropDouble m_timeRange; //!< The width of the graph in seconds as property.

    WPropDouble m_channelHeight; //!< The distance between two curves of the graph in pixel as property.

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

    WLABoundCalculator::SPtr m_boundCalculator; //!< Calculates the boundaries for 2D and 3D views based on the signal data.

    WLEMMeasurement::SPtr m_lastEmm; //!< The last processed EMM object.
};

#endif  // WLMODULEDRAWABLE_H
