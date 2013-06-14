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

//#include <boost/enable_shared_from_this.hpp>
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

/**
 * Virtual Implementation of WModule to let our modules use a VIEW including just 4 lines! of code
 */

namespace LaBP
{
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
        LaBP::WLModuleOutputDataCollectionable< WLEMMCommand >::SPtr m_output;

        /**
         * Initialize the properties for this module and load the widget
         */
        virtual void properties();

        /**
         * Sets the new data to draw.
         */
        void updateView( WLEMMeasurement::SPtr emm );

        /**
         * Set which elements of the view we want to see: info panels, channels and/or head. Called it after ready()!
         */
        void initView( LaBP::WLEMDDrawable2D::WEGraphType::Enum graphType );

        /**
         * Add commentaries here:
         */
        void resetView();

        /**
         * Initializes the underlying algorithm with the values of WProperties. Called it after ready()!
         */
        virtual void initModule() = 0;

        /**
         * Add commentaries here:
         */
        LaBP::WEModalityType::Enum getViewModality();

        /**
         * Add commentaries here:
         */
        LaBP::WEModalityType::Enum getCalculateModality();

        /**
         * Add commentaries here:
         */
        double getTimerange();

        /**
         * Add commentaries here:
         */
        void setTimerange( double value );

        /**
         * Add commentaries here:
         */
        void setTimerangeInformationOnly( bool enable );

        void hideComputeModalitySelection( bool enable );

        /**
         * Add commentaries here:
         */
        LaBP::WLEMDDrawable2D::SPtr m_drawable2D;

        /**
         * Add commentaries here:
         */
        LaBP::WLEMDDrawable3D::SPtr m_drawable3D;

    private:
        /**
         * Add commentaries here:
         */
        void createColorMap();

        /**
         * Add commentaries here:
         */
        void callbackTimeRangeChanged();

        /**
         * Add commentaries here:
         */
        void callbackChannelHeightChanged();

        /**
         * Add commentaries here:
         */
        void callbackColorChanged();

        /**
         * Add commentaries here:
         */
        void callbackColorModeChanged();

        /**
         * Add commentaries here:
         */
        void callbackViewModalityChanged();

        /**
         * Add commentaries here:
         */
        void callbackMin3DChanged();

        /**
         * Add commentaries here:
         */
        void callbackMax3DChanged();

        /**
         * Add commentaries here:
         */
        void callbackAmplitudeScaleChanged();

        /**
         * Add commentaries here:
         */
        void callbackAutoSensitivityChanged();

        void callbackLabelsChanged();

        LaBP::WLEMDDrawable2D::WEGraphType::Enum m_graphType;

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

        /**
         * A Property used to show some selection to the user.
         */
        WPropSelection m_selectionColor;

        /**
         * A Property used to show some selection to the user.
         */
        WPropSelection m_selectionView;

        /**
         * A Property used to show some selection to the user.
         */
        WPropSelection m_selectionCalculate;

        /**
         * A Property used to show some selection to the user.
         */
        WPropSelection m_selectionColorMode;

        /**
         * Custom widget which is used by this module to display its data.
         */
        LaBP::WLEMDWidget::SPtr m_widget;

        /**
         * Add commentaries here:
         */
        WPropDouble m_amplitudeScale;

        /**
         * Add commentaries here:
         */
        WPropDouble m_minSensitity3D;

        /**
         * Add commentaries here:
         */
        WPropDouble m_maxSensitity3D;

        int m_autoScaleCounter;

        static const int AUTO_SCALE_PACKETS;

        /**
         * Add commentaries here:
         */
        LaBP::WLColorMap::SPtr m_colorMap;
    };
}

#endif  // WLMODULEDRAWABLE_H
