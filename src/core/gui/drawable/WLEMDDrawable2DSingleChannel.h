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

#ifndef WLEMDDRAWABLE2DSINGLECHANNEL_H
#define WLEMDDRAWABLE2DSINGLECHANNEL_H

#include <string>
#include <utility>  // for pair<>

#include <boost/shared_ptr.hpp>

#include <osg/Geode>
#include <osg/ref_ptr>

#include <core/graphicsEngine/WGEGroupNode.h>
#include <core/ui/WUIViewWidget.h>

#include "core/data/emd/WLEMData.h"

#include "WLEMDDrawable2D.h"

/**
 * A butterfly plot 2D view.
 *
 * \author pieloth
 * \ingroup gui
 */
class WLEMDDrawable2DSingleChannel: public WLEMDDrawable2D
{
public:
    /**
     * Abbreviation for a shared pointer on a instance of this class.
     */
    typedef boost::shared_ptr< WLEMDDrawable2DSingleChannel > SPtr;

    /**
     * Abbreviation for a const shared pointer on a instance of this class.
     */
    typedef boost::shared_ptr< const WLEMDDrawable2DSingleChannel > ConstSPtr;

    static const std::string CLASS;

    explicit WLEMDDrawable2DSingleChannel( WUIViewWidget::SPtr widget );
    virtual ~WLEMDDrawable2DSingleChannel();

    /**
     * Invokes a draw with the new data.
     *
     * \param emm data to draw.
     */
    virtual void draw( WLEMMeasurement::SPtr emm );

    /**
     * Checks whether data is available.
     */
    virtual bool hasData() const;

    virtual std::pair< WLEMMeasurement::SPtr, size_t > getSelectedData( ValueT pixel ) const;

protected:
    WLEMMeasurement::SPtr m_emm;

    virtual void osgNodeCallback( osg::NodeVisitor* nv );

    /**
     * Draws and adds a value grid.
     *
     * \param emd Data for scaling.
     */
    void osgAddValueGrid( const WLEMData& emd );

    virtual size_t maxChannels( const WLEMData& emd ) const;

    ValueT m_valueGridHeight;
    ValueT m_valueGridWidth;
    osg::ref_ptr< WGEGroupNode > m_valueGridGroup; //!< Contains the value grid.

private:
    /**
     * Draws and adds the channel data.
     *
     *  \param emd Data to draw.
     */
    void osgAddChannels( const WLEMData& emd );

    /**
     * Draws and adds a marker for events/stimuli.
     *
     * \param events Event data.
     */
    void osgSetTrigger( const WLEMMeasurement::EDataT& events );

    osg::ref_ptr< osg::Geode > m_triggerGeode; //!< Contains the event marker.

    osg::ref_ptr< WLColorArray > m_triggerColors;
};

#endif  // WLEMDDRAWABLE2DSINGLECHANNEL_H
