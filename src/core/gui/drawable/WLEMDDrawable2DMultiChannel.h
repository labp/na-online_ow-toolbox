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

#ifndef WLEMDDRAWABLE2DMULTICHANNEL_H_
#define WLEMDDRAWABLE2DMULTICHANNEL_H_

#include <string>
#include <utility>  // for pair<>

#include <boost/shared_ptr.hpp>

#include <osg/Geode>
#include <osg/PositionAttitudeTransform>

#include <core/ui/WUIViewWidget.h>

#include "core/data/emd/WLEMData.h"

#include "WLEMDDrawable2D.h"

class WLEMDDrawable2DMultiChannel: public WLEMDDrawable2D
{
public:
    /**
     * Abbreviation for a shared pointer on a instance of this class.
     */
    typedef boost::shared_ptr< WLEMDDrawable2DMultiChannel > SPtr;

    /**
     * Abbreviation for a const shared pointer on a instance of this class.
     */
    typedef boost::shared_ptr< const WLEMDDrawable2DMultiChannel > ConstSPtr;

    static const std::string CLASS;

    explicit WLEMDDrawable2DMultiChannel( WUIViewWidget::SPtr widget );
    virtual ~WLEMDDrawable2DMultiChannel();

    virtual void draw( WLEMMeasurement::SPtr emm ) = 0;

    virtual bool hasData() const = 0;

    virtual ValueT getChannelHeight() const;
    virtual void setChannelHeight( ValueT spacing );

    virtual std::pair< WLEMMeasurement::SPtr, size_t > getSelectedData( ValueT pixel ) const = 0;

    virtual size_t getChannelBegin() const;
    virtual size_t setChannelBegin( size_t channelNr );

protected:
    virtual bool mustDraw() const;

    virtual void osgNodeCallback( osg::NodeVisitor* nv );

    void virtual osgAddLabels( const WLEMData& emd );

    virtual size_t maxChannels( const WLEMData& emd ) const;
    virtual size_t getChannelBegin( const WLEMData& emd );

    const ValueT m_labelWidth;

    ValueT m_channelHeight;
    bool m_channelHeightChanged;

    size_t m_channelBegin;
    bool m_channelBeginChanged;

    osg::ref_ptr< osg::PositionAttitudeTransform > m_labelsText;
    osg::ref_ptr< osg::Geode > m_labelsBackground;
};

#endif  // WLEMDDRAWABLE2DMULTICHANNEL_H_
