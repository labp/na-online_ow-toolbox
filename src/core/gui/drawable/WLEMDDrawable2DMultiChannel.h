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

#ifndef WLEMDDRAWABLE2DMULTICHANNEL_H_
#define WLEMDDRAWABLE2DMULTICHANNEL_H_

#include <string>

#include <boost/shared_ptr.hpp>

#include <osg/Geode>
#include <osg/PositionAttitudeTransform>

#include <core/gui/WCustomWidget.h>

#include "core/dataHandler/WDataSetEMMEMD.h"

#include "WLEMDDrawable2D.h"

namespace LaBP
{
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

        static std::string CLASS;

        explicit WLEMDDrawable2DMultiChannel( WCustomWidget::SPtr widget );
        virtual ~WLEMDDrawable2DMultiChannel();

        virtual void draw( LaBP::WLDataSetEMM::SPtr emm ) = 0;

        virtual bool hasData() const = 0;

        void redraw();

        virtual ValueT getChannelHeight() const;
        virtual void setChannelHeight( ValueT spacing );

        virtual std::pair< LaBP::WLDataSetEMM::SPtr, size_t > getSelectedData( ValueT pixel ) const = 0;

    protected:
        virtual void osgNodeCallback( osg::NodeVisitor* nv ) = 0;

        void virtual osgAddLabels( const LaBP::WDataSetEMMEMD* emd );

        virtual size_t maxChannels( const LaBP::WDataSetEMMEMD* emd ) const;

        const ValueT m_labelWidth;

        ValueT m_channelHeight;
        bool m_channelHeightChanged;

        osg::ref_ptr< osg::PositionAttitudeTransform > m_labelsText;
        osg::ref_ptr< osg::Geode > m_labelsBackground;
    };

} /* namespace LaBP */
#endif  // WLEMDDRAWABLE2DMULTICHANNEL_H_
