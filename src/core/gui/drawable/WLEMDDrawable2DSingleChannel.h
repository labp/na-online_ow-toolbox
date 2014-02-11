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

#ifndef WLEMDDRAWABLE2DSINGLECHANNEL_H
#define WLEMDDRAWABLE2DSINGLECHANNEL_H

#include <string>
#include <utility>  // for pair<>

#include <boost/shared_ptr.hpp>

#include <osg/Geode>
#include <osg/ref_ptr>

#include <core/graphicsEngine/WGEGroupNode.h>
#include <core/ui/WCustomWidget.h>

#include "core/data/emd/WLEMData.h"

#include "WLEMDDrawable2D.h"

namespace LaBP
{
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

        explicit WLEMDDrawable2DSingleChannel( WCustomWidget::SPtr widget );
        virtual ~WLEMDDrawable2DSingleChannel();

        /**
         * Invokes a draw with the new data.
         *
         * @param emm data to draw.
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

        void osgAddValueGrid( const WLEMData& emd );

        virtual size_t maxChannels( const WLEMData& emd ) const;

        ValueT m_valueGridHeight;
        ValueT m_valueGridWidth;
        osg::ref_ptr< WGEGroupNode > m_valueGridGroup;

    private:
        void osgAddChannels( const WLEMData& emd );

        void osgSetTrigger( const WLEMMeasurement::EDataT& events );

        osg::ref_ptr< osg::Geode > m_triggerGeode;

        osg::ref_ptr< WLColorArray > m_triggerColors;
    };

} /* namespace LaBP */
#endif  // WLEMDDRAWABLE2DSINGLECHANNEL_H
