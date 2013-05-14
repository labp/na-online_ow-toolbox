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

#ifndef WLEMDDRAWABLE2DOVERLAYCHANNEL_H_
#define WLEMDDRAWABLE2DOVERLAYCHANNEL_H_

#include <boost/shared_ptr.hpp>

#include <core/gui/WCustomWidget.h>

#include "core/data/emd/WLEMD.h"

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

        explicit WLEMDDrawable2DSingleChannel( WCustomWidget::SPtr widget );
        virtual ~WLEMDDrawable2DSingleChannel();

        /**
         * Invokes a draw with the new data.
         *
         * @param emm data to draw.
         */
        virtual void draw( LaBP::WLDataSetEMM::SPtr emm );

        /**
         * Checks whether data is available.
         */
        virtual bool hasData() const;

        void redraw();

        virtual std::pair< LaBP::WLDataSetEMM::SPtr, size_t > getSelectedData( ValueT pixel ) const;

    protected:
        LaBP::WLDataSetEMM::SPtr m_emm;

        virtual void osgNodeCallback( osg::NodeVisitor* nv );

        void osgAddValueGrid( const LaBP::WLEMD* emd );

        ValueT m_valueGridHeight;
        ValueT m_valueGridWidth;
        osg::ref_ptr< osg::Group > m_valueGridGroup;

        virtual size_t maxChannels( const LaBP::WLEMD* emd ) const;

    private:
        void osgAddChannels( const LaBP::WLEMD* emd );
    };

} /* namespace LaBP */
#endif  // WLEMDDRAWABLE2DOVERLAYCHANNEL_H_
