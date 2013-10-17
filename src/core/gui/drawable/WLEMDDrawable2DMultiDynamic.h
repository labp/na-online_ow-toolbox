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

#ifndef WLEMDDRAWABLE2DMULTICHANNELDYNAMIC_H_
#define WLEMDDRAWABLE2DMULTICHANNELDYNAMIC_H_

#include <queue>

#include <osg/MatrixTransform>

#include "core/gui/drawable/WLAnimationSideScroll.h"
#include "WLEMDDrawable2DMultiChannel.h"

namespace LaBP
{
    class WLEMDAnimationFetchCallback;

    class WLEMDDrawable2DMultiDynamic: public LaBP::WLEMDDrawable2DMultiChannel
    {
    public:
        /**
         * Abbreviation for a shared pointer on a instance of this class.
         */
        typedef boost::shared_ptr< WLEMDDrawable2DMultiDynamic > SPtr;

        /**
         * Abbreviation for a const shared pointer on a instance of this class.
         */
        typedef boost::shared_ptr< const WLEMDDrawable2DMultiDynamic > ConstSPtr;

        static const  std::string CLASS;

        WLEMDDrawable2DMultiDynamic( WCustomWidget::SPtr widget );
        virtual ~WLEMDDrawable2DMultiDynamic();

        virtual bool setTimeRange( ValueT timeRange );

        virtual void draw( WLEMMeasurement::SPtr emm );

        virtual bool hasData() const;

        virtual std::pair< WLEMMeasurement::SPtr, size_t > getSelectedData( ValueT pixel ) const;

    protected:
        virtual void osgNodeCallback( osg::NodeVisitor* nv );

        virtual osg::ref_ptr< WLAnimationSideScroll::EMMNode > createEmdNode( WLEMMeasurement::SPtr emd );

    private:
        /**
         * Buffer to store EMM and ensure producer-consumer data access.
         */
        std::queue< osg::ref_ptr< WLAnimationSideScroll::EMMNode > > m_emmQueue;

        WLAnimationSideScroll* m_animation;

        /**
         * Block length in seconds.
         */
        ValueT m_blockLength;

        /**
         * How many blocks must be shwon on view.
         * @param blockLength Length of 1 block in seconds
         * @return count
         */
        inline ValueT getBlocksOnView( const ValueT& blockLength ) const;

        /**
         * Returns the pixel per block in relation to the scale.
         *
         * @param blockLength
         * @return px/block
         */
        inline ValueT getPixelPerBlock( const ValueT& blockLength ) const;

        /**
         * Returns the pixel per second in relation to the scale.
         *
         * @return px/second
         */
        inline ValueT getPixelPerSeconds() const;

        osg::ref_ptr< osg::MatrixTransform > m_osgChannelBlocks;
    };

} /* namespace LaBP */
#endif  // WLEMDDRAWABLE2DMULTICHANNELDYNAMIC_H_
