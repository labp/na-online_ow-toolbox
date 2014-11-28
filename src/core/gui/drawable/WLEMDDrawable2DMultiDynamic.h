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

#ifndef WLEMDDRAWABLE2DMULTIDYNAMIC_H_
#define WLEMDDRAWABLE2DMULTIDYNAMIC_H_

#include <queue>
#include <string>
#include <utility>  // pair<>

#include <core/graphicsEngine/WGEGroupNode.h>

#include "core/gui/drawable/WLAnimationSideScroll.h"
#include "WLEMDDrawable2DMultiChannel.h"

class WLEMDAnimationFetchCallback;

/**
 * \author pieloth
 * \ingroup gui
 */
class WLEMDDrawable2DMultiDynamic: public WLEMDDrawable2DMultiChannel
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

    static const std::string CLASS;

    explicit WLEMDDrawable2DMultiDynamic( WUIViewWidget::SPtr widget );
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
     * \param blockLength Length of 1 block in seconds
     * \return count
     */
    inline ValueT getBlocksOnView( const ValueT& blockLength ) const;

    /**
     * Returns the pixel per block in relation to the scale.
     *
     * \param blockLength
     * \return px/block
     */
    inline ValueT getPixelPerBlock( const ValueT& blockLength ) const;

    /**
     * Returns the pixel per second in relation to the scale.
     *
     * \return px/second
     */
    inline ValueT getPixelPerSeconds() const;

    osg::ref_ptr< WGEGroupNode > m_osgChannelBlocks;
};

#endif  // WLEMDDRAWABLE2DMULTIDYNAMIC_H_
