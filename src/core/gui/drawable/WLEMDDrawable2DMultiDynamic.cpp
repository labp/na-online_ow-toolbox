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

#include <cmath>
#include <list>
#include <string>
#include <utility>  // for pair<>

#include <osg/Group>
#include <osgGA/Export>
#include <osgGA/GUIEventHandler>
#include <osgGA/GUIEventAdapter>
#include <osgGA/GUIActionAdapter>

#include <core/common/WLogger.h>
#include <core/common/WRealtimeTimer.h>
#include <core/common/exceptions/WOutOfBounds.h>
#include <core/graphicsEngine/WGEGroupNode.h>

#include "core/data/emd/WLEMData.h"
#include "core/data/enum/WLEModality.h"
#include "core/exception/WLNoDataException.h"

#include "WLEMDDrawable2DMultiDynamic.h"

const std::string WLEMDDrawable2DMultiDynamic::CLASS = "WLEMDDrawable2DMultiDynamic";

WLEMDDrawable2DMultiDynamic::WLEMDDrawable2DMultiDynamic( WUIViewWidget::SPtr widget ) :
                WLEMDDrawable2DMultiChannel( widget )
{
    m_osgChannelBlocks = new WGEGroupNode;
    m_rootGroup->insert( m_osgChannelBlocks );
    m_blockLength = 0;

    m_animation = new WLAnimationSideScroll( m_osgChannelBlocks );
    m_animation->setTime( 1 );
    m_animation->setXTranslation( m_widget->width() - m_xOffset );
    m_animation->setStartPosition( osg::Vec2d( m_widget->width(), m_widget->height() ) );
}

WLEMDDrawable2DMultiDynamic::~WLEMDDrawable2DMultiDynamic()
{
    delete m_animation;
    if( m_osgChannelBlocks.valid() )
    {
        m_rootGroup->remove( m_osgChannelBlocks );
        m_osgChannelBlocks = NULL;
    }
}

bool WLEMDDrawable2DMultiDynamic::setTimeRange( ValueT timeRange )
{
    bool rc = WLEMDDrawable2D::setTimeRange( timeRange );
    m_animation->setTime( getTimeRange() );
    return rc;
}

void WLEMDDrawable2DMultiDynamic::draw( WLEMMeasurement::SPtr emm )
{
    osg::ref_ptr< WLAnimationSideScroll::EMMNode > emdNode = createEmdNode( emm );
    if( emdNode.valid() )
    {
        m_emmQueue.push( emdNode );
        m_dataChanged = true;
        redraw();
    }
}

bool WLEMDDrawable2DMultiDynamic::hasData() const
{
    return !m_animation->getNodes().empty();
}

void WLEMDDrawable2DMultiDynamic::osgNodeCallback( osg::NodeVisitor* nv )
{
    if( mustDraw() )
    {
        if( m_dataChanged && !m_emmQueue.empty() )
        {
            osg::ref_ptr< WLAnimationSideScroll::EMMNode > emmNode = m_emmQueue.front();
            m_emmQueue.pop();
            WLEMMeasurement::SPtr emm = emmNode->getEmm();
            const WLEModality::Enum modality = m_modality;
            if( emm->hasModality( modality ) )
            {
                const WLEMData* const emd = emm->getModality( modality ).get();
#ifdef DEBUG
                wlog::debug( CLASS ) << "osgNodeCallback() - samplesPerChan: " << emd->getSamplesPerChan();
                wlog::debug( CLASS ) << "osgNodeCallback() - freq: " << emd->getSampFreq();
                wlog::debug( CLASS ) << "osgNodeCallback() - secondsPerChan: " << emd->getLength();
#endif // DEBUG
                osgAddLabels( *emd );
            }

            m_animation->append( emmNode );
        }
        WLEMDDrawable2DMultiChannel::osgNodeCallback( nv );
    }
    m_animation->sweep();
}

osg::ref_ptr< WLAnimationSideScroll::EMMNode > WLEMDDrawable2DMultiDynamic::createEmdNode( WLEMMeasurement::SPtr emm )
{
    const WLEModality::Enum modality = m_modality;
    if( !emm->hasModality( modality ) )
    {
        wlog::error( CLASS ) << "createEmdNode() - Modality not available!";
        return osg::ref_ptr< WLAnimationSideScroll::EMMNode >();
    }

    WLEMData* emd = emm->getModality( modality ).get();
    m_blockLength = emd->getLength().value();

    const ValueT x_pos = m_widget->width();
    const ValueT y_pos = m_widget->height() - m_yOffset;
    m_animation->setStartPosition( osg::Vec2d( x_pos, y_pos ) );
    m_animation->setXTranslation( m_widget->width() - m_xOffset );
    wlog::debug( CLASS ) << "block length pixel: " << getPixelPerBlock( m_blockLength );
    m_animation->setXBlockLength( getPixelPerBlock( m_blockLength ) );

    const ValueT x_scale = getPixelPerBlock( m_blockLength ) / emd->getSamplesPerChan();
//        const ValueT x_scale = 2;
    const ValueT y_scale = ( m_channelHeight / 2 ) / m_amplitudeScale;

    osg::ref_ptr< osg::Group > blockGroup = new osg::Group;
    blockGroup->setDataVariance( osg::Object::DYNAMIC );

    // Every new packed come at the end of the widget!
    const WLEMData::DataT& emdData = emd->getData();
    const size_t channels_emd = emd->getNrChans();
    const size_t channels_count = maxChannels( *emd );
    wlog::debug( CLASS ) << "channels_count: " << channels_count;

    osg::ref_ptr< osg::Geode > channelGeode;
    for( size_t channel = getChannelBegin( *emd ), channelPos = 0; channelPos < channels_count && channel < channels_emd;
                    ++channel, ++channelPos )
    {
        channelGeode = drawChannel( emdData.row( channel ) );
        osg::ref_ptr< osg::MatrixTransform > channelTransform = new osg::MatrixTransform;
        channelTransform->setMatrix(
                        osg::Matrix::scale( x_scale, y_scale, 1.0 )
                                        * osg::Matrix::translate( 0.0, -m_channelHeight * channelPos, 0.0 ) );
        channelTransform->setDataVariance( osg::Object::DYNAMIC );
        channelTransform->addChild( channelGeode );

        blockGroup->addChild( channelTransform );
    }

    osg::ref_ptr< WLAnimationSideScroll::EMMNode > pat = new WLAnimationSideScroll::EMMNode( emm );
    pat->addChild( blockGroup );
    return pat;
}

std::pair< WLEMMeasurement::SPtr, size_t > WLEMDDrawable2DMultiDynamic::getSelectedData( ValueT pixel ) const
{
    m_animation->setPause( true );

    const std::list< osg::ref_ptr< WLAnimationSideScroll::EMMNode > >& nodes = m_animation->getNodes();
    if( nodes.empty() )
    {
        m_animation->setPause( false );
        throw WLNoDataException( "No data to select for pixel!"/* + pixel*/ );
    }

    const ValueT width = m_widget->width();
    if( !( m_xOffset <= pixel && pixel < width ) )
    {
        throw WOutOfBounds( "Pixel out of bounds!" );
    }

    const ValueT pixelPerBlock = getPixelPerBlock( m_blockLength );
    osg::ref_ptr< WLAnimationSideScroll::EMMNode > emmNode;
    std::list< osg::ref_ptr< WLAnimationSideScroll::EMMNode > >::const_iterator it = nodes.begin();
    for( ; it != nodes.end(); ++it )
    {
        const ValueT x = ( *it )->getPosition().x();
        if( x <= pixel && pixel <= x + pixelPerBlock )
        {
            emmNode = *it;
            break;
        }
    }

    if( !emmNode.valid() )
    {
        m_animation->setPause( false );
        throw WLNoDataException( "No data found for pixel!"/* + pixel*/ );
    }

    WLEMMeasurement::SPtr emm = emmNode->getEmm();

    const ValueT xBlock = emmNode->getPosition().x();
    const ValueT xRelative = pixel - xBlock;
    size_t sample = ( xRelative / pixelPerBlock ) * emm->getModality( m_modality )->getSamplesPerChan();

    wlog::debug( CLASS ) << "getSelectedData() - start: " << xBlock << " pixelperBlock: " << pixelPerBlock << " pixel: " << pixel
                    << " sample: " << sample;

    m_animation->setPause( false );
    return std::make_pair( emm, sample );
}

WLEMDDrawable2DMultiDynamic::ValueT WLEMDDrawable2DMultiDynamic::getBlocksOnView( const ValueT& blockLength ) const
{
    return ceil( m_timeRange / blockLength ) + 1;
}

WLEMDDrawable2DMultiDynamic::ValueT WLEMDDrawable2DMultiDynamic::getPixelPerBlock( const ValueT& blockLength ) const
{
    return ( m_widget->width() - m_xOffset ) / ( m_timeRange / blockLength );
}

WLEMDDrawable2DMultiDynamic::ValueT WLEMDDrawable2DMultiDynamic::getPixelPerSeconds() const
{
    return ( m_widget->width() - m_xOffset ) / m_timeRange;
}
