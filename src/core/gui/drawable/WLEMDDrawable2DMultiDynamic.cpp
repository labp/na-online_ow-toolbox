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

#include <cmath>

#include <osgGA/Export>
#include <osgGA/GUIEventHandler>
#include <osgGA/GUIEventAdapter>
#include <osgGA/GUIActionAdapter>

#include <core/common/WRealtimeTimer.h>

#include "WLEMDDrawable2DMultiDynamic.h"

namespace LaBP
{
    std::string WLEMDDrawable2DMultiDynamic::CLASS = "WLEMDDrawable2DMultiDynamic";

    WLEMDDrawable2DMultiDynamic::WLEMDDrawable2DMultiDynamic( WCustomWidget::SPtr widget ) :
                    WLEMDDrawable2DMultiChannel( widget )
    {
        m_draw = false;
        m_osgChannelBlocks = new osg::MatrixTransform;
        m_rootGroup->addChild( m_osgChannelBlocks );
        m_blockLength = 0;

        m_animation = new WLAnimationSideScroll( m_osgChannelBlocks );
        m_animation->setTime( 1 );
        m_animation->setXTranslation( m_widget->width() - m_labelWidth - m_xOffset );
        m_animation->setStartPosition( osg::Vec2d( m_widget->width(), m_widget->height() ) );
    }

    WLEMDDrawable2DMultiDynamic::~WLEMDDrawable2DMultiDynamic()
    {
        delete m_animation;
    }

    void WLEMDDrawable2DMultiDynamic::draw( LaBP::WLDataSetEMM::SPtr emm )
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

    void WLEMDDrawable2DMultiDynamic::redraw()
    {
        m_draw = true;
    }

    void WLEMDDrawable2DMultiDynamic::osgNodeCallback( osg::NodeVisitor* nv )
    {
        if( m_draw )
        {
            if( m_dataChanged && !m_emmQueue.empty() )
            {
                osg::ref_ptr< WLAnimationSideScroll::EMMNode > emmNode = m_emmQueue.front();
                m_emmQueue.pop();
                WLDataSetEMM::SPtr emm = emmNode->getEmm();
                const WEModalityType::Enum modality = m_modality;
                if( emm->hasModality( modality ) )
                {
                    const WDataSetEMMEMD* const emd = emm->getModality( modality ).get();
                    wlog::debug( CLASS ) << "osgNodeCallback() - samplesPerChan: " << emd->getSamplesPerChan();
                    wlog::debug( CLASS ) << "osgNodeCallback() - freq: " << emd->getSampFreq();
                    wlog::debug( CLASS ) << "osgNodeCallback() - secondsPerChan: " << emd->getLength();
                    osgAddLabels( emd );
                }

                m_animation->append( emmNode );
                m_dataChanged = false;
            }
            osgAddMarkLine();
            m_draw = false;
        }
        m_animation->sweep();
    }

    osg::ref_ptr< WLAnimationSideScroll::EMMNode > WLEMDDrawable2DMultiDynamic::createEmdNode( LaBP::WLDataSetEMM::SPtr emm )
    {
        const WEModalityType::Enum modality = m_modality;
        if( !emm->hasModality( modality ) )
        {
            wlog::error( CLASS ) << "createEmdNode() - Modality not available!";
            return osg::ref_ptr< WLAnimationSideScroll::EMMNode >();
        }

        LaBP::WDataSetEMMEMD* emd = emm->getModality( modality ).get();
        m_blockLength = emd->getLength();

        const ValueT x_pos = m_widget->width();
        const ValueT y_pos = m_widget->height() - m_yOffset - ( m_channelHeight / 2 );
        m_animation->setStartPosition( osg::Vec2d( x_pos, y_pos ) );
        m_animation->setXTranslation( m_widget->width() - m_labelWidth - m_xOffset );
        wlog::debug( CLASS ) << "block length pixel: " << getPixelPerBlock( m_blockLength );
        m_animation->setXBlockLength( getPixelPerBlock( m_blockLength ) );

        const ValueT x_scale = getPixelPerBlock( m_blockLength ) / emd->getSamplesPerChan();
//        const ValueT x_scale = 2;
        const ValueT y_scale = ( m_channelHeight / 2 ) / m_amplitudeScale;

        osg::ref_ptr< osg::Group > blockGroup = new osg::Group;
        blockGroup->setDataVariance( osg::Object::DYNAMIC );

        // Every new packed come at the end of the widget!
        const WDataSetEMMEMD::DataT& emdData = emd->getData();
        const size_t channels_begin = 0;
        const size_t channels_count = maxChannels( emd );
        wlog::debug( CLASS ) << "channels_count: " << channels_count;

        osg::ref_ptr< osg::Geode > channelGeode;
        for( size_t channel = channels_begin, channelPos = 0; channelPos < channels_count && channel < emd->getNrChans();
                        ++channel, ++channelPos )
        {
            channelGeode = drawChannel( emdData[channel] );
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

    std::pair< LaBP::WLDataSetEMM::SPtr, size_t > WLEMDDrawable2DMultiDynamic::getSelectedData( ValueT pixel ) const
    {
        m_animation->setPause( true );

        const std::list< osg::ref_ptr< WLAnimationSideScroll::EMMNode > >& nodes = m_animation->getNodes();
        if( nodes.empty() )
        {
            wlog::error( CLASS ) << "getSelectedData() - No data to select!";
            m_animation->setPause( false );
            return std::make_pair( LaBP::WLDataSetEMM::SPtr(), 0 );
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
            wlog::error( CLASS ) << "getSelectedData() - No data found for pixel: " << pixel;
            m_animation->setPause( false );
            return std::make_pair( LaBP::WLDataSetEMM::SPtr(), 0 );
        }

        LaBP::WLDataSetEMM::SPtr emm = emmNode->getEmm();

        const ValueT xBlock = emmNode->getPosition().x();
        const ValueT xRelative = pixel - xBlock;
        size_t sample = ( xRelative / pixelPerBlock ) * emm->getModality( m_modality )->getSamplesPerChan();

        wlog::debug( CLASS ) << "getSelectedData() - start: " << xBlock << " pixelperBlock: " << pixelPerBlock << " pixel: "
                        << pixel << " sample: " << sample;

        m_animation->setPause( false );
        return std::make_pair( emm, sample );
    }

    WLEMDDrawable2DMultiDynamic::ValueT WLEMDDrawable2DMultiDynamic::getBlocksOnView( const ValueT& blockLength ) const
    {
        return ceil( m_timeRange / blockLength ) + 1;
    }

    WLEMDDrawable2DMultiDynamic::ValueT WLEMDDrawable2DMultiDynamic::getPixelPerBlock( const ValueT& blockLength ) const
    {
        return ( m_widget->width() - m_xOffset - m_labelWidth ) / ( m_timeRange / blockLength );
    }

    WLEMDDrawable2DMultiDynamic::ValueT WLEMDDrawable2DMultiDynamic::getPixelPerSeconds() const
    {
        return ( m_widget->width() - m_xOffset - m_labelWidth ) / m_timeRange;
    }

/* namespace LaBP */
}
