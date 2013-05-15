/*
 * WLEMDDrawable2DMultiStatic.cpp
 *
 *  Created on: 16.04.2013
 *      Author: pieloth
 */

#include <string>

#include "WLEMDDrawable2DMultiStatic.h"

namespace LaBP
{
    const std::string WLEMDDrawable2DMultiStatic::CLASS = "WLEMDDrawable2DMultiStatic";

    WLEMDDrawable2DMultiStatic::WLEMDDrawable2DMultiStatic( WCustomWidget::SPtr widget ) :
                    WLEMDDrawable2DMultiChannel( widget )
    {
    }

    WLEMDDrawable2DMultiStatic::~WLEMDDrawable2DMultiStatic()
    {
    }

    void WLEMDDrawable2DMultiStatic::draw( LaBP::WLDataSetEMM::SPtr emm )
    {

        m_emm = emm;
        m_dataChanged = true;
        redraw();
    }

    bool WLEMDDrawable2DMultiStatic::hasData() const
    {
        return m_emm.get() && m_emm->hasModality( m_modality );
    }

    std::pair< LaBP::WLDataSetEMM::SPtr, size_t > WLEMDDrawable2DMultiStatic::getSelectedData( ValueT pixel ) const
    {
        size_t sample = 0;
        const ValueT x_offset = m_xOffset;

        LaBP::WLDataSetEMM::SPtr emm = m_emm;

        if( pixel > x_offset )
        {
            const ValueT width = m_widget->width() - x_offset;
            const ValueT relPix = pixel - x_offset;
            const size_t samples_total = emm->getModality( m_modality )->getSamplesPerChan();

            sample = relPix * samples_total / width;
        }
        wlog::debug( CLASS ) << "selected sample: " << sample;
        return std::make_pair( emm, sample );
    }

    void WLEMDDrawable2DMultiStatic::osgNodeCallback( osg::NodeVisitor* nv )
    {
        if( !mustDraw() )
            return;
        if( !hasData() )
        {
            return;
        }

        LaBP::WLDataSetEMM::ConstSPtr emm = m_emm;
        LaBP::WLEMD::ConstSPtr emd = emm->getModality( m_modality );
        osgAddLabels( emd.get() );
        osgAddChannels( emd.get() );

        WLEMDDrawable2DMultiChannel::osgNodeCallback( nv );
    }

    void WLEMDDrawable2DMultiStatic::osgAddChannels( const LaBP::WLEMD* emd )
    {
        m_rootGroup->removeChild( m_channelGroup );
        m_channelGroup = new osg::Group;

        const ValueT x_pos = m_xOffset;
        const ValueT y_pos = m_widget->height() - m_yOffset;
        const ValueT width = m_widget->width() - x_pos;
        const ValueT x_scale = width / emd->getSamplesPerChan();
        const ValueT y_scale = ( m_channelHeight / 2 ) / m_amplitudeScale;

        osg::ref_ptr< osg::MatrixTransform > panTransform = new osg::MatrixTransform;
        panTransform->setDataVariance( osg::Object::DYNAMIC );
        // TODO(pieloth): dynamic shift scale ... width / m_timeRange
        // TODO(pieloth): dynamic shift scale ... x_pos * width / m_timeRange,
        panTransform->setMatrix( osg::Matrix::translate( x_pos, y_pos, 0.0 ) );
        const WLEMD::DataT& emdData = emd->getData();
        const size_t channels_emd = emd->getNrChans();
        const size_t channels_count = maxChannels( emd );
        osg::ref_ptr< osg::Geode > channelGeode;
        for( size_t channel = getChannelBegin( emd ), channelPos = 0; channelPos < channels_count && channel < channels_emd;
                        ++channel, ++channelPos )
        {
            channelGeode = drawChannel( emdData[channel] );
            osg::ref_ptr< osg::MatrixTransform > scaleSpacingTransform = new osg::MatrixTransform;
            scaleSpacingTransform->setMatrix(
                            osg::Matrix::scale( x_scale, y_scale, 1.0 )
                                            * osg::Matrix::translate( 0.0, -m_channelHeight * channelPos, 0.0 ) );
            scaleSpacingTransform->setDataVariance( osg::Object::DYNAMIC );
            scaleSpacingTransform->addChild( channelGeode );

            panTransform->addChild( scaleSpacingTransform );

        }

        m_channelGroup->addChild( panTransform );
        m_rootGroup->addChild( m_channelGroup );
    }

} /* namespace LaBP */
