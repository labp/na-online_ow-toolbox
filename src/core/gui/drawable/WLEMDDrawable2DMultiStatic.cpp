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

    void WLEMDDrawable2DMultiStatic::draw( LaBP::WDataSetEMM::SPtr emm )
    {

        m_emm = emm;
        m_dataChanged = true;
        redraw();
    }

    bool WLEMDDrawable2DMultiStatic::hasData() const
    {
        return m_emm.get() && m_emm->hasModality( m_modality );
    }

    std::pair< LaBP::WDataSetEMM::SPtr, size_t > WLEMDDrawable2DMultiStatic::getSelectedData( ValueT pixel ) const
    {
        size_t sample = 0;
        const ValueT x_offset = m_xOffset + 2 * m_labelWidth;

        LaBP::WDataSetEMM::SPtr emm = m_emm;

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
        if( !m_draw )
            return;
        if( !hasData() )
        {
            return;
        }

        LaBP::WDataSetEMM::ConstSPtr emm = m_emm;
        LaBP::WDataSetEMMEMD::ConstSPtr emd = emm->getModality( m_modality );
        osgAddLabels( emd.get() );
        osgAddChannels( emd.get() );
        osgAddMarkLine();

        m_draw = false;
    }

    void WLEMDDrawable2DMultiStatic::osgAddChannels( const LaBP::WDataSetEMMEMD* emd )
    {
        m_rootGroup->removeChild( m_channelGroup );
        m_channelGroup = new osg::Group;

        const ValueT x_pos = m_xOffset + m_labelWidth;
        const ValueT y_pos = m_widget->height() - m_yOffset - ( m_channelHeight / 2 );
        const ValueT width = m_widget->width() - x_pos;
        const ValueT x_scale = width / emd->getSamplesPerChan();
        const ValueT y_scale = ( m_channelHeight / 2 ) / m_amplitudeScale;

        osg::ref_ptr< osg::MatrixTransform > panTransform = new osg::MatrixTransform;
        panTransform->setDataVariance( osg::Object::DYNAMIC );
        // TODO(pieloth): dynamic shift scale ... width / m_timeRange
        // TODO(pieloth): dynamic shift scale ... x_pos * width / m_timeRange,
        panTransform->setMatrix( osg::Matrix::translate( x_pos, y_pos, 0.0 ) );
        const WDataSetEMMEMD::DataT& emdData = emd->getData();
        const size_t channels_begin = 0;
        const size_t channels_count = maxChannels( emd );
        wlog::debug( CLASS ) << "channels_count: " << channels_count;
        osg::ref_ptr< osg::Geode > channelGeode;
        for( size_t channel = channels_begin, channelPos = 0; channelPos < channels_count && channel < emd->getNrChans();
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
