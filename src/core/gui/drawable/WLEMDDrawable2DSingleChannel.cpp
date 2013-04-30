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

#include <core/gui/WCustomWidget.h>

#include "core/data/WLDataSetEMM.h"
#include "core/dataHandler/WDataSetEMMEMD.h"

#include "WLEMDDrawable2DSingleChannel.h"

namespace LaBP
{
    WLEMDDrawable2DSingleChannel::WLEMDDrawable2DSingleChannel( WCustomWidget::SPtr widget ) :
                    WLEMDDrawable2D( widget )
    {
    }

    WLEMDDrawable2DSingleChannel::~WLEMDDrawable2DSingleChannel()
    {
    }

    void WLEMDDrawable2DSingleChannel::draw( LaBP::WLDataSetEMM::SPtr emm )
    {

        m_emm = emm;
        m_dataChanged = true;
        redraw();
    }

    bool WLEMDDrawable2DSingleChannel::hasData() const
    {
        return m_emm.get() && m_emm->hasModality( m_modality );
    }

    size_t WLEMDDrawable2DSingleChannel::maxChannels( const LaBP::WDataSetEMMEMD* emd ) const
    {
        return emd->getNrChans();
    }

    void WLEMDDrawable2DSingleChannel::redraw()
    {
        m_draw = true;
    }

    void WLEMDDrawable2DSingleChannel::osgAddChannels( const LaBP::WDataSetEMMEMD* emd )
    {
        m_rootGroup->removeChild( m_channelGroup );
        m_channelGroup = new osg::Group;

        const ValueT x_pos = m_xOffset;
        const ValueT y_pos = m_widget->height() / 2;
        const ValueT width = m_widget->width() - x_pos;
        const ValueT x_scale = width / emd->getSamplesPerChan();
        const ValueT y_scale = ( m_widget->height() / 2 ) / m_amplitudeScale;

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
            scaleSpacingTransform->setMatrix( osg::Matrix::scale( x_scale, y_scale, 1.0 ) );
            scaleSpacingTransform->setDataVariance( osg::Object::DYNAMIC );
            scaleSpacingTransform->addChild( channelGeode );

            panTransform->addChild( scaleSpacingTransform );
        }

        m_channelGroup->addChild( panTransform );
        m_rootGroup->addChild( m_channelGroup );
    }

    std::pair< LaBP::WLDataSetEMM::SPtr, size_t > WLEMDDrawable2DSingleChannel::getSelectedData( ValueT pixel ) const
    {
        size_t sample = 0;

        LaBP::WLDataSetEMM::SPtr emm = m_emm;

        if( pixel > m_xOffset )
        {
            const ValueT width = m_widget->width() - m_xOffset;
            const ValueT relPix = pixel - m_xOffset;
            const size_t samples_total = emm->getModality( m_modality )->getSamplesPerChan();

            sample = relPix * samples_total / width;
        }
        wlog::debug( CLASS ) << "selected sample: " << sample;
        return std::make_pair( emm, sample );
    }

    void WLEMDDrawable2DSingleChannel::osgNodeCallback( osg::NodeVisitor* nv )
    {
        if( !m_draw )
        {
            return;
        }
        if( !hasData() )
        {
            return;
        }

        LaBP::WLDataSetEMM::ConstSPtr emm = m_emm;
        LaBP::WDataSetEMMEMD::ConstSPtr emd = emm->getModality( m_modality );
        osgAddChannels( emd.get() );
        osgAddMarkLine();

        m_draw = false;
    }

} /* namespace LaBP */
