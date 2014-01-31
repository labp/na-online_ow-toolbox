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

#include <cmath> // fabs()
#include <list>
#include <string>

#include <core/common/WLogger.h>

#include "WLAnimationSideScroll.h"

namespace LaBP
{
    const std::string WLAnimationSideScroll::CLASS = "WLAnimationSideScroll";

    WLAnimationSideScroll::WLAnimationSideScroll( osg::ref_ptr< osg::Group > groupPAT ) :
                    m_groupPAT( groupPAT )
    {
        m_maxFps = 1 / 30;
        m_time = 1;
        m_translation = osg::Vec3d( 1.0, 0.0, 0.0 );
        m_startPosition = osg::Vec3d( 1.0, 0.0, 0.0 );
        m_blockLength = osg::Vec3d( 1.0, 0.0, 0.0 );
        m_pause = true;
        m_syncThreshold = 100;
    }

    WLAnimationSideScroll::~WLAnimationSideScroll()
    {
    }

    double WLAnimationSideScroll::getXTranslation() const
    {
        return m_translation.x();
    }

    void WLAnimationSideScroll::setXTranslation( double vec )
    {
        m_translation = osg::Vec3d( -fabs( vec ), 0.0, 0.0 );
    }

    double WLAnimationSideScroll::getXBlockLength() const
    {
        return fabs( m_blockLength.x() );
    }

    void WLAnimationSideScroll::setXBlockLength( double length )
    {
        m_blockLength = osg::Vec3d( fabs( length ), 0.0, 0.0 );
    }

    osg::Vec2d WLAnimationSideScroll::getStartPosition() const
    {
        return osg::Vec2d( m_startPosition.x(), m_startPosition.y() );
    }

    void WLAnimationSideScroll::setStartPosition( const osg::Vec2d& pos )
    {
        m_startPosition = osg::Vec3d( pos.x(), pos.y(), 0.0 );
    }

    double WLAnimationSideScroll::getTime() const
    {
        return m_time;
    }

    void WLAnimationSideScroll::setTime( double sec )
    {
        m_time = fabs( sec );
    }

    int WLAnimationSideScroll::getMaxFps() const
    {
        return 1 / m_maxFps;
    }

    void WLAnimationSideScroll::setMaxFps( int maxFps )
    {
        m_maxFps = 1 / abs( maxFps );
    }

    void WLAnimationSideScroll::append( osg::ref_ptr< EMMNode > pat )
    {
        osg::Vec3d pos;
        if( m_listPAT.empty() )
        {
            m_fpsTimer.reset();
            pos = m_startPosition;
        }
        else
        {
            pos = m_listPAT.back()->getPosition() + m_blockLength;
            if( m_startPosition.x() - pos.x() > m_syncThreshold )
            {
                wlog::warn( CLASS ) << "Side scroll animation synchronized!";
                pos = m_startPosition;
            }
        }
        pat->setPosition( pos );

        m_listPAT.push_back( pat );
        m_groupPAT->addChild( pat );
        m_pause = false;
    }

    void WLAnimationSideScroll::sweep()
    {
        const double time = m_fpsTimer.elapsed();
        if( time < m_maxFps || m_pause )
        {
            return;
        }

        if( !m_listPAT.empty() )
        {
            const double length = fabs( m_translation.x() ); // Sweep x-axis only
            // Calculate distance/translation for elapsed time
            const double partLength = length / ( m_time / time );
            osg::Vec3d translation = m_translation * ( partLength / length );

            // Translate all blocks
            std::list< osg::ref_ptr< EMMNode > >::iterator it;
            for( it = m_listPAT.begin(); it != m_listPAT.end(); ++it )
            {
                const osg::Vec3d& pos = ( *it )->getPosition();
                const osg::Vec3d posNew = pos + translation;
                ( *it )->setPosition( posNew );
            }

            // Cull outdated blocks
            osg::ref_ptr< EMMNode > front = m_listPAT.front();
            if( front->getPosition().x() < -m_blockLength.x() )
            {
                m_listPAT.pop_front();
                m_groupPAT->removeChild( 0, 1 );
            }
        }
        else
        {
            m_pause = true;
        }

        m_fpsTimer.reset();
        return;
    }

    void WLAnimationSideScroll::setPause( bool pause )
    {
        m_pause = pause;
    }

    bool WLAnimationSideScroll::isPaused() const
    {
        return m_pause;
    }

    double WLAnimationSideScroll::getSyncThreshold() const
    {
        return m_syncThreshold;
    }

    void WLAnimationSideScroll::setSyncThreshold( double threshold )
    {
        m_syncThreshold = fabs( threshold );
    }

    const std::list< osg::ref_ptr< WLAnimationSideScroll::EMMNode > >& WLAnimationSideScroll::getNodes() const
    {
        return m_listPAT;
    }

    WLAnimationSideScroll::EMMNode::EMMNode( WLEMMeasurement::SPtr emm ) :
                    m_emm( emm )
    {
    }

    WLAnimationSideScroll::EMMNode::~EMMNode()
    {
    }

    WLEMMeasurement::SPtr WLAnimationSideScroll::EMMNode::getEmm()
    {
        return m_emm;
    }
}
/* namespace LaBP */
