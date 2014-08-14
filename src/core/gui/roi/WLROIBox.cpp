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

#include <typeinfo>

#include <osg/LineWidth>
#include <osg/LightModel>

#include <core/common/WLogger.h>
#include <core/graphicsEngine/WGEGeodeUtils.h>
#include <core/graphicsEngine/WGEUtils.h>
#include <core/graphicsEngine/WGraphicsEngine.h>

#include "WLROIBox.h"

const std::string WLROIBox::CLASS = "WLROIBox";

size_t WLROIBox::maxBoxId = 0;

WLROIBox::WLROIBox( WPosition minPos, WPosition maxPos, WUIViewWidget::SPtr widget ) :
                WLROI( widget ), m_boxId( maxBoxId++ ), m_minPosInit( minPos ), m_maxPosInit( maxPos ), m_isPicked( false ), m_pickNormal(
                                WVector3d() ), m_oldPixelPosition( WVector2d::zero() ), m_oldScrollWheel( 0 )
{
    initProperties();

    std::stringstream ss;
    ss << "ROIBox" << m_boxId;
    setName( ss.str() );

    osg::StateSet* state = getOrCreateStateSet();
    state->setRenderingHint( osg::StateSet::TRANSPARENT_BIN );

    osg::LineWidth* linewidth = new osg::LineWidth();
    linewidth->setWidth( 2.0f );
    state->setAttributeAndModes( linewidth, osg::StateAttribute::ON );

    osg::ref_ptr< osg::LightModel > lightModel = new osg::LightModel();
    lightModel->setTwoSided( true );
    state->setMode( GL_BLEND, osg::StateAttribute::ON );

    // add a simple default lighting shader
    m_lightShader = new WGEShader( "WGELighting" );

    m_not->set( false );

    assert( WGraphicsEngine::getGraphicsEngine() );
    WGraphicsEngine::getGraphicsEngine()->getScene()->addChild( this );

    setUserData( this );
    setUpdateCallback( osg::ref_ptr< ROIBoxNodeCallback >( new ROIBoxNodeCallback ) );

    m_lightShader->apply( this );

    m_needVertexUpdate = true;
    setDirty();
}

WLROIBox::~WLROIBox()
{
}

void WLROIBox::initProperties()
{
    wlog::debug( CLASS ) << "properties()";

    m_propGrp = m_properties->addPropertyGroup( "ROI Box", "Properties of this ROI Box" );
    m_minPos = m_propGrp->addProperty( "Min Position",
                    "When a box is described by its diagonal, this is the lower, left, front corner of it.", m_minPosInit,
                    boost::bind( &WLROIBox::boxPropertiesChanged, this, _1 ) );
    m_minPos->setHidden( true );
    m_maxPos = m_propGrp->addProperty( "Max Position",
                    "When a box is described by its diagonal, this is the upper, right, back corner of it.", m_maxPosInit,
                    boost::bind( &WLROIBox::boxPropertiesChanged, this, _1 ) );
    m_maxPos->setHidden( true );

    m_width = m_propGrp->addProperty( "Width", "The box width.", m_maxPos->get().x() - m_minPos->get().x(),
                    boost::bind( &WLROIBox::boxDimensionsChanged, this, _1 ) );
    m_width->setMin( 1 );
    m_width->setMax( 100 );

    m_height = m_propGrp->addProperty( "Height", "The box height.", m_maxPos->get().y() - m_minPos->get().y(),
                    boost::bind( &WLROIBox::boxDimensionsChanged, this, _1 ) );
    m_height->setMin( 1 );
    m_height->setMax( 100 );

    m_depth = m_propGrp->addProperty( "Depth", "The box depth.", m_maxPos->get().z() - m_minPos->get().z(),
                    boost::bind( &WLROIBox::boxDimensionsChanged, this, _1 ) );
    m_depth->setMin( 1 );
    m_depth->setMax( 100 );

    m_dimensions = getMaxPos() - getMinPos();
}

WPosition WLROIBox::getMinPos() const
{
    return m_minPos->get();
}

WPosition WLROIBox::getMaxPos() const
{
    return m_maxPos->get();
}

void WLROIBox::updateGFX()
{
    boost::unique_lock< boost::shared_mutex > lock;
    lock = boost::unique_lock< boost::shared_mutex >( m_updateLock );

    //if( m_pickInfo.getViewerName() == m_viewer->getName() && m_picked )
    if( m_pickInfo.getName() == getName() )
    {
        WVector2d newPixelPos( m_pickInfo.getPickPixel() );
        if( m_isPicked )
        {
            osg::Vec3 in( newPixelPos.x(), newPixelPos.y(), 0.0 );
            osg::Vec3 world = wge::unprojectFromScreen( in, m_viewer->getCamera() );

            // we want the vector pointing into the screen in world coordinates
            // NOTE: set w = 0 to use it as vector and ignore translation
            osg::Vec4 toDepth = wge::unprojectFromScreen( osg::Vec4( 0.0, 0.0, 1.0, 0.0 ), m_viewer->getCamera() );
            toDepth.normalize();
            WPosition toDepthWorld( toDepth[0], toDepth[1], toDepth[2] );

            float depthMove = m_pickInfo.getScrollWheel() - m_oldScrollWheel;

            WPosition newPixelWorldPos( world[0], world[1], world[2] );
            WPosition oldPixelWorldPos;
            if( m_oldPixelPosition.x() == 0 && m_oldPixelPosition.y() == 0 )
            {
                oldPixelWorldPos = newPixelWorldPos;
            }
            else
            {
                osg::Vec3 in( m_oldPixelPosition.x(), m_oldPixelPosition.y(), 0.0 );
                osg::Vec3 world = wge::unprojectFromScreen( in, m_viewer->getCamera() );
                oldPixelWorldPos = WPosition( world[0], world[1], world[2] );
            }

            WVector3d moveVec = newPixelWorldPos - oldPixelWorldPos;

            // resize Box
            // todo(maschke): change the resizing feature for width-height-depth
            /*
            if( m_pickInfo.getModifierKey() == WPickInfo::SHIFT )
            {
                if( m_pickNormal[0] <= 0 && m_pickNormal[1] <= 0 && m_pickNormal[2] <= 0 )
                {
                    m_maxPos->set( m_maxPos->get() + ( m_pickNormal * dot( moveVec, m_pickNormal ) ) );
                }
                if( m_pickNormal[0] >= 0 && m_pickNormal[1] >= 0 && m_pickNormal[2] >= 0 )
                {
                    m_minPos->set( m_minPos->get() + ( m_pickNormal * dot( moveVec, m_pickNormal ) ) );
                }
                // NOTE: this sets m_needVertexUpdate
            }
            */

            // move Box
            if( m_pickInfo.getModifierKey() == WPickInfo::NONE )
            {
                m_minPos->set( m_minPos->get() + moveVec + ( 2.0 * toDepthWorld * depthMove ) );
                m_maxPos->set( m_maxPos->get() + moveVec + ( 2.0 * toDepthWorld * depthMove ) );
                // NOTE: this sets m_needVertexUpdate
            }
        }
        else
        {
            m_pickNormal = m_pickInfo.getPickNormal();
            // color for moving box
            if( m_pickInfo.getModifierKey() == WPickInfo::NONE )
            {
                if( m_not->get() )
                {
                    updateColor( m_notColor );
                }
                else
                {
                    updateColor( m_color );
                }
            }
            if( m_pickInfo.getModifierKey() == WPickInfo::SHIFT )
            {
                updateColor( osg::Vec4( 0.0f, 1.0f, 0.0f, 0.4f ) );
            }

            m_oldScrollWheel = m_pickInfo.getScrollWheel();
        }
        m_oldPixelPosition = newPixelPos;
        setDirty();
        m_isPicked = true;
        m_oldScrollWheel = m_pickInfo.getScrollWheel();
    }
    if( m_isPicked && m_pickInfo.getName() == "unpick" )
    {
        wlog::debug( CLASS ) << "unpick.";

        // Perform all actions necessary for finishing a pick
        if( m_not->get() )
        {
            updateColor( m_notColor );
        }
        else
        {
            updateColor( m_color );
        }
        m_pickNormal = WVector3d();
        m_isPicked = false;
    }

    if( m_needVertexUpdate )
    {
        removeDrawable( m_surfaceGeometry );

        WPosition pos = getMinPos();
        WPosition size = getMaxPos() - getMinPos();

        // create a new geometry
        m_surfaceGeometry = wge::createCube( pos, size, WColor( 1.0, 1.0, 1.0, 1.0 ) );

        // create nice outline
        m_surfaceGeometry->addPrimitiveSet( new osg::DrawArrays( osg::PrimitiveSet::LINE_LOOP, 0, 4 ) );
        m_surfaceGeometry->addPrimitiveSet( new osg::DrawArrays( osg::PrimitiveSet::LINE_LOOP, 4, 4 ) );
        m_surfaceGeometry->addPrimitiveSet( new osg::DrawArrays( osg::PrimitiveSet::LINE_LOOP, 8, 4 ) );
        m_surfaceGeometry->addPrimitiveSet( new osg::DrawArrays( osg::PrimitiveSet::LINE_LOOP, 12, 4 ) );
        m_surfaceGeometry->addPrimitiveSet( new osg::DrawArrays( osg::PrimitiveSet::LINE_LOOP, 16, 4 ) );
        m_surfaceGeometry->addPrimitiveSet( new osg::DrawArrays( osg::PrimitiveSet::LINE_LOOP, 20, 4 ) );

        // name it and add to geode
        m_surfaceGeometry->setDataVariance( osg::Object::DYNAMIC );
        m_surfaceGeometry->setName( getName() );

        addDrawable( m_surfaceGeometry );

        // NOTE: as we set the roi dirty, we ensure the color gets set properly in the next if-statement.
        setDirty();
        m_needVertexUpdate = false;
    }

    if( m_dirty->get() )
    {
        if( m_not->get() )
        {
            updateColor( m_notColor );
        }
        else
        {
            updateColor( m_color );
        }
    }

    lock.unlock();
}

void WLROIBox::boxPropertiesChanged( boost::shared_ptr< WPropertyBase > property )
{
    m_needVertexUpdate = true;
}

void WLROIBox::boxDimensionsChanged( boost::shared_ptr< WPropertyBase > property )
{
    m_needVertexUpdate = true;

    WVector3d delta = ( WVector3d( m_width->get(), m_height->get(), m_depth->get() ) - m_dimensions ) / 2;

    m_maxPos->set( m_maxPos->get() + delta );
    m_minPos->set( m_minPos->get() - delta );

    m_dimensions = getMaxPos() - getMinPos();
}
