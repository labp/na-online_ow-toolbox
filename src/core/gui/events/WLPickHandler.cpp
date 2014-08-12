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

#include <core/common/WLogger.h>

#include "WLPickHandler.h"

const std::string WLPickHandler::CLASS = "WLMouseHandler";

WLPickHandler::WLPickHandler( WUIViewWidget::SPtr widget ) :
                WPickHandler( widget->getTitle() )
{
}

WLPickHandler::~WLPickHandler()
{
}

boost::signals2::signal1< void, WPickInfo >* WLPickHandler::getPickSignal()
{
    return &m_pickSignal;
}

void WLPickHandler::pick( osgViewer::View* view, const osgGA::GUIEventAdapter& ea )
{
    osgUtil::LineSegmentIntersector::Intersections intersections;
    m_hitResult = WPickInfo();
    float x = ea.getX(); // pixel position in x direction
    float y = ea.getY(); // pixel position in x direction

    WPickInfo pickInfo;

    updatePickInfoModifierKeys( &pickInfo );

    // if we are in another viewer than the main view we just need the pixel position
    /*
     if( m_viewerName != "" && m_viewerName != "Main View" )
     {
     pickInfo = WPickInfo( "", m_viewerName, m_startPick.getPickPosition(), std::make_pair( x, y ),
     m_startPick.getModifierKey(), m_mouseButton, m_startPick.getPickNormal(), m_scrollWheel );
     m_hitResult = pickInfo;

     // if nothing was picked before remember the currently picked.
     m_startPick = pickInfo;

     m_pickSignal( getHitResult() );

     return;
     }
     */

    bool intersetionsExist = view->computeIntersections( x, y, intersections, 0xFFFFFFF0 );

    // if something is picked, get the right thing from the list, because it might be hidden.
    bool startPickIsStillInList = false;
    osgUtil::LineSegmentIntersector::Intersections::iterator hitr;
    if( intersetionsExist )
    {
        assert( intersections.size() );
        hitr = intersections.begin();

        bool ignoreFirst = m_ctrl;

        while( hitr != intersections.end() )
        {
            std::string nodeName = extractSuitableName( hitr );
            // now we skip everything that starts with an underscore if not in paint mode
            if( nodeName[0] == '_' && ( m_paintMode == 0 ) )
            {
                ++hitr;
            }
            // if ctrl is pressed we skip the first thing that gets hit by the pick
            else
                if( ignoreFirst )
                {
                    ++hitr;
                    ignoreFirst = false;
                }
                else
                {
                    break;
                }
        }

        if( hitr == intersections.end() )
        {
            // after everything was ignored nothing pickable remained and we have noting picked before
            // we just stop further processing.
            if( m_startPick.getName() == "" )
            {
                return;
            }
        }

        // if we have a previous pick we search for it in the list
        if( m_startPick.getName() != "" && m_startPick.getName() != "unpick" )
        {
            while( ( hitr != intersections.end() ) && !startPickIsStillInList )
            {
                WPickInfo pickInfoTmp( extractSuitableName( hitr ), m_viewerName, WPosition(), std::make_pair( 0, 0 ),
                                WPickInfo::NONE );
                startPickIsStillInList |= ( pickInfoTmp.getName() == m_startPick.getName() );

                if( !startPickIsStillInList ) // if iteration not finished yet go on in list
                {
                    ++hitr;
                }
            }
        }
    } // end of if( intersetionsExist )
    else
    {
        // if we found no intersection and we have noting picked before
        // we want to return "nothing" in order to provide the pixel coordinates
        // even though we did not hit anything.
        if( m_startPick.getName() == "" )
        {
            pickInfo = WPickInfo( "nothing", m_viewerName, WPosition( 0.0, 0.0, 0.0 ), std::make_pair( x, y ),
                            m_startPick.getModifierKey(), m_mouseButton, WVector3d( 0.0, 0.0, 0.0 ), m_scrollWheel );

            m_hitResult = pickInfo;
            m_pickSignal( getHitResult() );
            return;
        }
    }

    // Set the new pickInfo if the previously picked is still in list or we have a pick in conjunction with previously no pick
    if( startPickIsStillInList || ( intersetionsExist && ( m_startPick.getName() == "unpick" || m_startPick.getName() == "" ) ) )
    {
        // if nothing was picked before, or the previously picked was found: set new pickInfo
        WPosition pickPos;
        pickPos[0] = hitr->getWorldIntersectPoint()[0];
        pickPos[1] = hitr->getWorldIntersectPoint()[1];
        pickPos[2] = hitr->getWorldIntersectPoint()[2];

        WVector3d pickNormal;
        pickNormal[0] = hitr->getWorldIntersectNormal()[0];
        pickNormal[1] = hitr->getWorldIntersectNormal()[1];
        pickNormal[2] = hitr->getWorldIntersectNormal()[2];
        pickInfo = WPickInfo( extractSuitableName( hitr ), m_viewerName, pickPos, std::make_pair( x, y ),
                        pickInfo.getModifierKey(), m_mouseButton, pickNormal, m_scrollWheel );
    }

    // Use the old PickInfo with updated pixel info if we have previously picked something but the old is not in list anymore
    if( !startPickIsStillInList && m_startPick.getName() != "" && m_startPick.getName() != "unpick" )
    {
        pickInfo = WPickInfo( m_startPick.getName(), m_viewerName, m_startPick.getPickPosition(), std::make_pair( x, y ),
                        m_startPick.getModifierKey(), m_mouseButton, m_startPick.getPickNormal(), m_scrollWheel );
    }

    m_hitResult = pickInfo;

    // if nothing was picked before remember the currently picked.
    m_startPick = pickInfo;
    m_inPickMode = true;

    m_pickSignal( getHitResult() );
}

std::string WLPickHandler::extractSuitableName( osgUtil::LineSegmentIntersector::Intersections::iterator hitr )
{
    if( !hitr->nodePath.empty() && !( hitr->nodePath.back()->getName().empty() ) )
    {
        return hitr->nodePath.back()->getName();
    }
    else
        if( hitr->drawable.valid() )
        {
            return hitr->drawable->className();
        }
    assert( 0 && "This should not happen. Tell \"wiebel\" if it does." );
    return ""; // This line will not be reached.
}

void WLPickHandler::updatePickInfoModifierKeys( WPickInfo* pickInfo )
{
    if( m_shift )
    {
        pickInfo->setModifierKey( WPickInfo::SHIFT );
    }

    if( m_ctrl )
    {
        pickInfo->setModifierKey( WPickInfo::STRG );
    }
}
