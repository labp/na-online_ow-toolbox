/*
 * WL2DChannelScrollHandler.cpp
 *
 *  Created on: 15.05.2013
 *      Author: pieloth
 */

#include <core/common/WLogger.h>

#include "WL2DChannelScrollHandler.h"

const std::string WL2DChannelScrollHandler::CLASS = "WL2DChannelScrollHandler";

WL2DChannelScrollHandler::WL2DChannelScrollHandler( LaBP::WLEMDDrawable2DMultiChannel::SPtr initiator,
                LaBP::WLEMDDrawable2DMultiChannel::SPtr acceptor ) :
                LaBP::WLGUIEventHandler( initiator, acceptor )
{
}

WL2DChannelScrollHandler::~WL2DChannelScrollHandler()
{
}

bool WL2DChannelScrollHandler::handle( const osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter& aa )
{
    if( ea.getEventType() != osgGA::GUIEventAdapter::SCROLL )
    {
        return false;
    }

    // Workaround: ea.getScrollingMotion() == osgGA::GUIEventAdapter::SCROLL_DOWN / SCROLL_UP
    const float y_delta = ea.getScrollingDeltaY();
    if( y_delta != 0 )
    {
        LaBP::WLEMDDrawable2DMultiChannel::SPtr drawable = m_acceptor->getAs< LaBP::WLEMDDrawable2DMultiChannel >();
        size_t channelNr = drawable->getChannelBegin();
        if( y_delta < 0 ) // down
        {
            drawable->setChannelBegin( ++channelNr );
            return true;
        }
        if( y_delta > 0 && channelNr > 0 ) // up
        {
            drawable->setChannelBegin( --channelNr );
            return true;
        }
    }

    return false;
}

