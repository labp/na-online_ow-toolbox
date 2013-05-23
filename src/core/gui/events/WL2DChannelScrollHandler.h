/*
 * TODO(pieloth): license
 * WL2DChannelScrollHandler.h
 *
 *  Created on: 15.05.2013
 *      Author: pieloth
 */

#ifndef WL2DCHANNELSCROLLHANDLER_H_
#define WL2DCHANNELSCROLLHANDLER_H_

#include <string>

#include <boost/shared_ptr.hpp>

#include "core/gui/drawable/WLEMDDrawable2DMultiChannel.h"

#include "WLGUIEventHandler.h"

class WL2DChannelScrollHandler: public LaBP::WLGUIEventHandler
{
public:
    /**
     * Abbreviation for a shared pointer on a instance of this class.
     */
    typedef boost::shared_ptr< WL2DChannelScrollHandler > SPtr;

    /**
     * Abbreviation for a const shared pointer on a instance of this class.
     */
    typedef boost::shared_ptr< const WL2DChannelScrollHandler > ConstSPtr;

    static const std::string CLASS;

    WL2DChannelScrollHandler( LaBP::WLEMDDrawable2DMultiChannel::SPtr initiator,
                    LaBP::WLEMDDrawable2DMultiChannel::SPtr acceptor );
    virtual ~WL2DChannelScrollHandler();

    virtual bool handle( const osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter& aa );
};

#endif /* WL2DCHANNELSCROLLHANDLER_H_ */
