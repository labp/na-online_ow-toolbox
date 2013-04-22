// TODO doc & license

#ifndef WLMARKTIMEPOSITIONHANDLER_H_
#define WLMARKTIMEPOSITIONHANDLER_H_

#include <boost/shared_ptr.hpp>
#include <osgGA/GUIActionAdapter>
#include <osgGA/GUIEventAdapter>
#include <osgGA/GUIEventHandler>

#include "core/gui/drawable/WLEMDDrawable2D.h"
#include "core/gui/drawable/WLEMDDrawable3D.h"

#include "WLGUIEventHandler.h"

namespace LaBP
{

    class WLMarkTimePositionHandler: public WLGUIEventHandler
    {
    public:
        /**
         * Abbreviation for a shared pointer on a instance of this class.
         */
        typedef boost::shared_ptr< WLMarkTimePositionHandler > SPtr;

        /**
         * Abbreviation for a const shared pointer on a instance of this class.
         */
        typedef boost::shared_ptr< const WLMarkTimePositionHandler > ConstSPtr;

        static std::string CLASS;

        WLMarkTimePositionHandler( LaBP::WLEMDDrawable2D::SPtr initiator, LaBP::WLEMDDrawable3D::SPtr acceptor );
        virtual ~WLMarkTimePositionHandler();
        virtual bool handle( const osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter& aa );

    };

} /* namespace LaBP */
#endif  // WLMARKTIMEPOSITIONHANDLER_H_
