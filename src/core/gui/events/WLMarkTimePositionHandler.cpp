// TODO doc & license

#include <osgGA/GUIActionAdapter>
#include <osgGA/GUIEventAdapter>
#include <osgGA/GUIEventHandler>

#include <core/common/WLogger.h>
#include <core/gui/WCustomWidget.h>

#include "core/gui/drawable/WLEMDDrawable2D.h"
#include "core/gui/drawable/WLEMDDrawable3D.h"

#include "WLMarkTimePositionHandler.h"

namespace LaBP
{

    std::string WLMarkTimePositionHandler::CLASS = "WLMarkTimePosition";

    WLMarkTimePositionHandler::WLMarkTimePositionHandler( LaBP::WLEMDDrawable2D::SPtr initiator,
                    LaBP::WLEMDDrawable3D::SPtr acceptor ) :
                    WLGUIEventHandler( initiator, acceptor )
    {

    }

    WLMarkTimePositionHandler::~WLMarkTimePositionHandler()
    {
    }

    bool WLMarkTimePositionHandler::handle( const osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter& /*aa*/)
    {
        if( ea.getEventType() == osgGA::GUIEventAdapter::PUSH && ea.getButton() == osgGA::GUIEventAdapter::LEFT_MOUSE_BUTTON )
        {
            // TODO (pizarro) m_timePos->get() + ( x - labelsWidth ) * m_timeRange->get() / m_graphWidth->get() instead 0.5 as timePosition
            if( m_initiator->hasData() )
            {
                std::pair< boost::shared_ptr< WLEMMeasurement >, size_t > data =
                                m_initiator->getAs< LaBP::WLEMDDrawable2D >()->getSelectedData( ea.getX() );
                wlog::debug( CLASS ) << "called handle with pixels: " << ea.getX() << " and time: " << data.second;
                boost::dynamic_pointer_cast< LaBP::WLEMDDrawable2D >( m_initiator )->setSelectedPixel( ea.getX() );
                // TODO pizarro change name: samples instead pixels
//                m_initiator->setSelectedSample( data.second );
//                boost::dynamic_pointer_cast< LaBP::WLEMDDrawable2D >( m_initiator )->setSelectedPixel( ea.getX() );
//                m_initiator->draw( data.first );

                m_acceptor->getAs< LaBP::WLEMDDrawable3D >()->setSelectedSample( data.second );
                m_acceptor->draw( data.first );
            }
            return true;
        }
        else
        {
            return false;
        }
    }
} /* namespace LaBP */
