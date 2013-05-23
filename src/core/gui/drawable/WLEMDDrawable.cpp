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

#include <string>

#include <osg/Group>
#include <osg/Node>
#include <osg/NodeCallback>
#include <osg/NodeVisitor>

#include <core/gui/WCustomWidget.h>
#include <core/graphicsEngine/WGEGroupNode.h> // Error: forward declaration
#include "core/data/WLEMMeasurement.h"
#include "core/data/WLEMMEnumTypes.h"

#include "WLEMDDrawable.h"

namespace LaBP
{
    const std::string WLEMDDrawable::CLASS = "WLEMDDrawable";

    WLEMDDrawable::WLEMDDrawable( WCustomWidget::SPtr widget ) :
                    m_widget( widget )
    {
        m_modality = LaBP::WEModalityType::EEG;
        m_dataChanged = false;
        m_modalityChanged = false;
        m_rootGroup = new osg::Group;

        m_callbackDelegator = new WLEMDDrawableCallbackDelegator( this );
        m_rootGroup->addUpdateCallback( m_callbackDelegator );
        m_widget->getScene()->insert( m_rootGroup );

        m_draw = false;
    }

    WLEMDDrawable::~WLEMDDrawable()
    {
        m_rootGroup->removeUpdateCallback( m_callbackDelegator );
        m_widget->getScene()->remove( m_rootGroup );
    }

    void WLEMDDrawable::redraw()
    {
        m_draw = true;
    }

    bool WLEMDDrawable::mustDraw() const
    {
        return m_draw || m_dataChanged || m_modalityChanged;
    }

    void WLEMDDrawable::resetDrawFlags()
    {
        m_draw = false;
        m_modalityChanged = false;
        m_dataChanged = false;
    }

    LaBP::WEModalityType::Enum WLEMDDrawable::getModality() const
    {
        return m_modality;
    }

    bool WLEMDDrawable::setModality( WEModalityType::Enum modality )
    {
        if( modality != m_modality )
        {
            m_modality = modality;
            m_modalityChanged = true;
            return true;
        }
        return false;
    }

    WCustomWidget::SPtr WLEMDDrawable::getWidget() const
    {
        return m_widget;
    }

    WLEMDDrawable::WLEMDDrawableCallbackDelegator::WLEMDDrawableCallbackDelegator( WLEMDDrawable* drawable ) :
                    m_drawable( drawable )
    {
    }

    WLEMDDrawable::WLEMDDrawableCallbackDelegator::~WLEMDDrawableCallbackDelegator()
    {
    }

    void WLEMDDrawable::WLEMDDrawableCallbackDelegator::operator()( osg::Node* node, osg::NodeVisitor* nv )
    {
        m_drawable->osgNodeCallback( nv );
    }

} /* namespace LaBP */
