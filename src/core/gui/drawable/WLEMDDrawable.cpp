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

#include <core/graphicsEngine/WGEGroupNode.h>

#include "WLEMDDrawable.h"

const std::string WLEMDDrawable::CLASS = "WLEMDDrawable";

WLEMDDrawable::WLEMDDrawable( WUIViewWidget::SPtr widget ) :
                boost::enable_shared_from_this< WLEMDDrawable >(), m_widget( widget )
{
    m_modality = WLEModality::UNKNOWN;
    m_dataChanged = false;
    m_modalityChanged = false;
    m_rootGroup = new WGEGroupNode;

    m_callbackDelegator = new WLEMDDrawableCallbackDelegator( this );
    m_rootGroup->addUpdateCallback( m_callbackDelegator );
    m_widget->getScene()->insert( m_rootGroup );

    m_draw = false;
}

WLEMDDrawable::~WLEMDDrawable()
{
    m_rootGroup->removeUpdateCallback( m_callbackDelegator );
    m_callbackDelegator->m_drawable = NULL;
    m_callbackDelegator = NULL;
    m_widget->getScene()->remove( m_rootGroup );
    m_rootGroup = NULL;
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

WLEModality::Enum WLEMDDrawable::getModality() const
{
    return m_modality;
}

bool WLEMDDrawable::setModality( WLEModality::Enum modality )
{
    if( modality != m_modality )
    {
        m_modality = modality;
        m_modalityChanged = true;
        return true;
    }
    return false;
}

WUIViewWidget::SPtr WLEMDDrawable::getWidget() const
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
