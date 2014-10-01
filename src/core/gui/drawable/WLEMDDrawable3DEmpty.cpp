//---------------------------------------------------------------------------
//
// Project: NA-Online ( http://www.labp.htwk-leipzig.de )
//
// Copyright 2010 Laboratory for Biosignal Processing, HTWK Leipzig, Germany
//
// This file is part of NA-Online.
//
// NA-Online is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// NA-Online is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with NA-Online. If not, see <http://www.gnu.org/licenses/>.
//
//---------------------------------------------------------------------------

#include <string>

#include <osgText/Text>

#include <core/ui/WUIViewWidget.h>

#include "core/util/WLDefines.h"
#include "WLEMDDrawable3DEmpty.h"

WLEMDDrawable3DEmpty::WLEMDDrawable3DEmpty( WUIViewWidget::SPtr widget ) :
                WLEMDDrawable3D( widget )
{
}

WLEMDDrawable3DEmpty::~WLEMDDrawable3DEmpty()
{
    if( m_textGeode.valid() )
    {
        m_rootGroup->remove( m_textGeode );
        m_textGeode = NULL;
    }
}

void WLEMDDrawable3DEmpty::draw( WLEMMeasurement::SPtr emm )
{
    WL_UNUSED( emm );
    redraw();
}

void WLEMDDrawable3DEmpty::osgNodeCallback( osg::NodeVisitor* nv )
{
    if( !mustDraw() )
    {
        return;
    }

    m_rootGroup->removeChild( m_textGeode );
    m_textGeode = new osg::Geode;

    const ValueT char_size = 16.0;
    const std::string text = "No 3D view implemented for this modality!";
    const ValueT x_pos = -m_widget->width() / 8;
    const ValueT y_pos = 0;
    const ValueT z_pos = 0;
    const osg::Vec3 text_pos( x_pos, y_pos, z_pos );
    const osg::Vec4 text_color( 0.0, 0.0, 0.0, 1.0 );

    osg::ref_ptr< osgText::Text > textDrawable = new osgText::Text;
    textDrawable->setText( text );
    textDrawable->setPosition( text_pos );
    textDrawable->setAlignment( osgText::Text::LEFT_CENTER );
    textDrawable->setAxisAlignment( osgText::Text::SCREEN );
    textDrawable->setCharacterSizeMode( osgText::Text::SCREEN_COORDS );
    textDrawable->setCharacterSize( char_size );
    textDrawable->setColor( text_color );

    m_textGeode->addDrawable( textDrawable );
    m_rootGroup->addChild( m_textGeode );

    WLEMDDrawable3D::osgNodeCallback( nv );
}
