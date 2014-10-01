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

#include <cstdlib>
#include <string>
#include <vector>

#include <osg/Image>
#include <osgSim/ColorRange>

#include <core/common/WAssert.h>
#include <core/common/WLogger.h>

#include "WLColorMap.h"
#include "WLColorMapClassic.h"
#include "WLColorMapHot.h"
#include "WLColorMapHSV.h"

const std::string WLColorMap::CLASS = "WLColorMap";

WLColorMap::WLColorMap( ValueT min, ValueT max, WEColorMapMode::Enum mode ) :
                m_mode( mode )
{
    if( mode == WEColorMapMode::ABSOLUTE && min < 0 )
    {
        min = 0;
        wlog::warn( CLASS ) << "Mode is absolute but min is not 0! Using 0 instead!";
    }
    if( min > max )
    {
        wlog::warn( CLASS ) << "min > max! Swapping values.";
        ValueT tmp = min;
        min = max;
        max = tmp;
    }
    else
        if( min == max )
        {
            max = min == 0.0 ? 1 : ( min + fabs( min ) );
            wlog::warn( CLASS ) << "min == max! Changed max to: " << max;
        }

    m_colorRange = new osgSim::ColorRange( min, max );
    m_range = max - min;
}

WLColorMap::~WLColorMap()
{
    delete m_colorRange;
}

WLColorMap::ValueT WLColorMap::getMin() const
{
    return m_colorRange->getMin();
}

WLColorMap::ValueT WLColorMap::getMax() const
{
    return m_colorRange->getMax();
}

WLColorMap::ColorT WLColorMap::getColor( ValueT scalar ) const
{
    if( m_mode == WEColorMapMode::ABSOLUTE )
    {
        scalar = fabs( scalar );
    }
    return m_colorRange->getColor( scalar );
}

void WLColorMap::setColors( std::vector< ColorT > colors )
{
    m_colorRange->setColors( colors );
}

std::vector< WLColorMap::ColorT > WLColorMap::getColor( const std::vector< ValueT >& values ) const
{
    std::vector< ColorT > colors;
    colors.reserve( values.size() );
    for( std::vector< ValueT >::const_iterator it = values.begin(); it != values.end(); ++it )
    {
        colors.push_back( WLColorMap::getColor( *it ) );
    }
    return colors;
}

WLColorMap::TextureRefT WLColorMap::getAsTexture( size_t resolution ) const
{
    osg::ref_ptr< osg::Image > image = new osg::Image;
    // allocate the image data, size x 1 x 1 with 4 rgba floats - equivalent to a Vec4!
    image->allocateImage( resolution, 1, 1, GL_RGBA, GL_FLOAT );
    image->setInternalTextureFormat( GL_RGBA );

    osg::Vec4* data = reinterpret_cast< osg::Vec4* >( image->data() );
    ValueT min = m_colorRange->getMin();
    ValueT step = ( m_range ) / static_cast< ValueT >( resolution );
    for( size_t i = 0; i < resolution; ++i )
    {
        data[i] = this->getColor( min + i * step );
    }

    TextureRefT refText = new TextureT( image );
    refText->setWrap( osg::Texture1D::WRAP_S, osg::Texture1D::CLAMP_TO_EDGE );
    refText->setFilter( osg::Texture1D::MIN_FILTER, osg::Texture1D::LINEAR );
    return refText;
}

WLColorMap::TextCoordT WLColorMap::getTextureCoordinate( ValueT scalar ) const
{
    if( m_mode == WEColorMapMode::ABSOLUTE )
    {
        scalar = fabs( scalar );
    }

    scalar = ( scalar - m_colorRange->getMin() ) / m_range;
    return scalar;
}

std::vector< WLColorMap::TextCoordT > WLColorMap::getTextureCoordinate( const std::vector< ValueT >& values ) const
{
    std::vector< TextCoordT > textCoords;
    textCoords.reserve( values.size() );
    for( std::vector< ValueT >::const_iterator it = values.begin(); it != values.end(); ++it )
    {
        textCoords.push_back( WLColorMap::getTextureCoordinate( *it ) );
    }
    return textCoords;
}

WEColorMapMode::Enum WLColorMap::getMode() const
{
    return m_mode;
}

std::vector< WEColorMap::Enum > WEColorMap::values()
{
    std::vector< WEColorMap::Enum > maps;
    maps.push_back( WEColorMap::HOT );
    maps.push_back( WEColorMap::HSV );
    maps.push_back( WEColorMap::CLASSIC );
    return maps;
}

std::string WEColorMap::name( WEColorMap::Enum val )
{
    switch( val )
    {
        case WEColorMap::HOT:
            return "White-Red";
        case WEColorMap::HSV:
            return "Red-Red";
        case WEColorMap::CLASSIC:
            return "Blue-Red";
        default:
            WAssert( false, "Unknown WEColorMap!" );
            return "ERROR: Unknown!";
    }
}

WLColorMap::SPtr WEColorMap::instance( WEColorMap::Enum val, float min, float max, WEColorMapMode::Enum mode )
{
    switch( val )
    {
        case WEColorMap::HOT:
            return WLColorMap::SPtr( new WLColorMapHot( min, max, mode ) );
        case WEColorMap::HSV:
            return WLColorMap::SPtr( new WLColorMapHSV( min, max, mode ) );
        case WEColorMap::CLASSIC:
            return WLColorMap::SPtr( new WLColorMapClassic( min, max, mode ) );
        default:
            WAssert( false, "Unknown WEColorMap!" );
            return WLColorMap::SPtr( new WLColorMapClassic( min, max, mode ) );
    }
}

std::string WEColorMapMode::name( Enum val )
{
    switch( val )
    {
        case WEColorMapMode::NORMAL:
            return "Bipolar";
        case WEColorMapMode::ABSOLUTE:
            return "Unipolar";
        default:
            WAssert( false, "Unknown WEColorMapMode!" );
            return "ERROR: Unknown!";
    }
}

std::vector< WEColorMapMode::Enum > WEColorMapMode::values()
{
    std::vector< WEColorMapMode::Enum > maps;
    maps.push_back( WEColorMapMode::NORMAL );
    maps.push_back( WEColorMapMode::ABSOLUTE );
    return maps;
}
