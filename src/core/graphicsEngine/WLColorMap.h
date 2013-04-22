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

#ifndef WLCOLORMAP_H_
#define WLCOLORMAP_H_

#include <string>
#include <vector>

#include <boost/shared_ptr.hpp>

#include <osg/Texture1D>
#include <osg/Vec4>
#include <osgSim/ColorRange>
#include <osgSim/ScalarsToColors>

namespace LaBP
{
    class WLColorMap;

    /**
     * Describes whether the absolute value or the value as it is should be used.
     */
    namespace WEColorMapMode
    {
        enum Enum
        {
            NORMAL, ABSOLUTE
        };

        std::string name( Enum val );
        std::vector< Enum > values();
    }

    /**
     * List of implemented predefined color mappings.
     */
    namespace WEColorMap
    {
        enum Enum
        {
            CLASSIC, HOT, HSV
        };

        std::string name( Enum val );
        boost::shared_ptr< WLColorMap > instance( Enum val, float min, float max, WEColorMapMode::Enum mode );
        std::vector< Enum > values();
    }

    /**
     * A base class for predefined color mappings from scalar values to colors.
     * This class implements all necessary methods.
     * Derived classes must set their specific color map in the constructor only.
     * Note: This class also generates a texture for a correct color mapping, which can be done on the GPU.
     * The "color map texture" has a domain of [0;1]. A scalar must be mapped on this domain to get a correct mapping with GPU support.
     * See page 44 of thesis "Kombinierte Visualisierung von EEG- und Diffusions-MRT-Nervenfaser-Daten" by Cornelius Müller.
     */
    class WLColorMap: public osgSim::ScalarsToColors
    {
    public:
        typedef osg::Vec4 ColorT;
        typedef float ValueT;
        typedef osg::Texture1D TextureT;
        typedef osg::ref_ptr< osg::Texture1D > TextureRefT;
        typedef float TextCoordT;

        /**
         * Abbreviation for a shared pointer.
         */
        typedef boost::shared_ptr< WLColorMap > SPtr;

        /**
         * Abbreviation for const shared pointer.
         */
        typedef boost::shared_ptr< const WLColorMap > ConstSPtr;

        /**
         * Constructor sets necessary variables for base class.
         *
         * \param min Defines the minimum scalar value used color interpolation. If mode is WEColorMapMode::ABSOLUTE, min is set to 0!
         * \param max Defines the maximum scalar value used for color interpolation.
         * \param mode If mode is WEColorMapMode::ABSOLUTE, minimum is set to 0 and the absolute value of the scalar is used.
         */
        WLColorMap( ValueT min = -1.0, ValueT max = 1.0, WEColorMapMode::Enum mode = WEColorMapMode::NORMAL );
        virtual ~WLColorMap();

        /**
         * Returns the color for a scalar value.
         * If the value is not between the minimum and maximum, the least distant value in the defined range is used.
         *
         * \param scalar Value to get the color of
         * \return associated color
         */
        virtual ColorT getColor( ValueT scalar ) const;

        /**
         * Same as getColor() for scalars.
         *
         * \param values Vector of scalar values
         * \return associated colors
         */
        virtual std::vector< ColorT > getColor( const std::vector< ValueT >& values ) const;

        /**
         * Creates a texture which can be used by the GPU to compute the correct color gradient.
         *
         * \param resolution Resolution of the texture. Standard: 256
         * \return 1D Texture for the associate color map. Domain: [0;1]
         */
        virtual TextureRefT getAsTexture( size_t resolution = 256 ) const;

        /**
         * Calculates the texture coordinate in the domain [0;1] for a scalar value between min/max. This is the color mapping on a one dimensional texture.
         * The texture coordinate can be used to set the "color", respectively the coordinate with the associated color, for the surface interpolation.
         *
         * \param scalar Value to get the texture coordinate of
         * \return texture coordinate for the scalar
         */
        virtual TextCoordT getTextureCoordinate( ValueT scalar ) const;

        /**
         * Same as getTextureCoordinate() for scalars.
         *
         * \param values Vector of scalar values
         * \return associated texture coordinates
         */
        virtual std::vector< TextCoordT > getTextureCoordinate( const std::vector< ValueT >& values ) const;

        /**
         * Returns the color mapping type of this instance.
         *
         * \return the color mapping type of this instance
         */
        virtual WEColorMap::Enum getType() const = 0;

        /**
         * Returns the mode of this instance.
         *
         * \return the mode of this instance
         */
        virtual WEColorMapMode::Enum getMode() const;

    protected:
        /**
         * Sets the colors for the color mapping.
         * This method must be called in the constructor of an derived class, otherwise there will be no color mapping.
         *
         * \param vector of colors
         */
        void setColors( std::vector< ColorT > colors );

    private:
        /**
         * Is actually used for the color mapping and interpolation.
         * Attention: Should be private, otherwise all setters of this variable must be delegated to base class osgSim::ScalarsToColors
         */
        osgSim::ColorRange* m_colorRange;

        /**
         * max-min
         */
        ValueT m_range;
        WEColorMapMode::Enum m_mode;
    };

} /* namespace LaBP */
#endif /* WLCOLORMAP_H_ */
