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

#ifndef WLPOSITIONS_H_
#define WLPOSITIONS_H_

#include <ostream>
#include <string>

#include <boost/shared_ptr.hpp>
#include <Eigen/Core>

#include "core/data/enum/WLECoordSystem.h"
#include "core/data/enum/WLEExponent.h"
#include "core/data/enum/WLEUnit.h"

/**
 * A container-like class for positions/points in different units and coordinate systems. The aim is to prevent calculations with
 * positions in different coordinate systems and units. Use it in combination with WLTransformation.
 *
 * \author pieloth
 * \ingroup data
 */
class WLPositions
{
public:
    typedef boost::shared_ptr< WLPositions > SPtr; //!< Abbreviation for a shared pointer.
    typedef boost::shared_ptr< const WLPositions > ConstSPtr; //!< Abbreviation for a const shared pointer.

    typedef Eigen::Vector3d PositionT; //!< A position with a x-, y- and z-value.
    typedef Eigen::Matrix3Xd PositionsT; //!< Rows: x, y, z; Columns: channels
    typedef PositionsT::Index IndexT; //!< Index type to access positions.
    typedef PositionsT::Scalar ScalarT; //!< Value type.

    static const std::string CLASS;

    /**
     * Constructor.
     *
     * \param unit Unit e.g. meter. Default: NONE
     * \param exponent Exponent e.g. milli. Default: BASE
     * \param coordSystem Coordinate system e.g. head. Default: UNKNOWN
     */
    WLPositions( WLEUnit::Enum unit = WLEUnit::UNKNOWN, WLEExponent::Enum exponent = WLEExponent::BASE,
                    WLECoordSystem::Enum coordSystem = WLECoordSystem::UNKNOWN );

    /**
     * Copy constructor, applies unit, exponent, coordSystem and resizes the positions.
     * \note Positions are not copied!
     *
     * \param pos WLPositions to copy.
     */
    explicit WLPositions( const WLPositions& pos );

    /**
     * Destructor.
     */
    virtual ~WLPositions();

    /**
     * Returns a new instance.
     *
     * \return new instance.
     */
    static WLPositions::SPtr instance();

    /**
     * Gets the i-th position.
     *
     * \throws WOutOfBounds
     * \param i Index
     * \return Position
     */
    PositionT at( PositionsT::Index i );

    /**
     * Gets the i-th position.
     *
     * \throws WOutOfBounds
     * \param i Index
     * \return Position
     */
    const PositionT at( PositionsT::Index i ) const;

    /**
     * Sets new positions or data.
     * \attention Old positions are lost!
     *
     * \param positions New positions.
     */
    void data( const PositionsT& positions );

    /**
     * Return positions for direct and in place access.
     *
     * \return A reference to the data.
     */
    PositionsT& data();

    /**
     * Return positions for direct and in place access.
     *
     * \return A const reference to the data.
     */
    const PositionsT& data() const;

    /**
     * Gets the count of positions.
     *
     * \return Count of positions.
     */
    WLPositions::IndexT size() const;

    /**
     * Resizes the position count.
     * \attention Old data is lost!
     *
     * \param nPos New count of positions.
     */
    void resize( WLPositions::IndexT nPos );

    /**
     * Checks if the size is 0.
     * \attention Does not check, if "empty" positions were set!, e.g. after a resize!
     *
     * \return True if size is 0.
     */
    bool empty() const;

    /**
     * Gets the unit.
     *
     * \return The unit.
     */
    WLEUnit::Enum unit() const;

    /**
     * Sets the unit.
     *
     * \param unit Unit to set.
     */
    void unit( WLEUnit::Enum unit );

    /**
     * Gets the exponent.
     *
     * \return THe exponent.
     */
    WLEExponent::Enum exponent() const;

    /**
     * Sets the exponent.
     *
     * \param exponent Exponent to set.
     */
    void exponent( WLEExponent::Enum exponent );

    /**
     * Gets the coordinate system.
     *
     * \return The coordinate system.
     */
    WLECoordSystem::Enum coordSystem() const;

    /**
     * Sets the coordinate system.
     *
     * \param coordSystem Coordinate system to set.
     */
    void coordSystem( WLECoordSystem::Enum coordSystem );

    /**
     * Checks if units and(!) exponents are compatible.
     *
     * \param positions Positions to check.
     * \return True if unit and exponent are compatible (equals or unknown).
     */
    bool isUnitCompatible( const WLPositions& positions ) const;

    /**
     * Checks if the positions are compatible.
     *
     * \param positions Positions to check.
     * \return True if unit, exponent and coordSystem are compatible (equals or unknown).
     */
    bool isCompatible( const WLPositions& positions ) const;

    /**
     * Appends positions to this instance.
     * \attention Be aware of "empty" positions!
     * \attention This function is costly and does a resize and copy!
     *
     * \throws WPreconditionNotMet
     * \param positions Positions to append.
     * \return Object with appended positions.
     */
    WLPositions& operator+=( const WLPositions& positions );

private:
    PositionsT m_positions;
    WLEUnit::Enum m_unit;
    WLEExponent::Enum m_exponent;
    WLECoordSystem::Enum m_coordSystem;
};

inline std::ostream& operator<<( std::ostream &strm, const WLPositions& obj )
{
    strm << WLPositions::CLASS << ": ";
    strm << "size=" << obj.size() << ", ";
    strm << "unit=" << obj.unit() << ", ";
    strm << "exponent=" << obj.exponent() << ", ";
    strm << "coordSystem=" << obj.coordSystem();
    return strm;
}

#endif  // WLPOSITIONS_H_
