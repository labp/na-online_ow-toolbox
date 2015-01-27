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

#ifndef WLTRANSFORMATION_H_
#define WLTRANSFORMATION_H_

#include <ostream>
#include <string>

#include <boost/shared_ptr.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>  // homogeneous

#include <core/common/WLogger.h>
#include <core/common/exceptions/WPreconditionNotMet.h>

#include "core/data/WLPositions.h"
#include "core/data/enum/WLECoordSystem.h"
#include "core/data/enum/WLEExponent.h"
#include "core/data/enum/WLEUnit.h"

/**
 * Transformation matrix for different units and coordinate systems. The aim is to prevent calculations with
 * positions in different coordinate systems and units. Use it in combination with WLPositions.
 *
 * \author pieloth
 * \ingroup data
 */
class WLTransformation
{
public:
    typedef boost::shared_ptr< WLTransformation > SPtr; //!< Abbreviation for a shared pointer.
    typedef boost::shared_ptr< const WLTransformation > ConstSPtr; //!< Abbreviation for a const shared pointer.

    typedef Eigen::Matrix4d TransformationT;
    typedef TransformationT::Scalar ScalarT; //!< Value type.

    static const std::string CLASS;

    /**
     * Constructor.
     *
     * \param from From coordinate system e.g. head. Default: UNKNOWN
     * \param to To coordinate system e.g. device. Default: UNKNOWN
     * \param unit Unit e.g. meter. Default: NONE
     * \param exponent Exponent e.g. milli. Default: BASE
     */
    WLTransformation( WLECoordSystem::Enum from = WLECoordSystem::UNKNOWN, WLECoordSystem::Enum to = WLECoordSystem::UNKNOWN,
                    WLEUnit::Enum unit = WLEUnit::NONE, WLEExponent::Enum exponent = WLEExponent::BASE );

    /**
     * Destructor.
     */
    virtual ~WLTransformation();

    /**
     * Returns a new instance.
     *
     * \return new instance.
     */
    static WLTransformation::SPtr instance();

    /**
     * Sets a new transformation matrix.
     *
     * \param transformation New positions.
     */
    void data( const TransformationT& transformation );

    /**
     * Return transformation matrix for direct and in place access.
     *
     * \return A reference to the data.
     */
    TransformationT& data();

    /**
     * Return transformation matrix for direct and in place access.
     *
     * \return A const reference to the data.
     */
    const TransformationT& data() const;

    /**
     * Checks if a transformation matrix is set.
     *
     * \return True if no transformation matrix is set.
     */
    bool empty() const;

    /**
     * Gets the from coordinate system.
     *
     * \return The coordinate system.
     */
    WLECoordSystem::Enum from() const;

    /**
     * Sets the from coordinate system.
     *
     * \param coordSystem Coordinate system to set.
     */
    void from( WLECoordSystem::Enum coordSystem );

    /**
     * Gets the  to coordinate system.
     *
     * \return The coordinate system.
     */
    WLECoordSystem::Enum to() const;

    /**
     * Sets the to coordinate system.
     *
     * \param coordSystem Coordinate system to set.
     */
    void to( WLECoordSystem::Enum coordSystem );

    /**
     * Gets the unit of the translation.
     *
     * \return The unit.
     */
    WLEUnit::Enum unit() const;

    /**
     * Sets the unit of the translation.
     *
     * \param unit Unit to set.
     */
    void unit( WLEUnit::Enum unit );

    /**
     * Gets the exponent of the translation.
     *
     * \return THe exponent.
     */
    WLEExponent::Enum exponent() const;

    /**
     * Sets the exponent of the translation.
     *
     * \param exponent Exponent to set.
     */
    void exponent( WLEExponent::Enum exponent );

    /**
     * Creates the inverse transformation, i.e. to->from.
     *
     * \return New instance with swapped from/to and inverse matrix.
     */
    WLTransformation::SPtr inverse() const;

    /**
     * Applies the transformation to positions.
     *
     * \throws WPreconditionNotMet
     * \param positions Positions to transfrom.
     * \return Transformed positions.
     */
    WLPositions::SPtr operator*( const WLPositions& positions ) const;

private:
    TransformationT m_transformation;

    WLECoordSystem::Enum m_from;
    WLECoordSystem::Enum m_to;

    WLEUnit::Enum m_unit;
    WLEExponent::Enum m_exponent;
};

inline std::ostream& operator<<( std::ostream &strm, const WLTransformation& obj )
{
    strm << "WLTransformation: ";
    strm << "from=" << obj.from() << ", ";
    strm << "to=" << obj.to() << ", ";
    strm << "unit=" << obj.unit() << ", ";
    strm << "exponent=" << obj.exponent() << ", ";
    strm << "matrix=[" << obj.data().row( 0 ) << ", " << obj.data().row( 1 ) << ", " << obj.data().row( 2 ) << ", "
                    << obj.data().row( 3 ) << "]";
    return strm;
}

#endif  // WLTRANSFORMATION_H_
