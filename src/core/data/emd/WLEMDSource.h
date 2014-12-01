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

#ifndef WLEMDSOURCE_H_
#define WLEMDSOURCE_H_

#include <string>

#include <boost/shared_ptr.hpp>

#include "core/data/emd/WLEMData.h"

/**
 * Reconstructed source activity.
 *
 * \author pieloth
 * \ingroup data
 */
class WLEMDSource: public WLEMData
{
public:
    /**
     * Abbreviation for a shared pointer.
     */
    typedef boost::shared_ptr< WLEMDSource > SPtr;

    /**
     * Abbreviation for const shared pointer.
     */
    typedef boost::shared_ptr< const WLEMDSource > ConstSPtr;

    static const std::string CLASS; //!< Class name for logging purpose.

    WLEMDSource();

    explicit WLEMDSource( const WLEMDSource& source );

    explicit WLEMDSource( const WLEMData& emd );

    virtual ~WLEMDSource();

    virtual WLEMData::SPtr clone() const;

    virtual WLEModality::Enum getModalityType() const;

    /**
     * Gets the modality type from which the source activity was estimated.
     *
     * \return Modality type.
     */
    WLEModality::Enum getOriginModalityType() const;

    /**
     * Sets the modality type from which the source activity was estimated.
     *
     * \param modality Modality type.
     */
    void setOriginModalityType( WLEModality::Enum modality );

private:
    WLEModality::Enum m_originModalityType; //!< Modality type from which the source activity was estimated.
};

#endif  // WLEMDSOURCE_H_
