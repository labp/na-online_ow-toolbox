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

#ifndef WLEMDEOG_H
#define WLEMDEOG_H

#include <boost/shared_ptr.hpp>

#include "WLEMData.h"

/**
 * Electrooculography data and related information. Not yet used neither developed.
 */
class WLEMDEOG: public WLEMData
{
public:
    /**
     * Abbreviation for a shared pointer.
     */
    typedef boost::shared_ptr< WLEMDEOG > SPtr;

    /**
     * Abbreviation for const shared pointer.
     */
    typedef boost::shared_ptr< const WLEMDEOG > ConstSPtr;

    WLEMDEOG();

    explicit WLEMDEOG( const WLEMDEOG& eog );

    virtual ~WLEMDEOG();

    virtual WLEMData::SPtr clone() const;

    virtual WLEModality::Enum getModalityType() const;
};

#endif  // WLEMDEOG_H
