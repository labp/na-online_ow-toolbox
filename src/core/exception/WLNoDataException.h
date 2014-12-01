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

#ifndef WLNODATAEXCEPTION_H_
#define WLNODATAEXCEPTION_H_

#include <core/common/WException.h>

#include <string>

/**
 * Indicates an exception caused by no data or empty containers.
 *
 * \author pieloth
 * \ingroup exception
 */
class WLNoDataException: public WException
{
public:
    /**
     * Default constructor.
     * \param msg Exception description.
     */
    explicit WLNoDataException( const std::string& msg = std::string() );

    /**
     * Destructor.
     */
    virtual ~WLNoDataException() throw();
};

#endif  // WLNODATAEXCEPTION_H_
