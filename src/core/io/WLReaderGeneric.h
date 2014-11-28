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

#ifndef WLREADERGENERIC_H_
#define WLREADERGENERIC_H_

#include <string>

#include <core/dataHandler/io/WReader.h>

#include "core/io/WLIOStatus.h"

/**
 * Generic interface/abstract class for a uniform file reader.
 *
 * \author pieloth
 * \ingroup io
 */
template< typename T >
class WLReaderGeneric: public WReader
{
public:
    explicit WLReaderGeneric( std::string fname );

    virtual ~WLReaderGeneric();

    /**
     * Reads the data into out.
     *
     * \param out Instance to set read data.
     * \return WLIOStatus::SUCCESS if data was read successfully.
     */
    virtual WLIOStatus::IOStatusT read( T* const out ) = 0;

    /**
     * Closes streams, releases handles and more.
     */
    virtual void close();
};

template< typename T >
WLReaderGeneric< T >::WLReaderGeneric( std::string fname ) :
                WReader( fname )
{
}

template< typename T >
WLReaderGeneric< T >::~WLReaderGeneric()
{
    close();
}

template< typename T >
void WLReaderGeneric< T >::close()
{
}

#endif  // WLREADERGENERIC_H_
