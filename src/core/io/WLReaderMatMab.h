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

#ifndef WLREADERMATMAB_H_
#define WLREADERMATMAB_H_

#include <string>

#include <boost/shared_ptr.hpp>
#include <Eigen/Core>

#include "core/data/WLDataTypes.h"
#include "core/io/WLReader.h"

class WLReaderMatMab: public WLReader
{
public:
    static const std::string CLASS;

    /**
     * Abbreviation for a shared pointer.
     */
    typedef boost::shared_ptr< WLReaderMatMab > SPtr;

    /**
     * Abbreviation for const shared pointer.
     */
    typedef boost::shared_ptr< const WLReaderMatMab > ConstSPtr;

    /**
     * Constructs a reader object.
     *
     * \param fname path to file which should be loaded
     */
    explicit WLReaderMatMab( std::string fname );
    virtual ~WLReaderMatMab();

    ReturnCode::Enum read( WLMatrix::SPtr& matrix );

private:
    ReturnCode::Enum readMab( WLMatrix::SPtr matrix, std::string fName, size_t rows, size_t cols );
};

#endif  // WLREADERMATMAB_H_
