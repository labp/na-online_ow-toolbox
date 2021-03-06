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

#ifndef WLREADERMAT_H_
#define WLREADERMAT_H_

#include <fstream>
#include <list>
#include <string>

#include <boost/shared_ptr.hpp>
#include <Eigen/Dense>

#include <core/dataHandler/exceptions/WDHNoSuchFile.h>
#include <core/dataHandler/io/WReader.h>

#include "core/data/WLDataTypes.h"
#include "core/dataFormat/mat/WLMatLib.h"
#include "WLIOStatus.h"

/**
 * Reads a matrix from a MATLAB MAT-file.
 *
 * \author pieloth
 * \ingroup io
 */
class WLReaderMAT: public WReader, public WLIOStatus::WLIOStatusInterpreter
{
public:
    const static std::string CLASS;

    const static WLIOStatus::IOStatusT ERROR_NO_MATRIXD; /**< File does not contain a double matrix. */
    const static WLIOStatus::IOStatusT ERROR_NO_MATRIXC; /**< File does not contain a complex matrix. */

    /**
     * Shared pointer abbreviation to a instance of this class.
     */
    typedef boost::shared_ptr< WLReaderMAT > SPtr;

    /**
     * Shared pointer abbreviation to a const instance of this class.
     */
    typedef boost::shared_ptr< const WLReaderMAT > ConstSPtr;

    explicit WLReaderMAT( std::string fname ) throw( WDHNoSuchFile );
    virtual ~WLReaderMAT();

    /**
     * Opens the input stream and reads initial information.
     *
     * \return SUCCESS, if successful.
     */
    WLIOStatus::IOStatusT init();

    /**
     * Reads the first matrix from the file which matches the data type.
     *
     * \param matrix Matrix to fill.
     *
     * \return SUCCESS, if successful.
     */
    WLIOStatus::IOStatusT read( WLMatrix::SPtr* const matrix );

    /**
     * Reads the first complex matrix from the file which matches the data type.
     *
     * \param matrix Matrix to fill.
     *
     * \return SUCCESS, if successful.
     */
    WLIOStatus::IOStatusT read( Eigen::MatrixXcd* const matrix );

    virtual void close();

    virtual std::string getIOStatusDescription( WLIOStatus::IOStatusT status ) const;

private:
    std::ifstream m_ifs;

    bool m_isInitialized;

    WLMatLib::FileInfo_t m_fileInfo;
    std::list< WLMatLib::ElementInfo_t > m_elements;
};

#endif  // WLREADERMAT_H_
