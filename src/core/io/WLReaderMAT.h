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

#ifndef WLREADERMAT_H_
#define WLREADERMAT_H_

#include <fstream>
#include <list>
#include <string>

#include <core/dataHandler/io/WReader.h>

#include "core/data/WLDataTypes.h"

#include "WLIOStatus.h"
#include "WLMatFileIO.h"

/**
 * Reads a matrix from a MATLAB MAT-file.
 *
 * @author pieloth
 */
class WLReaderMAT: public WReader, WLIOStatus::WLIOStatusInterpreter
{
public:
    const static std::string CLASS;

    explicit WLReaderMAT( std::string fname );
    virtual ~WLReaderMAT();

    /**
     * Opens the input stream and reads initial information.
     *
     * @return SUCCESS, if successful.
     */
    WLIOStatus::ioStatus_t init();

    /**
     * Reads the first matrix from the file which matchs the data type.
     *
     * @param matrix Matrix to fill.
     *
     * @return SUCCESS, if successful.
     */
    WLIOStatus::ioStatus_t readMatrix( WLMatrix::SPtr& matrix );

    /**
     * Closes the input stream.
     */
    void close();

private:
    std::ifstream m_ifs;

    bool m_isInitialized;

    WLMatFileIO::FileInfo_t m_fileInfo;
    std::list< WLMatFileIO::ElementInfo_t > m_elements;

};

#endif  // WLREADERMAT_H_
