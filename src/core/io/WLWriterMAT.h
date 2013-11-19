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

#ifndef WLWRITERMAT_H_
#define WLWRITERMAT_H_

#include <fstream>
#include <string>

#include <boost/shared_ptr.hpp>

#include <core/dataHandler/io/WWriter.h>

#include "core/data/WLDataTypes.h"
#include "WLIOStatus.h"

/**
 * Stores a Matrix in MATLAB MAT-file format.
 *
 * @author pieloth
 */
class WLWriterMAT: public WWriter, public WLIOStatus::WLIOStatusInterpreter
{
public:
    /**
     * Shared pointer abbreviation to a instance of this class.
     */
    typedef boost::shared_ptr< WLWriterMAT > SPtr;

    /**
     * Shared pointer abbreviation to a const instance of this class.
     */
    typedef boost::shared_ptr< const WLWriterMAT > ConstSPtr;

    static const std::string CLASS;

    WLWriterMAT( std::string fname, bool overwrite = false );
    virtual ~WLWriterMAT();

    /**
     * Opens the output streams and prepares the file.
     *
     * @return SUCCESS, if successful.
     */
    WLIOStatus::ioStatus_t init();

    /**
     * Writes a matrix to a file.
     *
     * @param matrix Matrix to write.
     * @param name Variable name in MAT-file.
     *
     * @return SUCCESS, if successful.
     */
    WLIOStatus::ioStatus_t writeMatrix( WLMatrix::ConstSPtr matrix, const std::string& name = "M" );

    WLIOStatus::ioStatus_t writeMatrix( const WLMatrix::MatrixT& matrix, const std::string& name = "M" );

    /**
     * Closes the output stream.
     */
    void close();

private:
    static const std::string DESCRIPTION;

    bool m_isInitialized;
    std::ofstream m_ofs;
};

#endif  // WLWRITERMAT_H_
