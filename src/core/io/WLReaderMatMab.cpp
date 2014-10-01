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

#include <fstream>
#include <string>
#include <vector>

#include <boost/shared_ptr.hpp>

#include <core/common/WLogger.h>
#include <core/common/WStringUtils.h>
#include <core/dataHandler/io/WReader.h>

#include "core/data/WLDataTypes.h"

#include "WLReaderMatMab.h"

using namespace std;
using WLMatrix::MatrixT;

const std::string WLReaderMatMab::CLASS = "WLReaderMatMab";

WLReaderMatMab::WLReaderMatMab( std::string fname ) :
                WLReader( fname )
{
    wlog::debug( CLASS ) << "file: " << fname;
}

WLReaderMatMab::~WLReaderMatMab()
{
}

WLReaderMatMab::ReturnCode::Enum WLReaderMatMab::read( WLMatrix::SPtr& matrix )
{
    ifstream ifs;
    ifs.open( m_fname.c_str(), ifstream::in );

    if( !ifs || ifs.bad() )
    {
        return ReturnCode::ERROR_FOPEN;
    }

    string line;

    ReturnCode::Enum rc = ReturnCode::ERROR_UNKNOWN;
    size_t cols = 0, rows = 0;
    try
    {
        while( ifs.good() )
        {
            getline( ifs, line );
            if( ifs.good() && line.find( "NumberRows=" ) == 0 )
            {
                vector< string > tokens = string_utils::tokenize( line );
                rows = string_utils::fromString< size_t >( tokens.at( 1 ) );
                wlog::debug( CLASS ) << "Number of rows: " << rows;
            }
            else
                if( ifs.good() && line.find( "NumberColumns=" ) == 0 )
                {
                    vector< string > tokens = string_utils::tokenize( line );
                    cols = string_utils::fromString< size_t >( tokens.at( 1 ) );
                    wlog::debug( CLASS ) << "Number of columns: " << cols;
                }
        }
    }
    catch( WTypeMismatch& e )
    {
        wlog::error( CLASS ) << e.what();
        rc = ReturnCode::ERROR_UNKNOWN;
    }
    ifs.close();

    if( cols > 0 && rows > 0 )
    {
        string mabFile = m_fname.substr( 0, m_fname.find_last_of( ".mat" ) - 3 );
        mabFile = mabFile.append( ".mab" );
        wlog::debug( CLASS ) << "Matix file: " << mabFile;
        matrix.reset( new MatrixT( rows, cols ) );
        rc = readMab( matrix, mabFile, rows, cols );
    }
    else
    {
        wlog::error( CLASS ) << "Rows/Cols < 1 or could not read! Matrix remains empty.";
    }

    return rc;
}

WLReaderMatMab::ReturnCode::Enum WLReaderMatMab::readMab( WLMatrix::SPtr matrix, std::string fName, size_t rows, size_t cols )
{
    if( static_cast< size_t >( matrix->rows() ) != rows || static_cast< size_t >( matrix->cols() ) != cols )
    {
        wlog::error( CLASS ) << "Row or column size incorrect!";
        return ReturnCode::ERROR_UNKNOWN;
    }

    ifstream ifs;
    ifs.open( fName.c_str(), ifstream::in | ifstream::binary );

    if( !ifs || ifs.bad() )
    {
        return ReturnCode::ERROR_FOPEN;
    }

    ifs.seekg( 0, ifs.end );
    wlog::debug( CLASS ) << "File size: " << ifs.tellg();
    ifs.seekg( 0, ios::beg );

    size_t row = 0, col = 0, count = 0;
    float fval = 0;
    const size_t ELEMENTS = rows * cols;

    while( ifs.read( ( char* )&fval, sizeof( fval ) ) && count < ELEMENTS )
    {
        ( *matrix )( row, col ) = fval;
        // row-major order
        ++col;
        if( col == cols )
        {
            col = 0;
            ++row;
        }
        ++count;
    }

    wlog::debug( CLASS ) << "Elements read: " << count;

    ifstream::pos_type current = ifs.tellg();
    ifs.seekg( 0, ifs.end );
    wlog::debug( CLASS ) << "Bytes left: " << ifs.tellg() - current;

    return ReturnCode::SUCCESS;
}

