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

#include <ctime>
#include <fstream>
#include <string>

#include <core/common/WLogger.h>

#include "core/dataFormat/mat/WLMatLib.h"

#include "WLWriterMAT.h"

using std::ofstream;

const std::string WLWriterMAT::CLASS = "WLWriterMAT";

const std::string WLWriterMAT::DESCRIPTION = "OpenWalnut, NA-Online, " + CLASS;

WLWriterMAT::WLWriterMAT( std::string fname, bool overwrite ) :
                WWriter( fname, overwrite )
{
    m_isInitialized = false;
}

WLWriterMAT::~WLWriterMAT()
{
    close();
}

WLIOStatus::IOStatusT WLWriterMAT::init()
{
    if( m_isInitialized && m_ofs.is_open() )
    {
        return WLIOStatus::SUCCESS;
    }
    else
    {
        m_isInitialized = false;
    }

    m_ofs.open( m_fname.c_str(), ofstream::out | ofstream::binary );
    if( !m_ofs || m_ofs.bad() )
    {
        wlog::error( CLASS ) << "Could not open file!";
        return WLIOStatus::ERROR_FOPEN;
    }

    // Write header //
    // ------------ //
    time_t timer;
    time( &timer );
    char timeString[20] = { '\0' };
    const std::string time_format = "%Y-%m-%dT%H:%M:%S";
    strftime( timeString, 20, time_format.c_str(), localtime( &timer ) );

    if( WLMatLib::MATWriter::writeHeader( m_ofs, DESCRIPTION + ", " + timeString ) )
    {
        m_isInitialized = true;
        return WLIOStatus::SUCCESS;
    }
    else
    {
        wlog::error( CLASS ) << "Could not write MAT-file header!";
        close();
        return WLIOStatus::ERROR_FWRITE;
    }
}

WLIOStatus::IOStatusT WLWriterMAT::writeMatrix( WLMatrix::ConstSPtr matrix, const std::string& name )
{
    return writeMatrix( *matrix, name );
}

WLIOStatus::IOStatusT WLWriterMAT::writeMatrix( const WLMatrix::MatrixT& matrix, const std::string& name )
{
    if( !m_isInitialized )
    {
        WLIOStatus::IOStatusT state = init();
        if( state != WLIOStatus::SUCCESS )
        {
            return state;
        }
    }

    // Write data //
#ifndef LABP_FLOAT_COMPUTATION
    const size_t bytes = WLMatLib::MATWriter::writeMatrixDouble( m_ofs, matrix, name );
#else
    const Eigen::MatrixXd matrixDbl = matrix.cast<double>();
    const size_t bytes = WLMatLib::MATWriter::writeMatrixDouble( m_ofs, matrixDbl, name );
#endif
    wlog::debug( CLASS ) << bytes << " bytes written to file.";
    // NOTE: min_bytes is only possible, if compression is not used!
    // min_byte = Tag(miMATRIX) + Tag(ArrayFlags)+Data(ArrayFlags)
    // min_byte += Tag(Dim)+Data(Dim) + SmallElement(ArrayName) + Tag(matrix)+Data(matrix)
    const size_t min_bytes = 8 + 2 * 8 + 2 * 8 + 8 + 8 + matrix.rows() * matrix.cols() * sizeof( WLMatLib::miDouble_t );
    if( min_bytes <= bytes )
    {
        return WLIOStatus::SUCCESS;
    }
    else
    {
        wlog::error( CLASS ) << "Error while writing matrix!";
        return WLIOStatus::ERROR_FWRITE;
    }
}

void WLWriterMAT::close()
{
    if( m_ofs && m_ofs.is_open() )
    {
        m_ofs.close();
    }
}
