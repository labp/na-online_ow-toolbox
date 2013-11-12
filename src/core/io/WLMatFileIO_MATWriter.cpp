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

#include <algorithm>

#include <core/common/WLogger.h>

#include "WLMatFileIO.h"

using std::ofstream;

// TODO(pieloth): Actually there must be a ofs.good() after each write to return the correct written bytes!

bool WLMatFileIO::MATWriter::writeHeader( ofstream& ofs, const std::string& description )
{
    if( !ofs || ofs.bad() )
    {
        wlog::error( LIBNAME ) << "Problem with output stream!!";
        return false;
    }

    if( ofs.tellp() != ofstream::beg )
    {
        wlog::warn( LIBNAME ) << "Set file pointer to beginning!";
        ofs.seekp( ofstream::beg );
    }

    char descrBytes[116] = { '\0' };
    const char subsys_data_offset[8] = { 0x20 };

    const size_t length = std::min< size_t >( 115, description.length() );
    for( size_t i = 0; i < length; ++i )
    {
        descrBytes[i] = description.at( i );
    }

    ofs.write( descrBytes, 116 );
    ofs.write( subsys_data_offset, 8 );

    const char version[2] = { 0x00, 0x01 };
    ofs.write( version, 2 );

    const char endian_indicator[2] = { 'I', 'M' };
    ofs.write( endian_indicator, 2 );

    return true;
}

size_t WLMatFileIO::MATWriter::writeTagField( std::ofstream& ofs, const mDataType_t& dataType, const mNumBytes_t numBytes )
{
    if( !ofs || ofs.bad() )
    {
        wlog::error( LIBNAME ) << "Problem with output stream!!";
        return 0;
    }

    ofs.write( ( char* )&dataType, sizeof( dataType ) );
    ofs.write( ( char* )&numBytes, sizeof( numBytes ) );

    return sizeof( dataType ) + sizeof( numBytes );
}

size_t WLMatFileIO::MATWriter::writeMatrixDouble( std::ofstream& ofs, const Eigen::MatrixXd& matrix,
                const std::string& arrayName )
{
    if( !ofs || ofs.bad() )
    {
        wlog::error( LIBNAME ) << "Problem with output stream!!";
        return 0;
    }

    // Init //
    // ---- //
    const std::streampos pos = ofs.tellp();
    mDataType_t type;
    mNumBytes_t bytes;
    size_t tmpBytes = 0;
    size_t writtenBytes = 0;

    // Write Array Tag //
    // --------------- //
    type = DataTypes::miMATRIX;
    bytes = 0; // set it after written subelements an data!
    tmpBytes = writeTagField( ofs, type, bytes );
    writtenBytes += tmpBytes;
    if( tmpBytes == 0 )
    {
        ofs.seekp( pos );
        wlog::error( LIBNAME ) << "Could not write Array Tag!";
        return writtenBytes;
    }
    // Write Array Flags //
    // ----------------- //
    type = DataTypes::miUINT32;
    bytes = 8;
    tmpBytes = writeTagField( ofs, type, bytes );
    writtenBytes += tmpBytes;
    if( tmpBytes == 0 )
    {
        ofs.seekp( pos );
        wlog::error( LIBNAME ) << "Could not write tag for Array Flags!";
        return writtenBytes;
    }

    mArrayFlags_t arrayFlags = 0;
    // arrayFlags |= ArrayFlags::MASK_GLOBAL;
    mArrayType_t arrayType = ArrayTypes::mxDOUBLE_CLASS;
    mArrayFlags_t arrayTypeCast = arrayType;
    arrayFlags |= arrayTypeCast; // TODO(pieloth): Check if this is correct
    ofs.write( ( char* )&arrayFlags, sizeof( arrayFlags ) );
    writtenBytes += sizeof( arrayFlags );
    arrayType = 0;
    ofs.write( ( char* )&arrayFlags, sizeof( arrayFlags ) );
    writtenBytes += sizeof( arrayFlags );

    // Write Dimension //
    // --------------- //
    type = DataTypes::miINT32;
    bytes = 8;
    tmpBytes = writeTagField( ofs, type, bytes );
    writtenBytes += tmpBytes;
    if( tmpBytes == 0 )
    {
        ofs.seekp( pos );
        wlog::error( LIBNAME ) << "Could not write tag for Dimension!";
        return writtenBytes;
    }
    const miINT32_t rows = matrix.rows();
    const miINT32_t cols = matrix.cols();
    ofs.write( ( char* )&rows, sizeof( rows ) );
    ofs.write( ( char* )&cols, sizeof( cols ) );
    writtenBytes += sizeof( rows ) + sizeof( cols );

    // Write Array Name //
    // ---------------- //
    type = DataTypes::miINT8;
    bytes = arrayName.length();
    tmpBytes = writeTagField( ofs, type, bytes );
    writtenBytes += tmpBytes;
    if( tmpBytes == 0 )
    {
        ofs.seekp( pos );
        wlog::error( LIBNAME ) << "Could not write tag for Array Name!";
        return writtenBytes;
    }
    ofs.write( arrayName.c_str(), bytes );
    writtenBytes += bytes;
    if( bytes % 8 )
    {
        bytes = 8 - ( bytes % 8 );
        for( size_t i = 0; i < bytes; ++i )
        {
            ofs.put( '\0' );
            ++writtenBytes;
        }
    }

    // Write matrix data //
    // ----------------- //
    type = DataTypes::miDOUBLE;
    bytes = matrix.rows() * matrix.cols() * sizeof(miDouble_t);
    tmpBytes = writeTagField( ofs, type, bytes );
    writtenBytes += tmpBytes;
    if( tmpBytes == 0 )
    {
        ofs.seekp( pos );
        wlog::error( LIBNAME ) << "Could not write tag for Matrix!";
        return writtenBytes;
    }

    ofs.write( ( char* )matrix.data(), bytes );
    writtenBytes += bytes;
    if( bytes % 8 )
    {
        bytes = 8 - ( bytes % 8 );
        for( size_t i = 0; i < bytes; ++i )
        {
            ofs.put( '\0' );
            ++writtenBytes;
        }
    }

    // Set correct numBytes for miMatrix //
    // --------------------------------- //
    bytes = writtenBytes - sizeof(mDataType_t) - sizeof(mNumBytes_t);
    ofs.seekp( pos );
    ofs.seekp( sizeof(mDataType_t), ofstream::cur );
    ofs.write( ( char* )&bytes, sizeof(mNumBytes_t) );
    ofs.seekp( bytes, ofstream::cur );

    return writtenBytes;
}
