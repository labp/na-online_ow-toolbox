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

#include <core/common/WAssert.h>
#include <core/common/WLogger.h>

#include "WLMatFileIO.h"

using std::ifstream;

bool WLMatFileIO::MATReader::readHeader( FileInfo_t* const infoIn, std::ifstream& ifs )
{
    if( infoIn == NULL )
    {
        wlog::error( LIBNAME ) << "FileInfo_t is null!";
        ifs.seekg( 0, ifs.beg );
        return false;
    }

    infoIn->fileSize = 0;
    infoIn->isMatFile = false;

    if( !ifs || ifs.bad() )
    {
        wlog::error( LIBNAME ) << "Problem with input stream!";
        ifs.seekg( 0, ifs.beg );
        return false;
    }

    // Check minimum file size
    ifs.seekg( 0, ifs.end );
    const ifstream::pos_type file_size = ifs.tellg();
    infoIn->fileSize = file_size;
    wlog::debug( LIBNAME ) << "File size: " << infoIn->fileSize;
    if( file_size < 127 )
    {
        wlog::error( LIBNAME ) << "File size is to small for a MAT file!";
        ifs.seekg( 0, ifs.beg );
        return false;
    }
    ifs.seekg( 0, ifs.beg );

    // Read description text
    char description[117];
    ifs.read( description, 116 );
    description[116] = '\0';
    infoIn->description.assign( description );
    wlog::debug( LIBNAME ) << description;

    // Read version
    ifs.seekg( 8, ifstream::cur );
    char version[2] = { 0 };
    ifs.read( version, 2 );
    if( version[0] != 0x00 || version[1] != 0x01 )
    {
        wlog::error( LIBNAME ) << "Wrong version!";
        ifs.seekg( 0, ifs.beg );
        return false;
    }
    infoIn->isMatFile = true;

    // Read endian indicator
    char endian[2] = { 0 };
    ifs.read( endian, 2 );
    if( endian[0] == 'I' && endian[1] == 'M' )
    {
        infoIn->isLittleEndian = true;
    }
    else
        if( endian[0] == 'M' && endian[1] == 'I' )
        {
            infoIn->isLittleEndian = false;
            WAssert( true, "Big endian is not yet supported!" );
        }
        else
        {
            wlog::error( LIBNAME ) << "Unknown endian indicator!";
            return false;
        }

    ifs.seekg( 128 );

    return true;
}

bool WLMatFileIO::MATReader::retrieveDataElements( std::list< ElementInfo_t >* const elements, std::ifstream& ifs,
                const FileInfo_t& info )
{
    if( elements == NULL )
    {
        wlog::error( LIBNAME ) << "List for ElementInfo_t is null!";
        return false;
    }
    ifs.seekg( 128 );

    // Temporary data to write to
    WLMatFileIO::mDataType_t type;
    WLMatFileIO::mNumBytes_t bytes;
    std::streampos pos;
    const std::streamoff min_tag_size = 4;
    while( ifs.good() && ifs.tellg() + min_tag_size < info.fileSize )
    {
        type = 0;
        bytes = 0;
        pos = ifs.tellg();
        if( !readTagField( &type, &bytes, ifs, info ) )
        {
            wlog::error( LIBNAME ) << "Unknown data type or wrong data structure. Cancel retrieving!";
            ifs.seekg( 128 );
            return false;
        }

        ElementInfo_t element;
        element.dataType = type;
        element.numBytes = bytes;
        element.pos = pos;
        wlog::debug( LIBNAME ) << "Data Type: " << element.dataType;
        wlog::debug( LIBNAME ) << "Number of Bytes: " << element.numBytes;

        if( element.dataType == WLMatFileIO::DataTypes::miMATRIX )
        {
            if( !readArraySubelements( &element, ifs, info ) )
            {
                nextElement( ifs, element.pos, element.numBytes );
                continue;
            }
        }

        nextElement( ifs, element.pos, element.numBytes );
        elements->push_back( element );
    }

    ifs.clear();
    ifs.seekg( 128 );
    return true;
}

bool WLMatFileIO::MATReader::readTagField( mDataType_t* const dataType, mNumBytes_t* const numBytes, std::ifstream& ifs,
                const FileInfo_t& info )
{
    std::streampos pos = ifs.tellg();
    ifs.read( ( char* )dataType, sizeof(WLMatFileIO::mDataType_t) );
    ifs.read( ( char* )numBytes, sizeof(WLMatFileIO::mNumBytes_t) );
    if( *dataType > WLMatFileIO::DataTypes::miUTF32 )
    {
        wlog::debug( LIBNAME ) << "Small Data Element Format found.";
        WLMatFileIO::mDataTypeSmall_t typeSmall;
        WLMatFileIO::mNumBytesSmall_t bytesSmall;
        ifs.seekg( -( sizeof(WLMatFileIO::mDataType_t) + sizeof(WLMatFileIO::mNumBytes_t) ), ifstream::cur );
        ifs.read( ( char* )&typeSmall, sizeof(WLMatFileIO::mDataTypeSmall_t) );
        ifs.read( ( char* )&bytesSmall, sizeof(WLMatFileIO::mNumBytesSmall_t) );
        *dataType = typeSmall;
        *numBytes = bytesSmall;
    }
    if( *dataType > WLMatFileIO::DataTypes::miUTF32 )
    {
        wlog::error( LIBNAME ) << "Unknown data type or wrong data structure!";
        ifs.seekg( pos );
        return false;
    }
    return true;
}

bool WLMatFileIO::MATReader::readArraySubelements( ElementInfo_t* const element, std::ifstream& ifs, const FileInfo_t& info )
{
    if( element == NULL )
    {
        wlog::error( LIBNAME ) << "ElementInfo_t is null!";
        return false;
    }

    ifs.seekg( element->pos );
    ifs.seekg( 8, ifstream::cur );
    if( !ifs.good() )
    {
        wlog::error( LIBNAME ) << "Could not jump to element: " << element->pos << "/" << info.fileSize;
        ifs.seekg( element->pos );
        return false;
    }

    WLMatFileIO::mDataType_t type;
    WLMatFileIO::mNumBytes_t bytes;
    std::streampos tagStart;
    // Read Array Flags //
    // ---------------- //
    if( !readTagField( &type, &bytes, ifs, info ) )
    {
        wlog::error( LIBNAME ) << "Could not read Array Flags!";
        return false;
    }
    else
        if( bytes != 8 || type != DataTypes::miUINT32 )
        {
            wlog::error( LIBNAME ) << "Bytes for Array Flags or Data Type is wrong: " << bytes << " (expected: 8) or " << type
                            << " (expected: " << DataTypes::miUINT32 << ")";
            ifs.seekg( element->pos );
            return false;
        }

    // Read flags and class
    mArrayFlags_t arrayFlags[2];
    ifs.read( ( char* )&arrayFlags, 8 );
    const mArrayFlags_t arrayFlag = arrayFlags[0];
    wlog::debug( LIBNAME ) << "Array Flag: " << arrayFlag;
    if( ArrayFlags::isComplex( arrayFlag ) )
    {
        wlog::debug( LIBNAME ) << "Is complex.";
    }
    if( ArrayFlags::isGlobal( arrayFlag ) )
    {
        wlog::debug( LIBNAME ) << "Is global.";
    }
    if( ArrayFlags::isLogical( arrayFlag ) )
    {
        wlog::debug( LIBNAME ) << "Is logical.";
    }

    element->arrayFlags = arrayFlag;
    const mArrayType_t clazz = ArrayFlags::getArrayType( arrayFlag );
    wlog::debug( LIBNAME ) << "Array Type/Class: " << ( miUINT32_t )ArrayFlags::getArrayType( arrayFlag );
    if( !ArrayTypes::isNumericArray( clazz ) && clazz != ArrayTypes::mxCHAR_CLASS )
    {
        element->posData = ifs.tellg();
        ifs.seekg( element->pos );
        return true;
    }

    wlog::debug( LIBNAME ) << "Data element is numeric array. Retrieving more subelements.";

    // Read Dimension //
    // -------------- //
    tagStart = ifs.tellg();
    if( !readTagField( &type, &bytes, ifs, info ) )
    {
        wlog::error( LIBNAME ) << "Could not read Dimension!";
        return false;
    }
    else
        if( bytes < 8 || type != DataTypes::miINT32 )
        {
            wlog::error( LIBNAME ) << "Bytes for Dimension or Data Type is wrong: " << bytes << " (expected: 8) or " << type
                            << " (expected: " << DataTypes::miINT32 << ")";
            ifs.seekg( element->pos );
            return false;
        }

    // TODO(pieloth): Check for dimensions n > 2
    WAssert( bytes == 8, "Dimension n != 2 is not yet supported!" );
    ifs.read( ( char* )&element->rows, sizeof(miINT32_t) );
    ifs.read( ( char* )&element->cols, sizeof(miINT32_t) );

    if( element->rows < 1 || element->cols < 1 )
    {
        wlog::error( LIBNAME ) << "Rows/Cols error: " << element->rows << "x" << element->cols;
        ifs.seekg( element->pos );
        return false;
    }
    wlog::debug( LIBNAME ) << "Array size: " << element->rows << "x" << element->cols;
    nextElement( ifs, tagStart, bytes );

    // Read Array Name //
    // --------------- //
    tagStart = ifs.tellg();
    if( !readTagField( &type, &bytes, ifs, info ) )
    {
        wlog::error( LIBNAME ) << "Could not read Array Name!";
        return false;
    }
    else
        if( type != DataTypes::miINT8 )
        {
            wlog::error( LIBNAME ) << "Data Type is wrong: " << type << " (expected: " << DataTypes::miINT8 << ")";
            ifs.seekg( element->pos );
            return false;
        }

    char* tmp = ( char* )malloc( bytes + 1 );
    ifs.read( tmp, bytes );
    tmp[bytes] = '\0';
    element->arrayName.assign( tmp );
    free( tmp );
    wlog::debug( LIBNAME ) << "Array Name: " << element->arrayName;
    nextElement( ifs, tagStart, bytes );

    // Set Data Position
    element->posData = ifs.tellg();
    return true;
}

bool WLMatFileIO::MATReader::readMatrixDouble( Eigen::MatrixXd* const matrix, const ElementInfo_t& element, std::ifstream& ifs,
                const FileInfo_t& info )
{
    // Check some errors //
    // ----------------- //
    if( matrix == NULL )
    {
        wlog::error( LIBNAME ) << "Matrix object is null!";
        return false;
    }

    if( info.fileSize <= element.posData )
    {
        wlog::error( LIBNAME ) << "Data position is beyond file end!";
        return false;
    }

    if( element.dataType != DataTypes::miMATRIX )
    {
        wlog::error( LIBNAME ) << "Data type is not a matrix: " << element.dataType;
        return false;
    }

    const mArrayType_t arrayType = ArrayFlags::getArrayType( element.arrayFlags );
    if( arrayType != ArrayTypes::mxDOUBLE_CLASS )
    {
        wlog::error( LIBNAME ) << "Numeric Types does not match!";
        return false;
    }

    std::streampos pos = ifs.tellg();

    // Read data //
    // --------- //
    ifs.seekg( element.posData );
    mDataType_t type;
    mNumBytes_t bytes;
    if( !readTagField( &type, &bytes, ifs, info ) )
    {
        wlog::error( LIBNAME ) << "Could not read Data Element!";
        ifs.seekg( pos );
        return false;
    }
    if( type != DataTypes::miDOUBLE )
    {
        wlog::error( LIBNAME ) << "Numeric Types does not match or compressed data, which is not supported: " << type;
        ifs.seekg( pos );
        return false;
    }

    matrix->resize( element.rows, element.cols );
    ifs.read( ( char* )matrix->data(), bytes );

    nextElement( ifs, element.posData, bytes );
    return true;
}

void WLMatFileIO::MATReader::nextElement( std::ifstream& ifs, const std::streampos& tagStart, size_t numBytes )
{
    ifs.seekg( tagStart );
    if( numBytes > 4 ) // short data element
    {
        ifs.seekg( 8, ifstream::cur );
        if( numBytes % 8 )
        {
            numBytes = 8 - ( numBytes % 8 );
        }
        ifs.seekg( numBytes, ifstream::cur );
    }
    else
    {
        ifs.seekg( 8, ifstream::cur );
    }
}
