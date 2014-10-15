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

#ifndef WLMATLIB_H_
#define WLMATLIB_H_

#include <fstream>
#include <list>
#include <string>

#include <boost/cstdint.hpp> // fixed data types

#include <Eigen/Core>

/**
 * A low-level C++ API to read/write MAT-file format v5 from MATLAB.
 *
 * \author pieloth
 */
namespace WLMatLib
{
    const std::string LIBNAME = "WLMatFileIO";

    typedef uint8_t mArrayType_t;
    typedef uint32_t mArrayFlags_t;

    typedef uint32_t mDataType_t;
    typedef uint16_t mDataTypeSmall_t;

    typedef uint32_t mNumBytes_t;
    typedef uint16_t mNumBytesSmall_t;

    typedef int8_t miINT8_t;
    typedef uint8_t miUINT8_t;

    typedef int16_t miINT16_t;
    typedef uint16_t miUINT16_t;

    typedef int32_t miINT32_t;
    typedef uint32_t miUINT32_t;

    typedef int64_t miINT64_t;
    typedef uint64_t miUINT64_t;

    typedef float miSinge_t;
    typedef double miDouble_t;

    /**
     * MAT-file Data Types for the Tag Field.
     */
    namespace DataTypes
    {
        // NOTE: incomplete
        const mDataType_t miINT8 = 1;
        const mDataType_t miUINT8 = 2;
        const mDataType_t miINT16 = 3;
        const mDataType_t miUINT16 = 4;
        const mDataType_t miINT32 = 5;
        const mDataType_t miUINT32 = 6;
        const mDataType_t miSINGLE = 7;
        const mDataType_t miDOUBLE = 9;
        const mDataType_t miINT64 = 12;
        const mDataType_t miUINT64 = 13;
        const mDataType_t miMATRIX = 14;
        const mDataType_t miUTF32 = 18;
    }

    /**
     * MATLAB Array Types (Classes) for Array Flags Subelement.
     *
     * \param type
     * \return
     */
    namespace ArrayTypes
    {
        /**
         * Checks if Class/Type belongs to a numeric array.
         *
         * \param type
         * \return
         */
        bool isNumericArray( const mArrayType_t& type );

        // NOTE: incomplete
        const mArrayType_t mxCHAR_CLASS = 4;
        const mArrayType_t mxDOUBLE_CLASS = 6;
        const mArrayType_t mxSINGLE_CLASS = 7;
        const mArrayType_t mxINT8_CLASS = 8;
        const mArrayType_t mxUINT8_CLASS = 9;
        const mArrayType_t mxINT16_CLASS = 10;
        const mArrayType_t mxUINT16_CLASS = 11;
        const mArrayType_t mxINT32_CLASS = 12;
        const mArrayType_t mxUINT32_CLASS = 13;
    }

    /**
     * Helper functions to handle Array Flags Subelement.
     */
    namespace ArrayFlags
    {
        const miUINT32_t MASK_COMPLEX = 0x00000800;
        const miUINT32_t MASK_GLOBAL = 0x00000400;
        const miUINT32_t MASK_LOGICAL = 0x00000200;
        const miUINT32_t MASK_GET_CLASS = 0x0000000F;

        bool isComplex( const mArrayFlags_t& data );
        bool isGlobal( const mArrayFlags_t& data );
        bool isLogical( const mArrayFlags_t& data );

        mArrayType_t getArrayType( const mArrayFlags_t& data );
    }

    /**
     * Information of a MAT-file e.g. read from the header.
     */
    struct FileInfo
    {
        bool isMatFile;
        bool isLittleEndian;
        std::string description;
        size_t fileSize;
    };

    typedef struct FileInfo FileInfo_t;

    /**
     * Information of a Data Element.
     */
    struct ElementInfo
    {
        std::streampos pos;
        std::streampos posData;
        mDataType_t dataType;
        mNumBytes_t numBytes;
        mArrayFlags_t arrayFlags;
        miINT32_t rows;
        miINT32_t cols;
        std::string arrayName;
    };

    typedef struct ElementInfo ElementInfo_t;

    /**
     * Low-level reader for MAT-file format.
     * NOTE: Does only supports: little endian, 2-dim double matrices, no compression.
     */
    class MATReader
    {
    public:
        /**
         * Reads the header from a MAT-file format.
         * If returns true, file position points to the start of the first data element,
         * otherwise it points to file beginning.
         *
         * \param info Struct to store the information.
         * \param ifs Open input stream to read from.
         *
         * \return true if successful, false otherwise.
         */
        static bool readHeader( FileInfo_t* const info, std::ifstream& ifs );

        /**
         * Retrieves all data elements in the file. Leaves file position at the end of the header.
         *
         * \param elements List to store found elements.
         * \param ifs Open input stream to read from.
         * \param info File information e.g. to handle endia format.
         *
         * \return true, if successful, false otherwise.
         */
        static bool retrieveDataElements( std::list< ElementInfo_t >* const elements, std::ifstream& ifs,
                        const FileInfo_t& info );

        /**
         * Reads the matrix which is contained by the element.
         *
         * \param matrix Matrix to fill.
         * \param element Element which contains the matrix to read.
         * \param ifs Open input stream to read from.
         * \param info File information e.g. to handle endian format.
         *
         * \return true, if successful, false otherwise.
         */
        static bool readMatrixDouble( Eigen::MatrixXd* const matrix, const ElementInfo_t& element, std::ifstream& ifs,
                        const FileInfo_t& info );

    private:
        static bool readTagField( mDataType_t* const dataType, mNumBytes_t* const numBytes, std::ifstream& ifs );

        static bool readArraySubelements( ElementInfo_t* const element, std::ifstream& ifs );

        static void nextElement( std::ifstream& ifs, const std::streampos& tagStart, size_t numBytes );
    };

    /**
     * Low-level writer for MAT-file format.
     * NOTE: Does only supports: little endian, 2-dim double matrices, no compression.
     */
    class MATWriter
    {
    public:
        /**
         * Writes the MAT-file header. Leaves file position at the end of header, if successful.
         *
         * \param ofs Open output stream.
         * \param description Description text for header.
         *
         * \return true, if successful.
         */
        static bool writeHeader( std::ofstream& ofs, const std::string& description );

        /**
         * Writes 2-dim matrix to file. If successful, file position points to the end of the written data.
         * Otherwise file positions is reset, but bytes are still written!
         *
         * \param ofs Open output stream.
         * \param matrix Matrix to write.
         * \param arrayName Variable name.
         *
         * \return Written bytes.
         */
        static size_t writeMatrixDouble( std::ofstream& ofs, const Eigen::MatrixXd& matrix, const std::string& arrayName );

    private:
        static size_t writeTagField( std::ofstream& ofs, const mDataType_t& dataType, const mNumBytes_t numBytes );
    };
}

#endif  // WLMATLIB_H_
