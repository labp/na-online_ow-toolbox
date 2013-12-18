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

#include <fstream>

#include <core/common/WLogger.h>

#include "core/fileFormat/mat/WLMatLib.h"

#include "WLReaderMAT.h"

using std::ifstream;

const std::string WLReaderMAT::CLASS = "WLReaderMat";

WLReaderMAT::WLReaderMAT( std::string fname ) throw( WDHNoSuchFile ) :
                WReader( fname )
{
    m_isInitialized = false;
}

WLReaderMAT::~WLReaderMAT()
{
    close();
}

WLIOStatus::ioStatus_t WLReaderMAT::init()
{
    if( m_isInitialized && m_ifs.is_open() )
    {
        return WLIOStatus::SUCCESS;
    }
    else
    {
        m_isInitialized = false;
    }

    m_ifs.open( m_fname.c_str(), ifstream::in | ifstream::binary );
    if( !m_ifs || m_ifs.bad() )
    {
        wlog::error( CLASS ) << "Could not open file!";
        return WLIOStatus::ERROR_FOPEN;
    }

    if( !WLMatLib::MATReader::readHeader( &m_fileInfo, m_ifs ) )
    {
        wlog::error( CLASS ) << "Error while reading MAT-file header!";
        return WLIOStatus::ERROR_FREAD;
    }

    std::list< WLMatLib::ElementInfo_t > elements;
    if( !WLMatLib::MATReader::retrieveDataElements( &m_elements, m_ifs, m_fileInfo ) )
    {
        wlog::error( CLASS ) << "Error while retrieving data elements!";
        return WLIOStatus::ERROR_UNKNOWN;
    }

    m_isInitialized = true;
    return WLIOStatus::SUCCESS;
}

void WLReaderMAT::close()
{
    if( m_ifs && m_ifs.is_open() )
    {
        m_ifs.close();
    }
}

WLIOStatus::ioStatus_t WLReaderMAT::readMatrix( WLMatrix::SPtr& matrix )
{
    if( !m_isInitialized )
    {
        WLIOStatus::ioStatus_t state = init();
        if( state != WLIOStatus::SUCCESS )
        {
            return state;
        }
    }

    WLIOStatus::ioStatus_t rc = WLIOStatus::ERROR_UNKNOWN;
    std::list< WLMatLib::ElementInfo_t >::const_iterator it;
    for( it = m_elements.begin(); it != m_elements.end(); ++it )
    {
        if( it->dataType == WLMatLib::DataTypes::miMATRIX )
        {
            wlog::debug( CLASS ) << "Found matrix element.";
            if( WLMatLib::ArrayFlags::getArrayType( it->arrayFlags ) == WLMatLib::ArrayTypes::mxDOUBLE_CLASS )
            {
                wlog::debug( CLASS ) << "Found matrix element with double class.";
                if( !matrix )
                {
                    matrix.reset( new WLMatrix::MatrixT() );
                }
#ifndef LABP_FLOAT_COMPUTATION
                if( WLMatLib::MATReader::readMatrixDouble( matrix.get(), *it, m_ifs, m_fileInfo ) )
                {
                    rc = WLIOStatus::SUCCESS;
                    break;
                }
#else
                Eigen::MatrixXd matrixDbl;
                if( WLMatLib::MATReader::readMatrixDouble( &matrixDbl, *it, m_ifs, m_fileInfo ) )
                {
                    (*matrix) = matrixDbl.cast<WLMatrix::ScalarT>();
                    rc = WLIOStatus::SUCCESS;
                    break;
                }
#endif
            }
        }
    }

    return rc;
}
