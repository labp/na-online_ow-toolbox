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
#include <list>
#include <string>

#include <core/common/WLogger.h>

#include "core/dataFormat/mat/WLMatLib.h"

#include "WLReaderMAT.h"

using std::ifstream;

const std::string WLReaderMAT::CLASS = "WLReaderMat";

const WLIOStatus::IOStatusT WLReaderMAT::ERROR_NO_MATRIXD = 0 + WLIOStatus::_USER_OFFSET;

WLReaderMAT::WLReaderMAT( std::string fname ) throw( WDHNoSuchFile ) :
                WLReaderGeneric< WLMatrix::SPtr >( fname ), WLIOStatus::WLIOStatusInterpreter()
{
    m_isInitialized = false;
}

WLReaderMAT::~WLReaderMAT()
{
}

WLIOStatus::IOStatusT WLReaderMAT::init()
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

WLIOStatus::IOStatusT WLReaderMAT::read( WLMatrix::SPtr* const matrix )
{
    if( !m_isInitialized )
    {
        WLIOStatus::IOStatusT state = init();
        if( state != WLIOStatus::SUCCESS )
        {
            return state;
        }
    }

    WLIOStatus::IOStatusT rc = ERROR_NO_MATRIXD;
    std::list< WLMatLib::ElementInfo_t >::const_iterator it;
    for( it = m_elements.begin(); it != m_elements.end(); ++it )
    {
        if( it->dataType == WLMatLib::DataTypes::miMATRIX )
        {
            wlog::debug( CLASS ) << "Found matrix element.";
            const bool isComplex = WLMatLib::ArrayFlags::isComplex( it->arrayFlags );
            const bool isDouble = WLMatLib::ArrayFlags::getArrayType( it->arrayFlags ) == WLMatLib::ArrayTypes::mxDOUBLE_CLASS;
            if( isDouble && !isComplex )
            {
                wlog::debug( CLASS ) << "Found matrix element with double class.";
                if( !( *matrix ) )
                {
                    matrix->reset( new WLMatrix::MatrixT() );
                }
#ifndef LABP_FLOAT_COMPUTATION
                if( WLMatLib::MATReader::readMatrixDouble( matrix->get(), *it, m_ifs, m_fileInfo ) )
                {
                    rc = WLIOStatus::SUCCESS;
                    break;
                }
#else
                Eigen::MatrixXd matrixDbl;
                if( WLMatLib::MATReader::readMatrixDouble( &matrixDbl, *it, m_ifs, m_fileInfo ) )
                {
                    ( *matrix ) = matrixDbl.cast< WLMatrix::ScalarT >();
                    rc = WLIOStatus::SUCCESS;
                    break;
                }
#endif
            }
        }
    }

    return rc;
}

std::string WLReaderMAT::getIOStatusDescription( WLIOStatus::IOStatusT status ) const
{
    if( status == ERROR_NO_MATRIXD )
    {
        return "File does not contain a double matrix!";
    }
    else
    {
        return WLIOStatus::description( status );
    }
}
