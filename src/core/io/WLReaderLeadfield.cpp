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

#include <string>

#include <mne/mne_forwardsolution.h>
#include <QtCore/QFile>

#include <core/common/WLogger.h>

#include "WLReaderLeadfield.h"

using std::string;

const std::string WLReaderLeadfield::CLASS = "WLReaderLeadfield";

WLReaderLeadfield::WLReaderLeadfield( string fname ) throw( WDHNoSuchFile ) :
                WReader( fname )
{
}

WLReaderLeadfield::~WLReaderLeadfield()
{
}

WLIOStatus::ioStatus_t WLReaderLeadfield::read( WLMatrix::SPtr& leadfield )
{
    QFile fileIn( m_fname.c_str() );

    MNELIB::MNEForwardSolution::SPtr fwdSolution( new MNELIB::MNEForwardSolution( fileIn ) );
    if( fwdSolution->isEmpty() )
    {
        wlog::error( CLASS ) << "Could not read leadfield file!";
        return WLIOStatus::ERROR_FREAD;
    }

#ifdef LABP_FLOAT_COMPUTATION
    WLMatrix::MatrixT* matrix = new WLMatrix::MatrixT( fwdSolution->sol->data.cast<ScalarT>() );
#else
    WLMatrix::MatrixT* matrix = new WLMatrix::MatrixT( fwdSolution->sol->data );
#endif  // LABP_FLOAT_COMPUTATION
    leadfield.reset( matrix );
    wlog::info( CLASS ) << "Matrix size: " << leadfield->rows() << "x" << leadfield->cols();
    return WLIOStatus::SUCCESS;
}
