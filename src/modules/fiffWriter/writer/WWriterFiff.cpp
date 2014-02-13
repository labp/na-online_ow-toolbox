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

#include <Eigen/Core>
#include <fiff/fiff_ch_info.h>
#include <fiff/fiff_constants.h>
#include <fiff/fiff_coord_trans.h>
#include <fiff/fiff_dig_point.h>
#include <fiff/fiff_info.h>
#include <fiff/fiff_stream.h>
#include <QFile>
#include <QList>
#include <QString>

#include <core/common/WLogger.h>
#include "core/data/WLEMMeasurement.h"
#include "core/data/WLDigPoint.h"
#include "core/data/emd/WLEMData.h"
#include "core/data/emd/WLEMDEEG.h"
#include "core/data/emd/WLEMDMEG.h"

#include "WWriterFiff.h"

using Eigen::Vector3d;
using Eigen::RowVectorXd;
using Eigen::MatrixXd;
using Eigen::MatrixXi;
using std::string;
using FIFFLIB::FiffChInfo;
using FIFFLIB::FiffCoordTrans;
using FIFFLIB::FiffDigPoint;
using FIFFLIB::FiffInfo;
using FIFFLIB::FiffStream;

const string WWriterFiff::CLASS = "WWriterFiff";

WWriterFiff::WWriterFiff( std::string fname, bool pickEEG, bool pickMEG, bool pickStim, bool overwrite ) throw( WDHIOFailure ) :
                WWriter( fname, overwrite ), m_pickEEG( pickEEG ), m_pickMEG( pickMEG ), m_pickStim( pickStim )
{
    m_file = NULL;
}

WWriterFiff::~WWriterFiff()
{
    close();
}

bool WWriterFiff::open()
{
    wlog::debug( CLASS ) << "open() called!";
    if( m_file != NULL )
    {
        return true;
    }

    QString fname( m_fname.c_str() );
    m_file = new QFile( fname );
    if( !m_file )
    {
        return false;
    }

    return true;
}

bool WWriterFiff::close()
{
    wlog::debug( CLASS ) << "close() called!";
    if( m_file != NULL )
    {
        if( m_fiffStream )
        {
            m_fiffStream->finish_writing_raw();
            m_fiffStream.clear();
        }
        m_file->close();
        free( m_file );
        m_file = NULL;
        return true;
    }
    else
    {
        return false;
    }
}

bool WWriterFiff::write( WLEMMeasurement::ConstSPtr emm )
{
    if( !beginFiff( emm.get() ) )
    {
        return false;
    }

    if( !m_pickEEG && !m_pickMEG && !m_pickStim )
    {
        wlog::warn( CLASS ) << "No picks were set - nothing to write!";
        return false;
    }

    return writeData( emm.get() );
}

bool WWriterFiff::beginFiff( const WLEMMeasurement* const emm )
{
    if( m_fiffStream )
    {
        return true;
    }

    // Creating FIFF info
    //-------------------
    FiffInfo info;
    info.nchan = 0;
    float sfreq = 0;
    QList< FiffChInfo > chs;
    WLEMDEEG::ConstSPtr eeg;
    if( m_pickEEG && emm->hasModality( WLEModality::EEG ) )
    {
        eeg = emm->getModality< const WLEMDEEG >( WLEModality::EEG );
        info.nchan += eeg->getNrChans();
        sfreq = eeg->getSampFreq();
        setChannelInfo( &chs, eeg.get() );
    }
    WLEMDMEG::ConstSPtr meg;
    if( m_pickMEG && emm->hasModality( WLEModality::MEG ) )
    {
        meg = emm->getModality< const WLEMDMEG >( WLEModality::MEG );
        info.nchan += meg->getNrChans();
        sfreq = sfreq == 0 ? meg->getSampFreq() : sfreq;
        setChannelInfo( &chs, meg.get() );
    }

    WLEMMeasurement::EDataT* stim = NULL; // dirty hack to get pointer
    if( m_pickStim )
    {
        stim = emm->getEventChannels().get();
        info.nchan += stim->size();
        setChannelInfo( &chs, stim );
    }

    info.sfreq = sfreq;
    info.chs = chs;

    FiffCoordTrans devToHead;
    devToHead.from = FIFFV_COORD_DEVICE;
    devToHead.to = FIFFV_COORD_HEAD;
#if LABP_FLOAT_COMPUTATION
    devToHead.trans = emm->getDevToFidTransformation();
#else
    devToHead.trans = emm->getDevToFidTransformation().cast< float >();
#endif
    info.dev_head_t = devToHead;

    setDigPoint( &info, emm );

    // Writing FIFF information
    // ------------------------
    MatrixXd cals = MatrixXd( 1, info.nchan );
    cals.setZero();
    m_fiffStream = FiffStream::start_writing_raw( *m_file, info, cals );
    if( m_fiffStream )
    {
        return true;
    }
    else
    {
        wlog::error( CLASS ) << "Could not start writing raw!";
        return false;
    }
}

bool WWriterFiff::setDigPoint( FIFFLIB::FiffInfo* const info, const WLEMMeasurement* const emm )
{
    if( emm->getDigPoints()->empty() )
    {
        wlog::debug( CLASS ) << "No digPoints to write!";
        return false;
    }

    QList< FiffDigPoint > digs;
    WLList< WLDigPoint >::ConstSPtr digPoints = emm->getDigPoints();
    WLList< WLDigPoint >::const_iterator it;
    for( it = digPoints->begin(); it != digPoints->end(); ++it )
    {
        FiffDigPoint digPoint;
        digPoint.ident = it->getIdent();
        digPoint.kind = it->getKind();
        digPoint.r[0] = it->getPoint().x();
        digPoint.r[1] = it->getPoint().y();
        digPoint.r[2] = it->getPoint().z();
        digs.append( digPoint );
    }

    info->dig = digs;

    return true;
}

bool WWriterFiff::writeData( const WLEMMeasurement* const emm )
{
    if( !m_fiffStream )
    {
        wlog::error( CLASS ) << "Could not writing raw!";
        return false;
    }

    size_t nchan = 0;
    size_t samples = 0;
    WLEMData::ConstSPtr eeg;
    size_t nchanEEG = 0;
    if( m_pickEEG && emm->hasModality( WLEModality::EEG ) )
    {
        eeg = emm->getModality( WLEModality::EEG );
        nchanEEG = eeg->getNrChans();
        nchan += nchanEEG;
        samples = eeg->getSamplesPerChan();
    }
    WLEMData::ConstSPtr meg;
    size_t nchanMEG = 0;
    if( m_pickMEG && emm->hasModality( WLEModality::MEG ) )
    {
        meg = emm->getModality( WLEModality::MEG );
        nchanMEG = meg->getNrChans();
        nchan += nchanMEG;
        if( samples > 0 && samples != meg->getSamplesPerChan() )
        {
            wlog::error( CLASS ) << "Samples are not equal!";
            return false;
        }
        else
        {
            samples = meg->getSamplesPerChan();
        }
    }

    WLEMMeasurement::EDataT* stim = NULL;
    size_t nchanStim = 0;
    if( m_pickStim )
    {
        stim = emm->getEventChannels().get(); // dirty hack to get pointer
        nchanStim = stim->size();
        nchan += nchanStim;
        if( samples > 0 && samples != stim->at( 0 ).size() )
        {
            wlog::error( CLASS ) << "Samples are not equal!";
            return false;
        }
        else
        {
            samples = stim->at( 0 ).size();
        }
    }

    WLEMData::DataT data( nchan, samples );
    size_t offsetChan = 0;
    if( nchanEEG > 0 )
    {
        wlog::debug( CLASS ) << "Writing EEG (" << nchanEEG << ") ...";
        data.block( offsetChan, 0, nchanEEG, samples ) = eeg->getData();
        offsetChan += nchanEEG;
    }
    if( nchanMEG > 0 )
    {
        wlog::debug( CLASS ) << "Writing MEG (" << nchanMEG << ") ...";
        data.block( offsetChan, 0, nchanMEG, samples ) = meg->getData();
        offsetChan += nchanMEG;
    }
    if( nchanStim > 0 )
    {
        wlog::debug( CLASS ) << "Writing stimuli (" << nchanStim << ") ...";
        for( size_t c = 0; c < nchanStim; ++c )
        {
            for( size_t t = 0; t < samples; ++t )
            {
                data( c + offsetChan, t ) = ( *stim )[c][t];
            }
        }
    }

    RowVectorXd cals( nchan );
    cals.setOnes();

#ifndef LABP_FLOAT_COMPUTATION
    return m_fiffStream->write_raw_buffer( data, cals );
#else
    return m_fiffStream->write_raw_buffer( data.cast< double >(), cals );
#endif
}

void WWriterFiff::setChannelInfo( FIFFLIB::FiffChInfo* const chInfo )
{
    chInfo->scanno = 1; // TODO
    chInfo->range = 1.0f;
    chInfo->cal = 1.0;
}

void WWriterFiff::setChannelInfo( QList< FIFFLIB::FiffChInfo >* const chs, const WLEMMeasurement::EDataT* const stim )
{
    std::stringstream sstream;
    for( size_t c = 0; c < stim->size(); ++c )
    {
        FiffChInfo chInfo;
        setChannelInfo( &chInfo );
        chInfo.coil_type = FIFFV_COIL_NONE;
        chInfo.kind = FIFFV_STIM_CH;
        sstream << "STI" << std::setw( 3 ) << std::setfill( '0' ) << c;
        chInfo.ch_name = QString::fromStdString( sstream.str() );
        sstream.str( "" );
        sstream.clear();
        chs->append( chInfo );
    }
}

void WWriterFiff::setChannelInfo( QList< FIFFLIB::FiffChInfo >* const chs, const WLEMDEEG* const eeg )
{
    const std::vector< std::string >& chNames = *eeg->getChanNames();
    const std::vector< WPosition >& pos = *eeg->getChannelPositions3d();
    for( size_t c = 0; c < eeg->getNrChans(); ++c )
    {
        FiffChInfo chInfo;
        const WPosition p = pos.at( c );
        const Vector3d v( p.x(), p.y(), p.z() );
        setChannelInfo( &chInfo );
        chInfo.coil_type = FIFFV_COIL_EEG;
        chInfo.kind = FIFFV_EEG_CH;
        chInfo.eeg_loc.col( 0 ) = v;
        chInfo.eeg_loc.col( 1 ) = v;
        chInfo.loc.setZero();
        chInfo.loc( 0, 0 ) = v.x();
        chInfo.loc( 1, 0 ) = v.y();
        chInfo.loc( 2, 0 ) = v.z();
        chInfo.ch_name = QString::fromStdString( chNames[c] );
        chs->append( chInfo );
    }
}

void WWriterFiff::setChannelInfo( QList< FIFFLIB::FiffChInfo >* const chs, const WLEMDMEG* const meg )
{
    const std::vector< std::string >& chNames = *meg->getChanNames();
    const std::vector< WPosition >& pos = *meg->getChannelPositions3d();
    for( size_t c = 0; c < meg->getNrChans(); ++c )
    {
        FiffChInfo chInfo;
        setChannelInfo( &chInfo );
        const WPosition p = pos.at( c );
        const Vector3d v( p.x(), p.y(), p.z() );
        chInfo.loc.setZero();
        chInfo.loc( 0, 0 ) = v.x();
        chInfo.loc( 1, 0 ) = v.y();
        chInfo.loc( 2, 0 ) = v.z();
        chInfo.kind = FIFFV_MEG_CH;
        if( ( c + 1 ) % 3 == 0 )
        {
            chInfo.coil_type = FIFFV_COIL_VV_MAG_T1;
        }
        else
        {
            chInfo.coil_type = FIFFV_COIL_VV_PLANAR_T1;
        }
        chInfo.ch_name = QString::fromStdString( chNames[c] );
        chs->append( chInfo );
    }
}
