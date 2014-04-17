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

#include "core/common/WLogger.h"

#include "WFiffTag.h"
#include "WFiffStream.h"

using namespace std;
using namespace FIFFLIB;

const std::string WFiffStream::CLASS = "WFiffStream";

WFiffStream::WFiffStream( QIODevice *p_pIODevice ) :
                FiffStream( p_pIODevice )
{

}

bool WFiffStream::read_meas_info( const FiffDirTree& p_Node, FiffInfo& info, FiffDirTree& p_NodeInfo )
{
    info.clear();

    //
    //   Find the desired blocks
    //
    QList< FiffDirTree > meas = p_Node.dir_tree_find( FIFFB_MEAS );

    if( meas.size() == 0 )
    {
        wlog::error( CLASS ) << "Could not find measurement data.";
        return false;
    }
    //
    QList< FiffDirTree > meas_info = meas[0].dir_tree_find( FIFFB_MEAS_INFO );
    if( meas_info.count() == 0 )
    {
        wlog::error( CLASS ) << "Could not find measurement info.";

        return false;
    }

    //
    //   Read measurement info
    //
    WFiffTag::SPtr t_pTag;

    fiff_int_t nchan = -1;
    float sfreq = -1.0f;
    QList< FiffChInfo > chs;
    float lowpass = -1.0f;
    float highpass = -1.0f;

    FiffChInfo t_chInfo;

    FiffCoordTrans cand;    // = NULL;
    FiffCoordTrans dev_head_t;    // = NULL;
    FiffCoordTrans ctf_head_t;    // = NULL;

    fiff_int_t meas_date[2];
    meas_date[0] = -1;
    meas_date[1] = -1;

    fiff_int_t kind = -1;
    fiff_int_t pos = -1;

    for( qint32 k = 0; k < meas_info[0].nent; ++k )
    {
        kind = meas_info[0].dir[k].kind;
        pos = meas_info[0].dir[k].pos;
        switch( kind )
        {
            case FIFF_NCHAN:
                WFiffTag::read_tag( this, t_pTag, pos );
                nchan = *t_pTag->toInt();
                break;
            case FIFF_SFREQ:
                WFiffTag::read_tag( this, t_pTag, pos );
                sfreq = *t_pTag->toFloat();
                break;
            case FIFF_CH_INFO:
                WFiffTag::read_tag( this, t_pTag, pos );
                chs.append( t_pTag->toChInfo() );
                break;
            case FIFF_LOWPASS:
                WFiffTag::read_tag( this, t_pTag, pos );
                lowpass = *t_pTag->toFloat();
                break;
            case FIFF_HIGHPASS:
                WFiffTag::read_tag( this, t_pTag, pos );
                highpass = *t_pTag->toFloat();
                break;
            case FIFF_MEAS_DATE:
                WFiffTag::read_tag( this, t_pTag, pos );
                meas_date[0] = t_pTag->toInt()[0];
                meas_date[1] = t_pTag->toInt()[1];
                break;
            case FIFF_COORD_TRANS:
                WFiffTag::read_tag( this, t_pTag, pos );
                cand = t_pTag->toCoordTrans();
                if( cand.from == FIFFV_COORD_DEVICE && cand.to == FIFFV_COORD_HEAD )
                    dev_head_t = cand;
                else
                    if( cand.from == FIFFV_MNE_COORD_CTF_HEAD && cand.to == FIFFV_COORD_HEAD )
                        ctf_head_t = cand;
                break;
        }
    }
    //
    //   Check that we have everything we need
    //
    if( nchan < 0 )
    {
        wlog::error( CLASS ) << "Number of channels in not defined.";
        return false;
    }
    if( sfreq < 0 )
    {
        wlog::error( CLASS ) << "Sampling frequency is not defined.";
        return false;
    }
    if( chs.size() == 0 )
    {
        wlog::error( CLASS ) << "Channel information not defined.";
        return false;
    }
    if( chs.size() != nchan )
    {
        wlog::error( CLASS ) << "Incorrect number of channel definitions found.";
        return false;
    }

    if( dev_head_t.isEmpty() || ctf_head_t.isEmpty() )
    {
        QList< FiffDirTree > hpi_result = meas_info[0].dir_tree_find( FIFFB_HPI_RESULT );
        if( hpi_result.size() == 1 )
        {
            for( qint32 k = 0; k < hpi_result[0].nent; ++k )
            {
                kind = hpi_result[0].dir[k].kind;
                pos = hpi_result[0].dir[k].pos;
                if( kind == FIFF_COORD_TRANS )
                {
                    WFiffTag::read_tag( this, t_pTag, pos );
                    cand = t_pTag->toCoordTrans();
                    if( cand.from == FIFFV_COORD_DEVICE && cand.to == FIFFV_COORD_HEAD )
                        dev_head_t = cand;
                    else
                        if( cand.from == FIFFV_MNE_COORD_CTF_HEAD && cand.to == FIFFV_COORD_HEAD )
                            ctf_head_t = cand;
                }
            }
        }
    }
    //
    //   Locate the Polhemus data
    //
    QList< FiffDirTree > isotrak = meas_info[0].dir_tree_find( FIFFB_ISOTRAK );

    QList< FiffDigPoint > dig;
    fiff_int_t coord_frame = FIFFV_COORD_HEAD;
    FiffCoordTrans dig_trans;
    qint32 k = 0;

    if( isotrak.size() == 1 )
    {
        for( k = 0; k < isotrak[0].nent; ++k )
        {
            kind = isotrak[0].dir[k].kind;
            pos = isotrak[0].dir[k].pos;
            if( kind == FIFF_DIG_POINT )
            {
                WFiffTag::read_tag( this, t_pTag, pos );
                dig.append( t_pTag->toDigPoint() );
            }
            else
            {
                if( kind == FIFF_MNE_COORD_FRAME )
                {
                    WFiffTag::read_tag( this, t_pTag, pos );
                    coord_frame = *t_pTag->toInt();
                }
                else
                    if( kind == FIFF_COORD_TRANS )
                    {
                        WFiffTag::read_tag( this, t_pTag, pos );
                        dig_trans = t_pTag->toCoordTrans();
                    }
            }
        }
    }
    for( k = 0; k < dig.size(); ++k )
        dig[k].coord_frame = coord_frame;

    if( !dig_trans.isEmpty() ) //if exist('dig_trans','var')
        if( dig_trans.from != coord_frame && dig_trans.to != coord_frame )
            dig_trans.clear();

    //
    //   Locate the acquisition information
    //
    QList< FiffDirTree > acqpars = meas_info[0].dir_tree_find( FIFFB_DACQ_PARS );
    QString acq_pars;
    QString acq_stim;
    if( acqpars.size() == 1 )
    {
        for( k = 0; k < acqpars[0].nent; ++k )
        {
            kind = acqpars[0].dir.at( k ).kind;
            pos = acqpars[0].dir.at( k ).pos;
            if( kind == FIFF_DACQ_PARS )
            {
                WFiffTag::read_tag( this, t_pTag, pos );
                acq_pars = t_pTag->toString();
            }
            else
                if( kind == FIFF_DACQ_STIM )
                {
                    WFiffTag::read_tag( this, t_pTag, pos );
                    acq_stim = t_pTag->toString();
                }
        }
    }
    // TODO(maschke): implement SSP data, CTF compensation data and bad channel list.
    //
    //   Load the SSP data
    //
    //QList<FiffProj> projs = this->read_proj(meas_info[0]);
    //
    //   Load the CTF compensation data
    //
    //QList<FiffCtfComp> comps = this->read_ctf_comp(meas_info[0], chs);
    //
    //   Load the bad channel list
    //
    //QStringList bads = this->read_bad_channels(p_Node);
    //
    //   Put the data together
    //
    if( p_Node.id.version != -1 )
        info.file_id = p_Node.id;
    else
        info.file_id.version = -1;

    //
    //  Make the most appropriate selection for the measurement id
    //
    if( meas_info[0].parent_id.version == -1 )
    {
        if( meas_info[0].id.version == -1 )
        {
            if( meas[0].id.version == -1 )
            {
                if( meas[0].parent_id.version == -1 )
                    info.meas_id = info.file_id;
                else
                    info.meas_id = meas[0].parent_id;
            }
            else
                info.meas_id = meas[0].id;
        }
        else
            info.meas_id = meas_info[0].id;
    }
    else
        info.meas_id = meas_info[0].parent_id;

    if( meas_date[0] == -1 )
    {
        info.meas_date[0] = info.meas_id.time.secs;
        info.meas_date[1] = info.meas_id.time.usecs;
    }
    else
    {
        info.meas_date[0] = meas_date[0];
        info.meas_date[1] = meas_date[1];
    }

    info.nchan = nchan;
    info.sfreq = sfreq;
    if( highpass != -1.0f )
        info.highpass = highpass;
    else
        info.highpass = 0.0f;

    if( lowpass != -1.0f )
        info.lowpass = lowpass;
    else
        info.lowpass = info.sfreq / 2.0;

    //
    //   Add the channel information and make a list of channel names
    //   for convenience
    //
    info.chs = chs;
    for( qint32 c = 0; c < info.nchan; ++c )
        info.ch_names << info.chs[c].ch_name;

    //
    //  Add the coordinate transformations
    //
    info.dev_head_t = dev_head_t;
    info.ctf_head_t = ctf_head_t;
    if( ( !info.dev_head_t.isEmpty() ) && ( !info.ctf_head_t.isEmpty() ) )
    {
        info.dev_ctf_t = info.dev_head_t;
        info.dev_ctf_t.to = info.ctf_head_t.from;
        info.dev_ctf_t.trans = ctf_head_t.trans.inverse() * info.dev_ctf_t.trans;
    }
    else
        info.dev_ctf_t.clear();

    //
    //   All kinds of auxliary stuff
    //
    info.dig = dig;
    if( !dig_trans.isEmpty() )
        info.dig_trans = dig_trans;

    //info.bads  = bads;
    //info.projs = projs;
    //info.comps = comps;
    info.acq_pars = acq_pars;
    info.acq_stim = acq_stim;

    p_NodeInfo = meas[0];

    return true;
}
