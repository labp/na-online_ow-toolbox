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

#include <string>
#include <vector>

#include <boost/shared_ptr.hpp>

#include <QtGlobal>
#include <QList>
#include <QMap>
#include <QString>

#include <fiff/fiff_ch_info.h>
#include <fiff/fiff_dig_point.h>

#include <core/common/WAssert.h>
#include <core/common/WLogger.h>
#include <core/common/math/linearAlgebra/WPosition.h>

#include "core/data/WLDataTypes.h"
#include "core/data/WLDigPoint.h"
#include "core/data/enum/WLEPointType.h"
#include "core/util/WLGeometry.h"
#include "WRtClient.h"

const std::string WRtClient::CLASS = "WRtClient";

WRtClient::WRtClient( const std::string& ip_address, const std::string& alias ) :
                m_ipAddress( ip_address ), m_alias( alias )
{
    m_isStreaming = false;
    m_isConnected = false;
    m_clientId = -1;
    m_conSelected = -1;
    m_blockSize = 500;

    m_chPosEeg.reset( new ChannelsPositionsT );
    m_facesEeg.reset( new FacesT );
    m_digPoints = WLList< WLDigPoint >::instance();
}

WRtClient::~WRtClient()
{
    disconnect();
}

bool WRtClient::connect()
{
    wlog::debug( CLASS ) << "connect() called!";

    QString ip_address = QString::fromStdString( m_ipAddress );
    wlog::info( CLASS ) << "Connecting to " << m_ipAddress;

    m_rtCmdClient.reset( new RTCLIENTLIB::RtCmdClient() );
    m_rtCmdClient->connectToHost( ip_address );
    if( m_rtCmdClient->waitForConnected( 300 ) )
    {
        wlog::debug( CLASS ) << "Requesting commands.";
        m_rtCmdClient->requestCommands();

        wlog::info( CLASS ) << "Command Client connected!";
    }

    m_rtDataClient.reset( new RTCLIENTLIB::RtDataClient() );
    m_rtDataClient->connectToHost( ip_address );
    if( m_rtDataClient->waitForConnected( 300 ) )
    {
        const QString rtClientAlias = QString::fromStdString( m_alias );
        m_rtDataClient->setClientAlias( rtClientAlias );
        wlog::info( CLASS ) << "Using client alias " << m_alias;

        m_clientId = m_rtDataClient->getClientId();
        wlog::info( CLASS ) << "Using client id " << m_clientId;

        wlog::info( CLASS ) << "Data Client connected!";
    }

    m_isConnected = m_rtCmdClient->state() == QAbstractSocket::ConnectedState
                    && m_rtDataClient->state() == QAbstractSocket::ConnectedState;

    if( !isConnected() )
    {
        disconnect();
    }

    return m_isConnected;
}

bool WRtClient::prepareStreaming()
{
    wlog::debug( CLASS ) << "prepareStreaming() called!";
    // Request measurement information
    wlog::debug( CLASS ) << "Requesting measurement information.";
    ( *m_rtCmdClient )["measinfo"].pValues()[0] = QVariant( m_clientId );
    ( *m_rtCmdClient )["measinfo"].send();

    wlog::debug( CLASS ) << "Read measurement information.";
    m_fiffInfo.clear();
    m_fiffInfo = m_rtDataClient->readInfo();
    if( m_fiffInfo.isNull() )
    {
        wlog::error( CLASS ) << "Measurement information could not received!";
        return false;
    }
    wlog::info( CLASS ) << "Measurement information received.";

    WLEMMeasurement* emm = new WLEMMeasurement();
    if( !preparePrototype( emm ) )
    {
        wlog::error( CLASS ) << "Could not read measurement information!";
        free( emm );
        return false;
    }
    m_emmPrototype.reset( emm );

    m_picksMeg = m_fiffInfo->pick_types( true, false, false );
    wlog::debug( CLASS ) << "picks meg: " << m_picksMeg.size();
    if( m_picksMeg.size() > 0 )
    {
        WLEMDMEG* meg = new WLEMDMEG();
        if( preparePrototype( meg, m_picksMeg ) )
        {
            m_megPrototype.reset( meg );
            if( m_picksMeg.size() > 2 )
            {
                wlog::debug( CLASS ) << "values: " << m_picksMeg[0] << ", " << m_picksMeg[1] << ", " << m_picksMeg[2] << " ...";
            }
        }
        else
        {
            wlog::error( CLASS ) << "Error on reading MEG information. Skip MEG data!";
            free( meg );
            m_picksMeg.resize( 0 );
        }
    }

    m_picksEeg = m_fiffInfo->pick_types( false, true, false );
    wlog::debug( CLASS ) << "picks eeg: " << m_picksEeg.size();
    if( m_picksEeg.size() > 0 )
    {
        WLEMDEEG* eeg = new WLEMDEEG();
        if( preparePrototype( eeg, m_picksEeg ) )
        {
            m_eegPrototype.reset( eeg );
            if( m_picksEeg.size() > 2 )
            {
                wlog::debug( CLASS ) << "values: " << m_picksEeg[0] << ", " << m_picksEeg[1] << ", " << m_picksEeg[2] << " ...";
            }
        }
        else
        {
            wlog::error( CLASS ) << "Error on reading EEG information. Skip EEG data!";
            free( eeg );
            m_picksEeg.resize( 0 );
        }
    }

    m_picksStim = m_fiffInfo->pick_types( false, false, true );
    wlog::debug( CLASS ) << "picks stim: " << m_picksStim.size();
    if( m_picksStim.size() > 2 )
    {
        wlog::debug( CLASS ) << "values: " << m_picksStim[0] << ", " << m_picksStim[1] << ", " << m_picksStim[2] << " ...";
    }

    return m_picksEeg.size() > 0 || m_picksMeg.size() > 0;
}

bool WRtClient::isConnected()
{
    return m_isConnected;
}

void WRtClient::disconnect()
{
    wlog::debug( CLASS ) << "disconnect() called!";
    stop();
    if( !m_rtDataClient.isNull() )
    {
        m_rtDataClient->disconnectFromHost();
        if( m_rtDataClient->state() == QAbstractSocket::UnconnectedState || m_rtDataClient->waitForDisconnected() )
        {
            wlog::info( CLASS ) << "Data Client disconnected!";
        }
    }

    if( !m_rtCmdClient.isNull() )
    {
        m_rtCmdClient->disconnectFromHost();
        if( m_rtCmdClient->state() == QAbstractSocket::UnconnectedState || m_rtCmdClient->waitForDisconnected() )
        {
            wlog::info( CLASS ) << "Command Client disconnected!";
        }
    }
    m_isConnected = false;
}

bool WRtClient::start()
{
    wlog::debug( CLASS ) << "start() called!";

    if( isStreaming() )
    {
        wlog::warn( CLASS ) << "Could start streaming. Server is already streaming!";
        return true;
    }

    if( !isConnected() )
    {
        wlog::error( CLASS ) << "Client not connected!";
        return false;
    }

    wlog::info( CLASS ) << "Prepare streaming.";
    if( prepareStreaming() )
    {
        wlog::info( CLASS ) << "Prepare streaming finished.";
    }
    else
    {
        wlog::error( CLASS ) << "Could not prepare steaming. Streaming is not started!";
        return false;
    }

    wlog::debug( CLASS ) << "Set buffer size.";
    ( *m_rtCmdClient )["bufsize"].pValues()[0] = QVariant( m_blockSize );
    ( *m_rtCmdClient )["bufsize"].send();

    wlog::debug( CLASS ) << "Start streaming ...";
    ( *m_rtCmdClient )["start"].pValues()[0] = QVariant( m_clientId );
    ( *m_rtCmdClient )["start"].send();
    // TODO(pieloth): Check if streaming has been started.

    m_isStreaming = true;

    return true;
}

bool WRtClient::stop()
{
    wlog::debug( CLASS ) << "stop() called!";

    if( !isConnected() )
    {
        wlog::error( CLASS ) << "Client not connected!";
        return true;
    }

    if( !isStreaming() )
    {
        wlog::warn( CLASS ) << "Server is not streaming!";
        return true;
    }

    ( *m_rtCmdClient )["stop-all"].send();
// TODO(pieloth): Check if streaming has been stopped.
    m_isStreaming = false;

    return true;
}

bool WRtClient::isStreaming()
{
    return m_isStreaming;
}

int WRtClient::getConnectors( std::map< int, std::string >* const conMap )
{
    if( !isConnected() )
    {
        wlog::error( CLASS ) << "Client not connected!";
        return false;
    }

    m_conMap.clear();
    QMap< qint32, QString > qMap;
    m_conSelected = m_rtCmdClient->requestConnectors( qMap );
    QMapIterator< int, QString > itMap( qMap );
    while( itMap.hasNext() )
    {
        itMap.next();
        ( *conMap )[itMap.key()] = itMap.value().toStdString();
        m_conMap[itMap.key()] = itMap.value().toStdString();
    }

    return m_conSelected;
}

bool WRtClient::setConnector( int conId )
{
    if( !isConnected() )
    {
        wlog::error( CLASS ) << "Client not connected!";
        return false;
    }

    if( m_conMap.find( conId ) != m_conMap.end() )
    {
        m_conSelected = conId;
        ( *m_rtCmdClient )["selcon"].pValues()[0] = QVariant( conId );
        ( *m_rtCmdClient )["selcon"].send();
        // TODO check if connector is set
        return true;
    }
    else
    {
        wlog::error( CLASS ) << "Could not find request connector!";
        return false;
    }
}

void WRtClient::setBlockSize( int blockSize )
{
    m_blockSize = blockSize;
}

int WRtClient::getBlockSize()
{
    return m_blockSize;
}

bool WRtClient::setSimulationFile( std::string fname )
{
    if( !isConnected() )
    {
        wlog::error( CLASS ) << "Client not connected!";
        return false;
    }

    std::map< int, std::string >::const_iterator it = m_conMap.find( m_conSelected );
    if( it != m_conMap.end() && it->second.find( "Simulator" ) != std::string::npos )
    {
        QString simFile = QString::fromStdString( fname );
        wlog::info( CLASS ) << "Set simulation file: " << simFile.toStdString();
        ( *m_rtCmdClient )["simfile"].pValues()[0] = QVariant( simFile );
        ( *m_rtCmdClient )["simfile"].send();
        return true;
    }
    else
    {
        wlog::warn( CLASS ) << "Skip simulation file, due to no simulation connector!";
        return false;
    }
}

bool WRtClient::readData( WLEMMeasurement::SPtr& emmIn )
{
    wlog::debug( CLASS ) << "readData() called!";
    if( !isConnected() )
    {
        wlog::error( CLASS ) << "Client not connected!";
        return false;
    }

    emmIn = m_emmPrototype->clone();

    FIFFLIB::fiff_int_t kind;
    Eigen::MatrixXf matRawBuffer;
    m_rtDataClient->readRawBuffer( m_fiffInfo->nchan, matRawBuffer, kind );

    if( kind == FIFF_DATA_BUFFER )
    {
        wlog::debug( CLASS ) << "matRawBuffer: " << matRawBuffer.rows() << "x" << matRawBuffer.cols();

        WLEMData::SPtr emd;
        if( m_picksEeg.size() > 0 )
        {
            emd = readEEG( matRawBuffer );
            emmIn->addModality( emd );
        }
        if( m_picksMeg.size() > 0 )
        {
            emd = readMEG( matRawBuffer );
            emmIn->addModality( emd );
        }
        if( m_picksStim.size() > 0 )
        {
            boost::shared_ptr< WLEMMeasurement::EDataT > events = readEvents( matRawBuffer );
            emmIn->setEventChannels( events );
        }
        return true;
    }
    else
    {
        if( kind == FIFF_BLOCK_END )
        {
            m_isStreaming = false;
            return false;
        }
        return false;
    }
}

bool WRtClient::readEmd( WLEMData* const emd, const Eigen::RowVectorXi& picks, const Eigen::MatrixXf& rawData )
{
    if( picks.size() == 0 )
    {
        wlog::error( CLASS ) << "No channels to pick!";
        return false;
    }

    const Eigen::RowVectorXi::Index rows = picks.size();
    const Eigen::MatrixXf::Index cols = rawData.cols();

    WLEMData::DataT& emdData = emd->getData();
    emdData.resize( rows, cols );
    WAssertDebug( rows <= rawData.rows(), "More selected channels than in raw data!" );

    for( Eigen::RowVectorXi::Index row = 0; row < rows; ++row )
    {
        WAssertDebug( picks[row] < rawData.rows(), "Selected channel index out of raw data boundary!" );
        for( Eigen::RowVectorXi::Index col = 0; col < cols; ++col )
        {
            emdData( row, col ) = ( WLEMData::ScalarT )rawData( picks[row], col );
        }
    }

    return true;
}

WLEMDEEG::SPtr WRtClient::readEEG( const Eigen::MatrixXf& rawData )
{
    wlog::debug( CLASS ) << "readEEG() called!";
    WLEMDEEG::SPtr eeg = m_eegPrototype->clone()->getAs< WLEMDEEG >();
    readEmd( eeg.get(), m_picksEeg, rawData );
    return eeg;
}

WLEMDMEG::SPtr WRtClient::readMEG( const Eigen::MatrixXf& rawData )
{
    wlog::debug( CLASS ) << "readMEG() called!";
    WLEMDMEG::SPtr meg = m_megPrototype->clone()->getAs< WLEMDMEG >();
    readEmd( meg.get(), m_picksMeg, rawData );
    return meg;
}

boost::shared_ptr< WLEMMeasurement::EDataT > WRtClient::readEvents( const Eigen::MatrixXf& rawData )
{
    wlog::debug( CLASS ) << "readStim() called!";

    boost::shared_ptr< WLEMMeasurement::EDataT > events( new WLEMMeasurement::EDataT );
    if( m_picksStim.size() == 0 )
    {
        wlog::error( CLASS ) << "No channels to pick!";
        return events;
    }

    const Eigen::RowVectorXi::Index rows = m_picksStim.size();
    const Eigen::MatrixXf::Index cols = rawData.cols();

    events->clear();
    events->reserve( rows );
    WAssertDebug( rows <= rawData.rows(), "More selected channels than in raw data!" );

    for( Eigen::RowVectorXi::Index row = 0; row < rows; ++row )
    {
        WAssertDebug( m_picksStim[row] < rawData.rows(), "Selected channel index out of raw data boundary!" );
        WLEMMeasurement::EChannelT eChannel;
        eChannel.reserve( cols );
        for( Eigen::RowVectorXi::Index col = 0; col < cols; ++col )
        {
            eChannel.push_back( ( WLEMMeasurement::EventT )rawData( m_picksStim[row], col ) );
        }
        events->push_back( eChannel );
    }

    return events;
}

bool WRtClient::setDigPointsAndEEG( const std::list< WLDigPoint >& digPoints )
{
    wlog::info( CLASS ) << "Set user-defined digitization points and EEG positions.";
    m_digPoints = WLList< WLDigPoint >::instance( digPoints );
    wlog::info( CLASS ) << "Digitization points: " << m_digPoints->size();

    m_chPosEeg->clear();
    m_chPosEeg->reserve( m_digPoints->size() );
    std::list< WLDigPoint >::const_iterator it;
    bool isFirst = true;
    for( it = m_digPoints->begin(); it != m_digPoints->end(); ++it )
    {
        if( it->getKind() != WLEPointType::EEG_ECG )
        {
            continue;
        }

        if( !isFirst )
        {
            m_chPosEeg->push_back( it->getPoint() );
        }
        else
        {
            isFirst = false;
            continue;
        }
    }

    wlog::info( CLASS ) << "EEG positions from digPoints: " << m_chPosEeg->size();

    m_facesEeg->clear();
    if( !WLGeometry::computeTriangulation( m_facesEeg.get(), *m_chPosEeg, -5 ) )
    {
        wlog::warn( CLASS ) << "Could not generate faces!";
    }
    else
    {
        wlog::info( CLASS ) << "EEG faces from digPoints: " << m_facesEeg->size();
    }

    if( !m_digPoints->empty() && !m_chPosEeg->empty() && !m_facesEeg->empty() )
    {
        return true;
    }
    else
    {
        return false;
    }
}

bool WRtClient::preparePrototype( WLEMMeasurement* const emm )
{
#if LABP_FLOAT_COMPUTATION
    WLMatrix4::Matrix4T devToHead = m_fiffInfo->dev_head_t.trans;
#else
    WLMatrix4::Matrix4T devToHead = m_fiffInfo->dev_head_t.trans.cast< WLMatrix4::Matrix4T::Scalar >();
#endif
    if( !devToHead.isZero() )
    {
        emm->setDevToFidTransformation( devToHead );
    }

    if( m_digPoints->empty() )
    {
        const QList< FIFFLIB::FiffDigPoint >& digs = m_fiffInfo->dig;
        m_digPoints->clear();
        QList< FIFFLIB::FiffDigPoint >::ConstIterator it;
        for( it = digs.begin(); it != digs.end(); ++it )
        {
            const WPosition pos( it->r[0], it->r[1], it->r[2] );
            WLDigPoint dig( pos, it->kind, it->ident );
            m_digPoints->push_back( dig );
        }
        emm->setDigPoints( m_digPoints );
    }
    else
    {
        emm->setDigPoints( m_digPoints );
    }
    return true;
}

bool WRtClient::preparePrototype( WLEMData* const emd, const Eigen::RowVectorXi& picks )
{
    const FIFFLIB::FiffChInfo fiffInfo = m_fiffInfo->chs[picks[0]];
    emd->setSampFreq( m_fiffInfo->sfreq );
    emd->setChanUnit( WLEUnit::fromFIFF( fiffInfo.unit ) );
    emd->setChanUnitExp( WLEExponent::fromFIFF( fiffInfo.unit_mul ) );

    readChannelNames( emd, picks );
    if( readChannelPositions( emd, picks ) )
    {
        readChannelFaces( emd, picks );
        return true;
    }
    else
    {
        return false;
    }

}

bool WRtClient::preparePrototype( WLEMDEEG* const emd, const Eigen::RowVectorXi& picks )
{
    return preparePrototype( ( WLEMData* )emd, picks );
}

bool WRtClient::preparePrototype( WLEMDMEG* const emd, const Eigen::RowVectorXi& picks )
{
    return preparePrototype( ( WLEMData* )emd, picks );
}

bool WRtClient::readChannelNames( WLEMData* const emd, const Eigen::RowVectorXi& picks )
{
    wlog::debug( CLASS ) << "readChannelNames() called!";

    if( picks.size() > 0 )
    {
        const QStringList chNames = m_fiffInfo->ch_names;
        WAssertDebug( picks.size() <= chNames.size(), "More selected channels than in chNames!" );
        ChannelNamesSPtr names( new ChannelNamesT );
        names->reserve( picks.size() );
        for( Eigen::RowVectorXi::Index row = 0; row < picks.size(); ++row )
        {
            WAssertDebug( picks[row] < chNames.size(), "Selected channel index out of chNames boundary!" );
            names->push_back( chNames.at( ( int )picks[row] ).toStdString() );
        }
        emd->setChanNames( names );
        return true;
    }
    else
    {
        wlog::error( CLASS ) << "No picks!";
        return false;
    }
}

bool WRtClient::readChannelPositions( WLEMData* const emd, const Eigen::RowVectorXi& picks )
{
    wlog::debug( CLASS ) << "readChannelPositions() called!";

    QList< FIFFLIB::FiffChInfo > chInfos = m_fiffInfo->chs;
// EEG
    if( picks.size() > 0 )
    {
        ChannelsPositionsSPtr positions( new ChannelsPositionsT );
        positions->reserve( picks.size() );
        const WPosition zero = WPosition::zero();
        for( Eigen::RowVectorXi::Index row = 0; row < picks.size(); ++row )
        {
            WAssertDebug( picks[row] < chInfos.size(), "Selected channel index out of chInfos boundary!" );
            const Eigen::Matrix< double, 3, 2, Eigen::DontAlign >& chPos = chInfos.at( ( int )picks[row] ).eeg_loc;
            const WPosition pos( chPos( 0, 0 ), chPos( 1, 0 ), chPos( 2, 0 ) );
            positions->push_back( pos );
        }

        WLEMDEEG* eeg = dynamic_cast< WLEMDEEG* >( emd );
        if( eeg != NULL )
        {
            if( !m_chPosEeg->empty() )
            {
                wlog::info( CLASS ) << "Using user-defined EEG positions and faces.";
                eeg->setChannelPositions3d( m_chPosEeg );
                return true;
            }
            else
                if( !positions->empty() )
                {
                    eeg->setChannelPositions3d( positions );
                    return true;
                }
                else
                {
                    wlog::error( CLASS ) << "No EEG positions found!";
                    return false;
                }
        }

        WLEMDMEG* meg = dynamic_cast< WLEMDMEG* >( emd );
        if( meg != NULL )
        {
            if( !positions->empty() )
            {
                meg->setChannelPositions3d( positions );
                return true;
            }
            else
            {
                wlog::error( CLASS ) << "No MEG positions found!";
                return false;
            }
        }
        return false;
    }
    else
    {
        wlog::error( CLASS ) << "No picks!";
        return false;
    }
}

bool WRtClient::readChannelFaces( WLEMData* const emd, const Eigen::RowVectorXi& picks )
{
    wlog::debug( CLASS ) << "readChannelFaces() called!";

    ChannelsPositionsSPtr positions;
    WLEMDEEG* eeg = dynamic_cast< WLEMDEEG* >( emd );
    if( eeg != NULL )
    {
        positions = eeg->getChannelPositions3d();
    }

    WLEMDMEG* meg = dynamic_cast< WLEMDMEG* >( emd );
    if( meg != NULL )
    {
        positions = meg->getChannelPositions3d();
        wlog::warn( CLASS ) << "Skipping triangulation for MEG, due to segFault!";
        return false;
    }

    if( !positions || positions->empty() )
    {
        wlog::error( CLASS ) << "No positions available, skipping faces!";
        return false;
    }

    const WPosition zero = WPosition::zero();
    ChannelsPositionsT::const_iterator it;
    size_t nzero = 0;
    for( it = positions->begin(); it != positions->end(); ++it )
    {
        if( *it == zero )
        {
            ++nzero;
        }
    }

    FacesSPtr faces( new FacesT );
    if( nzero < 3 )
    {
        if( WLGeometry::computeTriangulation( faces.get(), *positions, -5 ) )
        {
            if( eeg != NULL )
            {
                eeg->setFaces( faces );
                return true;
            }

            if( meg != NULL )
            {
                meg->setFaces( faces );
                return true;
            }
            return false;
        }
        else
        {
            wlog::error( CLASS ) << "Could not generated faces!";
            return false;
        }
    }
    else
    {
        wlog::warn( CLASS ) << "Counted " << nzero
                        << " (0,0,0) position - assumed incorrect EEG positions! Triangulation is skipped!";
        return false;
    }
}
