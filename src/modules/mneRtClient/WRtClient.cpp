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
#include "core/util/WLGeometry.h"
#include "WRtClient.h"

const std::string WRtClient::CLASS = "WRtClient";

WRtClient::WRtClient( const std::string& ip_address, const std::string& alias ) :
                m_ipAddress( ip_address ), m_alias( alias )
{
    m_isStreaming = false;
    m_isConnected = false;
    m_clientId = -1;
    m_devToHead = WLMatrix4::Matrix4T::Zero();
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

    // Set buffer size
    // TODO(pieloth): set variable block size.
    wlog::debug( CLASS ) << "Set buffer size.";
    ( *m_rtCmdClient )["bufsize"].pValues()[0] = QVariant( 500 );
    ( *m_rtCmdClient )["bufsize"].send();

    // Request measurement information
    wlog::debug( CLASS ) << "Requesting measurement information.";
    ( *m_rtCmdClient )["measinfo"].pValues()[0] = QVariant( m_clientId );
    ( *m_rtCmdClient )["measinfo"].send();

    wlog::debug( CLASS ) << "Read measurement information.";
    m_fiffInfo = m_rtDataClient->readInfo();
    wlog::info( CLASS ) << "Measurement information received.";

    m_picksMeg = m_fiffInfo->pick_types( true, false, false );
    wlog::debug( CLASS ) << "picks meg: " << m_picksMeg.size();
    if( m_picksMeg.size() > 2 )
    {
        wlog::debug( CLASS ) << "values: " << m_picksMeg[0] << ", " << m_picksMeg[1] << ", " << m_picksMeg[2] << " ...";
    }

    m_picksEeg = m_fiffInfo->pick_types( false, true, false );
    wlog::debug( CLASS ) << "picks eeg: " << m_picksEeg.size();
    if( m_picksEeg.size() > 2 )
    {
        wlog::debug( CLASS ) << "values: " << m_picksEeg[0] << ", " << m_picksEeg[1] << ", " << m_picksEeg[2] << " ...";
    }

    m_picksStim = m_fiffInfo->pick_types( false, false, true );
    wlog::debug( CLASS ) << "picks stim: " << m_picksStim.size();
    if( m_picksStim.size() > 2 )
    {
        wlog::debug( CLASS ) << "values: " << m_picksStim[0] << ", " << m_picksStim[1] << ", " << m_picksStim[2] << " ...";
    }
    readInfo();
    readChannelNames();
    readChannelPositionsFaces();

    wlog::info( CLASS ) << "Prepare streaming finished.";

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

    QMap< qint32, QString > qMap;
    const int selCon = m_rtCmdClient->requestConnectors( qMap );
    QMapIterator< int, QString > itMap( qMap );
    while( itMap.hasNext() )
    {
        itMap.next();
        ( *conMap )[itMap.key()] = itMap.value().toStdString();
    }

    return selCon;
}
bool WRtClient::setConnector( int conId )
{
    if( !isConnected() )
    {
        wlog::error( CLASS ) << "Client not connected!";
        return false;
    }

    ( *m_rtCmdClient )["selcon"].pValues()[0] = QVariant( conId );
    ( *m_rtCmdClient )["selcon"].send();
    // TODO check if connector is set
    return true;
}

bool WRtClient::setSimulationFile( std::string fname )
{
    if( !isConnected() )
    {
        wlog::error( CLASS ) << "Client not connected!";
        return false;
    }

//    QString simFile = "/home/pieloth/EMM-Data_light/intershift/rawdir/is05a/is05a1.fif";
    QString simFile = QString::fromStdString( fname );
    wlog::info( CLASS ) << "Set simulation file: " << simFile.toStdString();
    ( *m_rtCmdClient )["simfile"].pValues()[0] = QVariant( simFile );
    ( *m_rtCmdClient )["simfile"].send();

    // TODO Check if file is set
    return true;
}

bool WRtClient::readData( WLEMMeasurement::SPtr emmIn )
{
    wlog::debug( CLASS ) << "readData() called!";
    if( !isConnected() )
    {
        wlog::error( CLASS ) << "Client not connected!";
        return false;
    }

    if( !m_digPoints.empty() )
    {
        emmIn->setDigPoints( m_digPoints );
    }
    if( !m_devToHead.isZero() )
    {
        emmIn->setDevToFidTransformation( m_devToHead );
    }

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

WLEMDEEG::SPtr WRtClient::readEEG( const Eigen::MatrixXf& rawData )
{
    wlog::debug( CLASS ) << "readEEG() called!";
    WLEMDEEG::SPtr eeg( new WLEMDEEG() );
    readEmd( eeg.get(), m_picksEeg, rawData );
    // TODO(pieloth): setter
    eeg->setChanNames( m_chNamesEeg );
    eeg->setChannelPositions3d( m_chPosEeg );
    eeg->setFaces( m_facesEeg );
    return eeg;
}

WLEMDMEG::SPtr WRtClient::readMEG( const Eigen::MatrixXf& rawData )
{
    wlog::debug( CLASS ) << "readMEG() called!";
    WLEMDMEG::SPtr meg( new WLEMDMEG() );
    readEmd( meg.get(), m_picksMeg, rawData );
    // TODO(pieloth): setter
    meg->setChanNames( m_chNamesMeg );
    return meg;
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

    emd->setSampFreq( m_fiffInfo->sfreq );
    return true;
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

bool WRtClient::readChannelNames()
{
    wlog::debug( CLASS ) << "readChannelNames() called!";

    const QStringList chNames = m_fiffInfo->ch_names;

    // EEG
    const Eigen::RowVectorXi::Index eegSize = m_picksEeg.size();
    if( eegSize > 0 )
    {
        WAssertDebug( eegSize <= chNames.size(), "More selected channels than in chNames!" );
        m_chNamesEeg.reset( new std::vector< std::string > );
        m_chNamesEeg->reserve( eegSize );
        for( Eigen::RowVectorXi::Index row = 0; row < eegSize; ++row )
        {
            WAssertDebug( m_picksEeg[row] < chNames.size(), "Selected channel index out of chNames boundary!" );
            m_chNamesEeg->push_back( chNames.at( ( int )m_picksEeg[row] ).toStdString() );
        }
    }

    // MEG
    const Eigen::RowVectorXi::Index megSize = m_picksMeg.size();
    if( megSize > 0 )
    {
        WAssertDebug( megSize <= chNames.size(), "More selected channels than in chNames!" );
        m_chNamesMeg.reset( new std::vector< std::string > );
        m_chNamesMeg->reserve( megSize );
        for( Eigen::RowVectorXi::Index row = 0; row < megSize; ++row )
        {
            WAssertDebug( m_picksMeg[row] < chNames.size(), "Selected channel index out of chNames boundary!" );
            m_chNamesMeg->push_back( chNames.at( ( int )m_picksMeg[row] ).toStdString() );
        }
    }

    return true;
}

bool WRtClient::readChannelPositionsFaces()
{
    wlog::debug( CLASS ) << "readChannelPositions() called!";

    QList< FIFFLIB::FiffChInfo > chInfos = m_fiffInfo->chs;
    // EEG
    const Eigen::RowVectorXi::Index eegSize = m_picksEeg.size();
    if( eegSize > 0 )
    {
        WAssertDebug( eegSize <= chInfos.size(), "More selected channels than in chNames!" );
        m_chPosEeg.reset( new std::vector< WPosition > );
        m_chPosEeg->reserve( eegSize );
        for( Eigen::RowVectorXi::Index row = 0; row < eegSize; ++row )
        {
            WAssertDebug( m_picksEeg[row] < chInfos.size(), "Selected channel index out of chInfos boundary!" );
            const Eigen::Matrix< double, 3, 2, Eigen::DontAlign >& chPos = chInfos.at( ( int )m_picksEeg[row] ).eeg_loc;
            const WPosition pos( chPos( 0, 0 ), chPos( 1, 0 ), chPos( 2, 0 ) );
            m_chPosEeg->push_back( pos );
        }

        m_facesEeg.reset( new std::vector< WVector3i > );
        WLGeometry::computeTriangulation( *m_facesEeg, *m_chPosEeg, -5 );
    }

    // TODO(pieloth): MEG
    return true;
}

void WRtClient::readInfo()
{
#if LABP_FLOAT_COMPUTATION
    m_devToHead = m_fiffInfo->dev_head_t.trans);
#else
    m_devToHead = m_fiffInfo->dev_head_t.trans.cast< WLMatrix4::Matrix4T::Scalar >();
#endif

    const QList< FIFFLIB::FiffDigPoint >& digs = m_fiffInfo->dig;
    m_digPoints.clear();
    m_digPoints.reserve( digs.size() );
    QList< FIFFLIB::FiffDigPoint >::ConstIterator it;
    for( it = digs.begin(); it != digs.end(); ++it )
    {
        const WPosition pos( it->r[0], it->r[1], it->r[2] );
        WLDigPoint dig( pos, it->kind, it->ident );
        m_digPoints.push_back( dig );
    }
}
