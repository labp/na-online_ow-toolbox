// TODO doc & license

#include <string>
#include <vector>

#include <boost/shared_ptr.hpp>

#include <core/common/WLogger.h>

#include "core/data/WLMatrixTypes.h"

#include "core/data/WLEMMEnumTypes.h"
#include "WLEMData.h"
#include "WLEMDSource.h"

const std::string WLEMDSource::CLASS = "WDataSetEMMSource";

WLEMDSource::WLEMDSource() :
                WLEMData()
{
}

WLEMDSource::WLEMDSource( const WLEMDSource& source ) :
                WLEMData( source )
{
    m_originModalityType = source.m_originModalityType;
}

WLEMDSource::WLEMDSource( const WLEMData& emd ) :
                WLEMData( emd )
{
    // C++11 supports "delegating constructors". So default initialization could be moved to default constructor.
    m_chanNames.reset( new std::vector< std::string >() );
    m_originModalityType = emd.getModalityType();
}

WLEMDSource::~WLEMDSource()
{
}

LaBP::WEModalityType::Enum WLEMDSource::getModalityType() const
{
    return LaBP::WEModalityType::SOURCE;
}

boost::shared_ptr< WLEMDSource::DataT > WLEMDSource::convertMatrix( const LaBP::MatrixT& matrix )
{
    boost::shared_ptr< WLEMDSource::DataT > data( new DataT() );
    data->reserve( matrix.rows() );
    data->resize( matrix.rows() );

    for( LaBP::MatrixT::Index row = 0; row < matrix.rows(); ++row )
    {
        data->at( row ).reserve( matrix.cols() );
        for( LaBP::MatrixT::Index col = 0; col < matrix.cols(); ++col )
        {
            data->at( row ).push_back( matrix( row, col ) );
        }
    }

    return data;
}

WLEMData::SPtr WLEMDSource::clone() const
{
    WLEMDSource::SPtr emd( new WLEMDSource( *this ) );
    return emd;
}

WLEMData::DataT& WLEMDSource::getData() const
{
    wlog::warn( CLASS )
                    << "Do not use getData() to retrieve source reconstruction data! Due to performance issues, use getMatrix() instead.";
    return *m_data;
}

size_t WLEMDSource::getNrChans() const
{
    return static_cast< size_t >( m_matrix->rows() );
}

size_t WLEMDSource::getSamplesPerChan() const
{
    return static_cast< size_t >( m_matrix->cols() );
}

LaBP::MatrixT& WLEMDSource::getMatrix() const
{
    return *m_matrix;
}

void WLEMDSource::setMatrix( boost::shared_ptr< LaBP::MatrixT > matrix )
{
    m_data.reset();
    m_matrix = matrix;
}

LaBP::WEModalityType::Enum WLEMDSource::getOriginModalityType() const
{
    return m_originModalityType;
}

void WLEMDSource::setOriginModalityType( LaBP::WEModalityType::Enum modality )
{
    m_originModalityType = modality;
}
