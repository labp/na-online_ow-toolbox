// TODO doc & license

#include <string>
#include <vector>

#include <boost/shared_ptr.hpp>

#include <core/common/WLogger.h>

#include "core/data/WLMatrixTypes.h"

#include "core/data/emd/WLEMD.h"
#include "WDataSetEMMEnumTypes.h"
#include "WDataSetEMMSource.h"



namespace LaBP
{

    const std::string WDataSetEMMSource::CLASS = "WDataSetEMMSource";

    WDataSetEMMSource::WDataSetEMMSource() :
                    WLEMD()
    {
    }

    WDataSetEMMSource::WDataSetEMMSource( const WDataSetEMMSource& source ) :
                    WLEMD( source )
    {
        m_originModalityType = source.m_originModalityType;
    }

    WDataSetEMMSource::WDataSetEMMSource( const WLEMD& emd ) :
                    WLEMD( emd )
    {
        // C++11 supports "delegating constructors". So default initialization could be moved to default constructor.
        m_chanNames.reset( new std::vector< std::string >() );
        m_originModalityType = emd.getModalityType();
    }

    WDataSetEMMSource::~WDataSetEMMSource()
    {
    }

    WEModalityType::Enum WDataSetEMMSource::getModalityType() const
    {
        return WEModalityType::SOURCE;
    }

    boost::shared_ptr< WDataSetEMMSource::DataT > WDataSetEMMSource::convertMatrix( const MatrixT& matrix )
    {
        boost::shared_ptr< WDataSetEMMSource::DataT > data( new DataT() );
        data->reserve( matrix.rows() );
        data->resize( matrix.rows() );

        for( MatrixT::Index row = 0; row < matrix.rows(); ++row )
        {
            data->at( row ).reserve( matrix.cols() );
            for( MatrixT::Index col = 0; col < matrix.cols(); ++col )
            {
                data->at( row ).push_back( matrix( row, col ) );
            }
        }

        return data;
    }

    WLEMD::SPtr WDataSetEMMSource::clone() const
    {
        WDataSetEMMSource::SPtr emd( new WDataSetEMMSource( *this ) );
        return emd;
    }

    WLEMD::DataT& WDataSetEMMSource::getData() const
    {
        wlog::warn( CLASS )
                        << "Do not use getData() to retrieve source reconstruction data! Due to performance issues, use getMatrix() instead.";
        return *m_data;
    }

    size_t WDataSetEMMSource::getNrChans() const
    {
        return static_cast< size_t >( m_matrix->rows() );
    }

    size_t WDataSetEMMSource::getSamplesPerChan() const
    {
        return static_cast< size_t >( m_matrix->cols() );
    }

    MatrixT& WDataSetEMMSource::getMatrix() const
    {
        return *m_matrix;
    }

    void WDataSetEMMSource::setMatrix( boost::shared_ptr< MatrixT > matrix )
    {
        m_data.reset();
        m_matrix = matrix;
    }

    LaBP::WEModalityType::Enum WDataSetEMMSource::getOriginModalityType() const
    {
        return m_originModalityType;
    }

    void WDataSetEMMSource::setOriginModalityType( LaBP::WEModalityType::Enum modality )
    {
        m_originModalityType = modality;
    }

} /* namespace LaBP */
