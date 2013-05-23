// TODO doc & license

#ifndef WLEMDSOURCE_H_
#define WLEMDSOURCE_H_

#include <string>

#include <boost/shared_ptr.hpp>

#include "core/data/WLMatrixTypes.h"

#include "core/data/emd/WLEMData.h"

class WLEMDSource: public WLEMData
{
public:
    /**
     * Abbreviation for a shared pointer.
     */
    typedef boost::shared_ptr< WLEMDSource > SPtr;

    /**
     * Abbreviation for const shared pointer.
     */
    typedef boost::shared_ptr< const WLEMDSource > ConstSPtr;

    static const std::string CLASS;

    WLEMDSource();

    explicit WLEMDSource( const WLEMDSource& source );

    explicit WLEMDSource( const WLEMData& emd );

    virtual ~WLEMDSource();

    virtual WLEMData::SPtr clone() const;

    virtual DataT& getData() const;

    virtual size_t getNrChans() const;

    virtual size_t getSamplesPerChan() const;

    virtual LaBP::WEModalityType::Enum getModalityType() const;

    LaBP::WEModalityType::Enum getOriginModalityType() const;

    void setOriginModalityType( LaBP::WEModalityType::Enum modality );

    LaBP::MatrixT& getMatrix() const;

    void setMatrix( boost::shared_ptr< LaBP::MatrixT > matrix );

    static boost::shared_ptr< DataT > convertMatrix( const LaBP::MatrixT& matrix );

private:
    LaBP::WEModalityType::Enum m_originModalityType;
    boost::shared_ptr< LaBP::MatrixT > m_matrix;
};

#endif  // WLEMDSOURCE_H_
