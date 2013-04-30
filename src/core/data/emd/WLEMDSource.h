// TODO doc & license

#ifndef WLEMDSOURCE_H_
#define WLEMDSOURCE_H_

#include <string>

#include <boost/shared_ptr.hpp>

#include "core/data/WLMatrixTypes.h"

#include "core/data/emd/WLEMD.h"

namespace LaBP
{

    class WLEMDSource: public LaBP::WLEMD
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

        explicit WLEMDSource( const WLEMD& emd );

        virtual ~WLEMDSource();

        virtual WLEMD::SPtr clone() const;

        virtual DataT& getData() const;

        virtual size_t getNrChans() const;

        virtual size_t getSamplesPerChan() const;

        virtual LaBP::WEModalityType::Enum getModalityType() const;

        LaBP::WEModalityType::Enum getOriginModalityType() const;

        void setOriginModalityType( LaBP::WEModalityType::Enum modality );

        MatrixT& getMatrix() const;

        void setMatrix( boost::shared_ptr< MatrixT > matrix );

        static boost::shared_ptr< DataT > convertMatrix( const MatrixT& matrix );

    private:
        LaBP::WEModalityType::Enum m_originModalityType;
        boost::shared_ptr< MatrixT > m_matrix;
    };

} /* namespace LaBP */
#endif  // WLEMDSOURCE_H_
