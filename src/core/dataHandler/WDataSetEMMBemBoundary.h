// TODO license

#ifndef WDATASETEMMBEMBOUNDARY_H_
#define WDATASETEMMBEMBOUNDARY_H_

#include <vector>

#include <boost/shared_ptr.hpp>

#include <core/common/math/linearAlgebra/WPosition.h>
#include <core/common/math/linearAlgebra/WVectorFixed.h>

#include "WDataSetEMMEnumTypes.h"

namespace LaBP
{
    class WDataSetEMMBemBoundary
    {
    public:
        /**
         * Abbreviation for a shared pointer.
         */
        typedef boost::shared_ptr< WDataSetEMMBemBoundary > SPtr;

        /**
         * Abbreviation for const shared pointer.
         */
        typedef boost::shared_ptr< const WDataSetEMMBemBoundary > ConstSPtr;

        WDataSetEMMBemBoundary();
        ~WDataSetEMMBemBoundary();

        std::vector< WPosition >& getVertex() const;
        void setVertex( boost::shared_ptr< std::vector< WPosition > > vertex );

        WEUnit::Enum getVertexUnit() const;
        void setVertexUnit( WEUnit::Enum unit );

        WEExponent::Enum getVertexExponent() const;
        void setVertexExponent( WEExponent::Enum exponent );

        WEBemType::Enum getBemType() const;
        void setBemType( WEBemType::Enum exponent );

        std::vector< WVector3i >& getFaces() const;
        void setFaces( boost::shared_ptr< std::vector< WVector3i > > faces );

        float getConductivity() const;
        void setConductivity( float conductivity );

        WEUnit::Enum getConductivityUnit() const;
        void setConductivityUnit( WEUnit::Enum unit );

    private:
        boost::shared_ptr< std::vector< WPosition > > m_vertex;

        WEUnit::Enum m_vertexUnit;
        WEExponent::Enum m_vertexExponent;
        WEBemType::Enum m_bemType;

        boost::shared_ptr< std::vector< WVector3i > > m_faces;

        float m_conductivity;
        WEUnit::Enum m_conductivityUnit;
    };
}

#endif /* WDATASETEMMBEMBOUNDARY_H_ */
