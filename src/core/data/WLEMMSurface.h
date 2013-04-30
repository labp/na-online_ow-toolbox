// TODO license

#ifndef WLEMMSURFACE_H_
#define WLEMMSURFACE_H_

#include <vector>

#include <boost/shared_ptr.hpp>

#include <core/common/math/linearAlgebra/WPosition.h>
#include <core/common/math/linearAlgebra/WVectorFixed.h>

#include "WLEMMEnumTypes.h"

namespace LaBP
{
    class WLEMMSurface
    {
    public:
        /**
         * Abbreviation for a shared pointer.
         */
        typedef boost::shared_ptr< WLEMMSurface > SPtr;

        /**
         * Abbreviation for const shared pointer.
         */
        typedef boost::shared_ptr< const WLEMMSurface > ConstSPtr;

        struct Hemisphere
        {
            enum Enum
            {
                LEFT, RIGHT, BOTH
            };
        };

        WLEMMSurface();
        WLEMMSurface( boost::shared_ptr< std::vector< WPosition > > vertex, WEUnit::Enum vertexUnit,
                        WEExponent::Enum vertexExponent, boost::shared_ptr< std::vector< WVector3i > > faces,
                        Hemisphere::Enum hemisphere );

        WLEMMSurface( const WLEMMSurface& surface );

        ~WLEMMSurface();

        boost::shared_ptr< std::vector< WPosition > >  getVertex() const;
        void setVertex( boost::shared_ptr< std::vector< WPosition > > vertex );

        WEUnit::Enum getVertexUnit() const;
        void setVertexUnit( WEUnit::Enum unit );

        WEExponent::Enum getVertexExponent() const;
        void setVertexExponent( WEExponent::Enum exponent );

        std::vector< WVector3i >& getFaces() const;
        void setFaces( boost::shared_ptr< std::vector< WVector3i > > faces );

        Hemisphere::Enum getHemisphere() const;
        void setHemisphere( Hemisphere::Enum val );

    private:
        boost::shared_ptr< std::vector< WPosition > > m_vertex;

        Hemisphere::Enum m_hemisphere;

        WEUnit::Enum m_vertexUnit;
        WEExponent::Enum m_vertexExponent;

        boost::shared_ptr< std::vector< WVector3i > > m_faces;
    };
}

#endif  // WLEMMSURFACE_H_
