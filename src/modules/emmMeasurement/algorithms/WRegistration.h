/**
 * TODO license and documentation
 */

#ifndef WREGISTRATION_H_
#define WREGISTRATION_H_

#include <vector>

#include <boost/shared_ptr.hpp>

#include <core/common/math/linearAlgebra/WMatrixFixed.h>
#include <core/common/math/linearAlgebra/WPosition.h>
#include <core/common/math/linearAlgebra/WVectorFixed.h>

class WRegistration
{
public:
    /**
     * Abbreviation for a shared pointer.
     */
    typedef boost::shared_ptr< WRegistration > SPtr;

    /**
     * Abbreviation for const shared pointer.
     */
    typedef boost::shared_ptr< const WRegistration > ConstSPtr;

    typedef WPosition Point;
    typedef WVector3d Vector;
    typedef WMatrix3d MatrixRotation;
    typedef WMatrix4f MatrixTransformation;
    typedef std::vector< Point > PointCloud;

    WRegistration();
    virtual ~WRegistration();

    virtual MatrixTransformation getTransformationMatrix() const = 0;

    virtual double compute( const PointCloud& from, const PointCloud& to ) = 0;

    static PointCloud closestPointCorresponces( const PointCloud& P, const PointCloud& X );

    static double meanSquareError( const PointCloud& P, const PointCloud& X );
};

#endif /* WREGISTRATION_H_ */
