/**
 * TODO license and documentation
 */

#ifndef WREGISTRATIONNAIVE_HPP_
#define WREGISTRATIONNAIVE_HPP_

#include <string>

#include <boost/shared_ptr.hpp>

#include "WRegistration.h"

class WRegistrationNaive: public WRegistration
{
public:
    /**
     * Abbreviation for a shared pointer.
     */
    typedef boost::shared_ptr< WRegistrationNaive > SPtr;

    /**
     * Abbreviation for const shared pointer.
     */
    typedef boost::shared_ptr< const WRegistrationNaive > ConstSPtr;

    typedef double Angle;

    static const std::string CLASS;

    WRegistrationNaive();

    double getTranslationFactor();
    void setTranslationFactor( double val );

    double getRotationStepSize();
    void setRoationStepSize( double val );

    double getErrorDelta();
    void setErrorDelta( double val );

    void reset();

    virtual double compute( const PointCloud& from, const PointCloud& to );

    virtual MatrixTransformation getTransformationMatrix() const;

    MatrixRotation getRotationXYZ();
    double getRotationX();
    double getRotationY();
    double getRotationZ();

    Vector getTranslation();

private:
    struct Operation
    {
        enum Enum
        {
            ADD, SUB
        };
    };

    double errorFct( const PointCloud& P, const PointCloud& X );

    double compute( const PointCloud& FROM, const PointCloud& TO, const double TRANSLATION_FACTOR, const double ROTATION_STEP,
                    const double ERROR_DELTA );

    double rotate( Angle& rotAct, const Angle rotX, const Angle rotY, const Angle rotZ, const PointCloud& FROM,
                    const PointCloud& TO, double error, double step );

    double translateCom( Vector& translation, PointCloud& translated, const PointCloud& FROM, const PointCloud& TO, double error,
                    double factor );

    double m_rotationStep;

    double m_translationFactor;

    double m_errorDelta;

    double m_rotX, m_rotY, m_rotZ;

    Vector m_translation;
};

#endif /* WREGISTRATIONNAIVE_HPP_ */
