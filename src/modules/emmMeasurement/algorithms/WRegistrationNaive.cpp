/**
 * TODO license and documentation
 */

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

#include <boost/bind.hpp>

#include "core/common/WAssert.h"
#include "core/common/WLogger.h"

#include "WGeometry.h"
#include "WRegistration.h"
#include "WRegistrationNaive.h"

using namespace std;

const string WRegistrationNaive::CLASS = "WRegistrationNaive";

WRegistrationNaive::WRegistrationNaive()
{
    reset();
}

void WRegistrationNaive::reset()
{
    m_translationFactor = 0.1;
    m_rotationStep = ( 2 * M_PI ) / 360; // 1 degree
    m_errorDelta = 0.0001;

    m_rotX = 0;
    m_rotY = 0;
    m_rotZ = 0;

    m_translation.x() = 0;
    m_translation.y() = 0;
    m_translation.z() = 0;
}

double WRegistrationNaive::getTranslationFactor()
{
    return this->m_translationFactor;
}

void WRegistrationNaive::setTranslationFactor( double val )
{
    this->m_translationFactor = val;
}

double WRegistrationNaive::getRotationStepSize()
{
    return this->m_rotationStep;
}

void WRegistrationNaive::setRoationStepSize( double val )
{
    this->m_rotationStep = val;
}

double WRegistrationNaive::getErrorDelta()
{
    return this->m_errorDelta;
}

void WRegistrationNaive::setErrorDelta( double val )
{
    this->m_errorDelta = val;
}

WRegistration::MatrixTransformation WRegistrationNaive::getTransformationMatrix() const
{
    const MatrixRotation rotation = WGeometry::getRotationXYZMatrix( m_rotX, m_rotY, m_rotZ );
    const Vector& translation = m_translation;

    WRegistration::MatrixTransformation transformation;

    transformation( 0, 0 ) = rotation( 0, 0 );
    transformation( 0, 1 ) = rotation( 0, 1 );
    transformation( 0, 2 ) = rotation( 0, 2 );
    transformation( 0, 3 ) = translation.x();

    transformation( 1, 0 ) = rotation( 1, 0 );
    transformation( 1, 1 ) = rotation( 1, 1 );
    transformation( 1, 2 ) = rotation( 1, 2 );
    transformation( 1, 3 ) = translation.y();

    transformation( 2, 0 ) = rotation( 2, 0 );
    transformation( 2, 1 ) = rotation( 2, 1 );
    transformation( 2, 2 ) = rotation( 2, 2 );
    transformation( 2, 3 ) = translation.z();

    transformation( 3, 0 ) = 0;
    transformation( 3, 1 ) = 0;
    transformation( 3, 2 ) = 0;
    transformation( 3, 3 ) = 1;

    return transformation;
}

WRegistrationNaive::MatrixRotation WRegistrationNaive::getRotationXYZ()
{
    return WGeometry::getRotationXYZMatrix( m_rotX, m_rotY, m_rotZ );
}

double WRegistrationNaive::getRotationX()
{
    return this->m_rotX;
}

double WRegistrationNaive::getRotationY()
{
    return m_rotY;
}

double WRegistrationNaive::getRotationZ()
{
    return m_rotZ;
}

WRegistrationNaive::Vector WRegistrationNaive::getTranslation()
{
    return this->m_translation;
}

double WRegistrationNaive::compute( const PointCloud& from, const PointCloud& to )
{
    double error;
    if( from.size() <= to.size() )
    {
        error = compute( from, to, m_translationFactor, m_rotationStep, m_errorDelta );
    }
    else
    {
        wlog::info( CLASS ) << "compute() swap from <> to, due to better alignment.";
        error = compute( to, from, m_translationFactor, m_rotationStep, m_errorDelta );

        m_translation = m_translation * -1.0;
        m_rotX *= -1.0;
        m_rotY *= -1.0;
        m_rotZ *= -1.0;
    }
    return error;
}

double WRegistrationNaive::compute( const PointCloud& FROM, const PointCloud& TO, const double TRANSLATION_FACTOR,
                const double ROTATION_STEP, const double ERROR_DELTA )
{
    wlog::debug( CLASS ) << "compute() called! t_factor=" << TRANSLATION_FACTOR << " r_step=" << ROTATION_STEP << " e_delta="
                    << ERROR_DELTA;
    PointCloud translated;
    translated.reserve( FROM.size() );

    Vector trans;
    trans.x() = 0;
    trans.y() = 0;
    trans.z() = 0;

    Angle rotX = 0 * M_PI;
    Angle rotY = 0 * M_PI;
    Angle rotZ = 0 * M_PI;

    double mse = errorFct( FROM, closestPointCorresponces( FROM, TO ) );
    double mseOld = numeric_limits< double >::max();
    double count = 0;
    double step;
    double factor;

    do
    {
        ++count;
        mseOld = mse;
        step = ROTATION_STEP / pow( 10.0, ( count - 1 ) );
        factor = TRANSLATION_FACTOR / pow( 10.0, ( count - 1 ) );
        wlog::debug( CLASS ) << "Round #" << count << " started ...";
        wlog::debug( CLASS ) << "rotation step: " << step;
        wlog::debug( CLASS ) << "translation factor: " << factor;

        mse = translateCom( trans, translated, FROM, TO, mse, factor );
        mse = rotate( rotX, rotX, rotY, rotZ, translated, TO, mse, step );
        mse = rotate( rotY, rotX, rotY, rotZ, translated, TO, mse, step );
        mse = rotate( rotZ, rotX, rotY, rotZ, translated, TO, mse, step );

        wlog::debug( CLASS ) << " Round #" << count << " finished with error=" << mse << ".";
    } while( abs( mse - mseOld ) > ERROR_DELTA );

    m_translation = trans;
    m_rotX = rotX;
    m_rotY = rotY;
    m_rotZ = rotZ;

    return mse;
}

double WRegistrationNaive::rotate( Angle& rotAct, const Angle rotX, const Angle rotY, const Angle rotZ, const PointCloud& FROM,
                const PointCloud& TO, double error, double step )
{
    wlog::debug( CLASS ) << "rotate() called! error=" << error;

    MatrixRotation rotMat;
    PointCloud rotated;
    rotated.reserve( FROM.size() );

    // find most promising direction
    double errAdd, errSub, angleOld;
    angleOld = rotAct;

    rotAct = angleOld + step;
    rotated.clear();
    rotated.resize( FROM.size() );
    rotMat = WGeometry::getRotationXYZMatrix( rotX, rotY, rotZ );
    std::transform( FROM.begin(), FROM.end(), rotated.begin(), boost::bind( WGeometry::rotate, rotMat, _1 ) );
    errAdd = errorFct( rotated, closestPointCorresponces( rotated, TO ) );

    rotAct = angleOld - step;
    rotated.clear();
    rotated.resize( FROM.size() );
    rotMat = WGeometry::getRotationXYZMatrix( rotX, rotY, rotZ );
    std::transform( FROM.begin(), FROM.end(), rotated.begin(), boost::bind( WGeometry::rotate, rotMat, _1 ) );
    errSub = errorFct( rotated, closestPointCorresponces( rotated, TO ) );

    Operation::Enum dir;
    if( errAdd < errSub )
    {
        dir = Operation::ADD;
        wlog::debug( CLASS ) << "direction = ADD";
    }
    else
    {
        dir = Operation::SUB;
        wlog::debug( CLASS ) << "direction = SUB";
    }

    rotAct = angleOld;

    // start approximation
    double errorOld = numeric_limits< double >::max();
    do
    {
        rotated.clear();
        rotated.resize( FROM.size() );
        errorOld = error;
        angleOld = rotAct;

        switch( dir )
        {
            case Operation::ADD:
                rotAct += step;
                break;
            case Operation::SUB:
                rotAct -= step;
                break;
            default:
                WAssertDebug( false, "Unknown Operation!" );
                break;
        }

        rotMat = WGeometry::getRotationXYZMatrix( rotX, rotY, rotZ );
        std::transform( FROM.begin(), FROM.end(), rotated.begin(), boost::bind( WGeometry::rotate, rotMat, _1 ) );

        error = errorFct( rotated, closestPointCorresponces( rotated, TO ) );
    } while( error < errorOld );

    // restore next to last rotation
    rotAct = angleOld;

    wlog::debug( CLASS ) << "error=" << errorOld;
    return errorOld;
}

double WRegistrationNaive::translateCom( Vector& translation, PointCloud& translated, const PointCloud& FROM,
                const PointCloud& TO, double error, double factor )
{
    wlog::debug( CLASS ) << "translate() called! error=" << error;

    const Point COM_FROM = WGeometry::centerOfMass( FROM );
    const Point COM_TO = WGeometry::centerOfMass( TO );
    Vector step = COM_TO - COM_FROM;
    step = step * factor;

    // find most promising direction
    double errAdd, errSub;
    Vector tmp;

    tmp = translation + step;
    translated.clear();
    translated.resize( FROM.size() );
    std::transform( FROM.begin(), FROM.end(), translated.begin(), boost::bind( WGeometry::tranlate, tmp, _1 ) );
    errAdd = errorFct( translated, closestPointCorresponces( translated, TO ) );

    tmp = translation - step;
    translated.clear();
    translated.resize( FROM.size() );
    std::transform( FROM.begin(), FROM.end(), translated.begin(), boost::bind( WGeometry::tranlate, tmp, _1 ) );
    errSub = errorFct( translated, closestPointCorresponces( translated, TO ) );

    Operation::Enum dir;
    if( errAdd < errSub )
    {
        dir = Operation::ADD;
        wlog::debug( CLASS ) << "direction = ADD";
    }
    else
    {
        dir = Operation::SUB;
        wlog::debug( CLASS ) << "direction = SUB";
    }

    // start approximation
    double errorOld = numeric_limits< double >::max();

    do
    {
        errorOld = error;

        switch( dir )
        {
            case Operation::ADD:
                translation = translation + step;
                break;
            case Operation::SUB:
                translation = translation - step;
                break;
            default:
                WAssertDebug( false, "Unknown Operation!" );
                break;
        }

        translated.clear();
        translated.resize( FROM.size() );
        std::transform( FROM.begin(), FROM.end(), translated.begin(), boost::bind( WGeometry::tranlate, translation, _1 ) );

        error = errorFct( translated, closestPointCorresponces( translated, TO ) );
    } while( error < errorOld );

    // restore next to last translation
    if( dir == Operation::ADD )
        translation = translation - step;
    else
        translation = translation + step;

    translated.clear();
    translated.resize( FROM.size() );
    std::transform( FROM.begin(), FROM.end(), translated.begin(), boost::bind( WGeometry::tranlate, translation, _1 ) );

    wlog::debug( CLASS ) << "error" << errorOld;
    return errorOld;
}

double WRegistrationNaive::errorFct( const PointCloud& P, const PointCloud& X )
{
    return WRegistration::meanSquareError( P, X );
}
