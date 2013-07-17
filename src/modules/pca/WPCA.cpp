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

#include <cmath>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

#include <boost/shared_ptr.hpp>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <core/common/WAssert.h>
#include <core/common/WLogger.h>

#include "core/data/WLEMMEnumTypes.h"
#include "core/data/emd/WLEMData.h"
#include "core/data/emd/WLEMDPCA.h"

#include "WPCA.h"

#define DIMS 2
using namespace std;
using namespace Eigen;

const std::string WPCA::CLASS = "WPCA";

WPCA::WPCA( int newNumDimensions, bool newReverse )
{
    m_num_dims = newNumDimensions;
    m_reverse = newReverse;
}

WPCA::~WPCA()
{
}

void WPCA::setParams( int newNumDimensions, bool newReverse )
{
    m_num_dims = newNumDimensions;
    m_reverse = newReverse;
}

WLEMData::SPtr WPCA::processData( WLEMData::SPtr emdIn )
{
    wlog::debug( CLASS ) << "processData() called!";
    wlog::debug( CLASS ) << "Average pieces of first modality (PRE-COMPUTATION): ";
    wlog::debug( CLASS ) << "Reducing to dimensionality: " << m_num_dims;

// Test data from the PCA tutorial at http://www.sccg.sk/~haladova/principal_components.pdf
    /*
     std::vector< std::vector< double > > data_in(2, std::vector< double >(10, 0));
     data_in[0][0] = 2.5;
     data_in[0][1] = 0.5;
     data_in[0][2] = 2.2;
     data_in[0][3] = 1.9;
     data_in[0][4] = 3.1;
     data_in[0][5] = 2.3;
     data_in[0][6] = 2;
     data_in[0][7] = 1;
     data_in[0][8] = 1.5;
     data_in[0][9] = 1.1;

     data_in[1][0] = 2.4;
     data_in[1][1] = 0.7;
     data_in[1][2] = 2.9;
     data_in[1][3] = 2.2;
     data_in[1][4] = 3.0;
     data_in[1][5] = 2.7;
     data_in[1][6] = 1.6;
     data_in[1][7] = 1.1;
     data_in[1][8] = 1.6;
     data_in[1][9] = 0.9;
     */

    if( !m_reverse && emdIn->getModalityType() != LaBP::WEModalityType::PCA ) // EEG, MEG ... to PCA
    {
        wlog::debug( CLASS ) << "EEG, MEG ... to PCA";
        WLEMData::DataT& dataIn = emdIn->getData();
        wlog::debug( CLASS ) << "dataIn: " << dataIn.size() << " x " << dataIn.front().size();
#ifdef DEBUG
        wlog::debug( CLASS ) << "dataIn (first channels and samples only):";
        for( size_t row = 0; row < dataIn.size() && row < 5; ++row )
        {
            std::stringstream ss;
            for( size_t col = 0; col < dataIn[row].size() && col < 10; ++col )
            {
                ss << dataIn[row][col] << " ";
            }
            wlog::debug( CLASS ) << ss.str();
        }
#endif /* DEBUG */

        boost::shared_ptr< WLEMDPCA::DataT > dataOut = computePCA( dataIn );
        wlog::debug( CLASS ) << "dataOut: " << dataOut->size() << " x " << dataOut->front().size();
#ifdef DEBUG
        wlog::debug( CLASS ) << "dataOut (first channels and samples only):";
        for( size_t row = 0; row < dataOut->size() && row < 5; ++row )
        {
            std::stringstream ss;
            for( size_t col = 0; col < ( *dataOut )[row].size() && col < 10; ++col )
            {
                ss << ( *dataOut )[row][col] << " ";
            }
            wlog::debug( CLASS ) << ss.str();
        }
#endif /* DEBUG */

        WLEMDPCA::SPtr pcaOut = createPCAContainer( emdIn, dataOut );
        return pcaOut;
    }
    else
        if( m_reverse && emdIn->getModalityType() == LaBP::WEModalityType::PCA ) // PCA to EEG, MEG, ...
        {
            wlog::debug( CLASS ) << "PCA to EEG, MEG ...";
            WLEMDPCA::SPtr pcaIn = emdIn->getAs< WLEMDPCA >();
            WLEMData::SPtr emdOut = WPCA::convertPCAToModality( pcaIn );
            return emdOut;
        }
        else
        {
            // TODO
            wlog::error( CLASS ) << "Wrong input data! Only PCA can be reversed to other modality.";
            return emdIn->clone();
        }
}

WLEMData::SPtr WPCA::convertPCAToModality( WLEMDPCA::SPtr pcaIn )
{
    wlog::debug( CLASS ) << "convertPCAToModality() called!";
    std::vector< std::vector< double > >& oldPcaData = pcaIn->getData();
    MatrixXd pcaMatrix( oldPcaData.size(), oldPcaData.front().size() );
    for( MatrixXd::Index i = 0; i < pcaMatrix.rows(); ++i )
    {
        for( MatrixXd::Index j = 0; j < pcaMatrix.cols(); ++j )
        {
            pcaMatrix( i, j ) = oldPcaData[i][j];
        }
    }

    MatrixXd& transMatrix = pcaIn->getTransformationMatrix();
    MatrixXd transPcaMat = transMatrix.transpose() * pcaMatrix;

    boost::shared_ptr< std::vector< std::vector< double > > > modalityData( new std::vector< std::vector< double > >() );
    modalityData->reserve( transMatrix.cols() );
    modalityData->resize( transMatrix.cols() );
    VectorT& channelMeans = pcaIn->getChannelMeans();

    for( size_t i = 0; i < ( *modalityData ).size(); ++i )
    {
        ( *modalityData )[i].reserve( pcaMatrix.cols() );
        for( MatrixXd::Index j = 0; j < pcaMatrix.cols(); ++j )
        {
            ( *modalityData )[i].push_back( transPcaMat( i, j ) + channelMeans[i] );
        }
    }

#ifdef DEBUG
    wlog::debug( CLASS ) << "Results:";
    wlog::debug( CLASS ) << "Transformation matrix:";
    for( int i = 0; i < transMatrix.rows() && i < 5; i++ )
    {
        std::stringstream ss;
        for( int j = 0; j < transMatrix.cols() && j < 10; j++ )
        {
            ss << transMatrix( i, j ) << " ";
        }
        wlog::debug( CLASS ) << ss.str();
    }

    wlog::debug( CLASS ) << "PCA data:";
    for( MatrixXd::Index i = 0; i < pcaMatrix.rows() && i < 5; ++i )
    {
        std::stringstream ss;
        for( MatrixXd::Index j = 0; j < pcaMatrix.cols() && j < 10; ++j )
        {
            ss << pcaMatrix( i, j ) << " ";
        }
        wlog::debug( CLASS ) << ss.str();
    }

    wlog::debug( CLASS ) << "Modality data:";
    for( size_t i = 0; i < ( *modalityData ).size() && i < 5; ++i )
    {
        std::stringstream ss;
        for( size_t j = 0; j < ( *modalityData )[i].size() && j < 10; ++j )
        {
            ss << ( *modalityData )[i][j] << " ";
        }
        wlog::debug( CLASS ) << ss.str();
    }
#endif // DEBUG
    WLEMData::SPtr emdOut( pcaIn->getPreprocessedData()->clone() );
    emdOut->setData( modalityData );
    return emdOut;
}

WLEMDPCA::SPtr WPCA::createPCAContainer( WLEMData::SPtr emdIn,
                boost::shared_ptr< WLEMDPCA::DataT > pcaData )
{
    WLEMDPCA::SPtr pcaOut( new WLEMDPCA( *emdIn ) );
    pcaOut->setData( pcaData );
    pcaOut->setPreprocessedData( emdIn );
    pcaOut->setChannelMeans( m_channelMeans );
    pcaOut->setTransformationMatrix( m_transformationMatrix );

    return pcaOut;
}

// Accepts a row-major matrix
Eigen::MatrixXd WPCA::getCovarianceMatrix( Eigen::MatrixXd data )
{
    //stores the final covariance matrix
    MatrixXd cov( data.rows(), data.rows() );

    for( int i = 0; i < data.rows(); i++ )
    {
        for( int j = 0; j < data.rows(); j++ )
        {
            double this_cov = 0;
            for( int k = 0; k < data.cols(); k++ )
            {
                this_cov += data( i, k ) * data( j, k );
            }
            cov( i, j ) = this_cov / ( data.cols() - 1 );
        }
    }
    return cov;
}

boost::shared_ptr< WLEMData::DataT > WPCA::computePCA( WLEMData::DataT& rawData )
{
    wlog::debug( CLASS ) << "computePCA() called!";
    wlog::debug( CLASS ) << "Allocating data into Eigen structure ...";
    MatrixXd data( rawData.size(), rawData[0].size() );
    for( size_t i = 0; i < ( size_t )data.rows(); ++i )
    {
        for( size_t j = 0; j < ( size_t )data.cols(); ++j )
        {
            data( i, j ) = rawData[i][j];
        }
    }

    //pre-compute the mean for each dimension, used when reconstructing the data
    m_channelMeans.reset( new VectorT( data.rows() ) );
    for( MatrixXd::Index i = 0; i < data.rows(); ++i )
    {
        double mean = 0;
        for( MatrixXd::Index j = 0; j < data.cols(); ++j )
        {
            mean += data( i, j );
        }
        mean /= data.cols();
        ( *m_channelMeans )( i ) = mean;
    }

    wlog::debug( CLASS ) << "Beginning PCA computation ...";

    //std::cout << std::endl << "data:\n" << data;

    for( MatrixXd::Index i = 0; i < data.rows(); ++i )
    {
        for( MatrixXd::Index j = 0; j < data.cols(); ++j )
        {
            data( i, j ) -= ( *m_channelMeans )( i );
        }
    }

    MatrixXd covariance( DIMS, DIMS );
    covariance = getCovarianceMatrix( data );

    wlog::debug( CLASS ) << "Done generating covariance matrix ...";

    //now solve for the eigenvectors of the covariance matrix
    SelfAdjointEigenSolver< MatrixXd > eigensolver( covariance );
    if( eigensolver.info() != Success )
    {
        wlog::debug( CLASS ) << "Error with the eigensolver!" << eigensolver.info();
        WAssert( false, "Error with the eigensolver!" );
    }

    MatrixXd eigenvalues = eigensolver.eigenvalues();
    MatrixXd eigenvectors = eigensolver.eigenvectors();

    MatrixXd::Index dims = static_cast< MatrixXd::Index >( m_num_dims );
    if( dims > eigenvectors.cols() )
        dims = eigenvectors.cols();

    // create the reduced feature vector from the full vector of eigenvalues
    // there's can be done more efficiently using some Eigen features
    m_transformationMatrix.reset( new MatrixT( dims, eigenvectors.cols() ) );
    for( MatrixXd::Index i = 0; i < dims; ++i )
    {
        for( MatrixXd::Index j = 0; j < eigenvectors.rows(); ++j )
        {
            ( *m_transformationMatrix )( i, j ) = eigenvectors( j, eigenvectors.cols() - i - 1 );
        }
    }

// now pack the data back into the the vector of vectors for use in the module
    MatrixXd PCAData( m_transformationMatrix->rows(), data.cols() );
    PCAData = ( *m_transformationMatrix * data );

    // store the computed transformation matrix so that it can be associated with the WDataSetEMMPCA object
    //m_transformation_matrix = eigenMatrixTo2DVector( rowFeatureVector );

    // copy the matrix form back into the c++ 2D vector type
    boost::shared_ptr< std::vector< std::vector< double > > > retData = eigenMatrixTo2DVector( PCAData );

    wlog::debug( CLASS ) << "Done allocating results, PCA computation completed successfully.";
    return retData;
}

boost::shared_ptr< std::vector< std::vector< double > > > WPCA::eigenMatrixTo2DVector( Eigen::MatrixXd& matrix )
{
    boost::shared_ptr< std::vector< std::vector< double > > > vectors(
                    new std::vector< std::vector< double > >( matrix.rows(), std::vector< double >( matrix.cols() ) ) );
    for( size_t i = 0; i < ( size_t )matrix.rows(); ++i )
    {
        for( size_t j = 0; j < ( size_t )matrix.cols(); ++j )
        {
            ( *vectors )[i][j] = matrix( i, j );
        }
    }
    return vectors;
}

// expects a row vector Eigen matrix; returns a c++ vector representation
boost::shared_ptr< std::vector< double > > WPCA::eigenMatrixTo1DVector( Eigen::MatrixXd& matrix )
{
    boost::shared_ptr< std::vector< double > > vectors( new std::vector< double >( matrix.rows(), 0 ) );
    for( size_t i = 0; i < ( size_t )matrix.rows(); ++i )
    {
        ( *vectors )[i] = matrix( i );
    }
    return vectors;
}
