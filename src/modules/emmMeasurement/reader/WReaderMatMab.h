// TODO doc & license
#ifndef WREADERMATMAB_H_
#define WREADERMATMAB_H_

#include <string>

#include <boost/shared_ptr.hpp>
#include <Eigen/Core>

#include "core/common/math/WLMatrixTypes.h"
#include "core/dataHandler/io/WReader.h"

class WReaderMatMab: public WReader
{
public:
    /**
     * Abbreviation for a shared pointer.
     */
    typedef boost::shared_ptr< WReaderMatMab > SPtr;

    /**
     * Abbreviation for const shared pointer.
     */
    typedef boost::shared_ptr< const WReaderMatMab > ConstSPtr;

    /**
     * Constructs a reader object.
     *
     * \param fname path to file which should be loaded
     */
    explicit WReaderMatMab( std::string fname );
    virtual ~WReaderMatMab();

    struct ReturnCode
    {
        enum Enum
        {
            SUCCESS, /**< Normal */
            ERROR_FOPEN, /**< Error opening file */
            ERROR_FREAD, /**< File read error */
            ERROR_UNKNOWN /**< Unknown error */
        };
    };

    ReturnCode::Enum read( LaBP::MatrixSPtr& matrix );

private:
    ReturnCode::Enum readMab( LaBP::MatrixSPtr matrix, std::string fName, size_t rows, size_t cols );
};

#endif /* WREADERMATMAB_H_ */
