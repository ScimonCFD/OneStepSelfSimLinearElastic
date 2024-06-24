/* License
    This program is free software: you can redistribute it and/or modify 
    it under the terms of the GNU General Public License as published 
    by the Free Software Foundation, either version 3 of the License, 
    or (at your option) any later version.

    This program is distributed in the hope that it will be useful, 
    but WITHOUT ANY WARRANTY; without even the implied warranty of 
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 

    See the GNU General Public License for more details. You should have 
    received a copy of the GNU General Public License along with this 
    program. If not, see <https://www.gnu.org/licenses/>. 

Class
    pythonPalLinearElastic

Description
    Mechanical law that calculates stresses (linear elastic Hookean law) in 
    Python via pythonPal4foam.

SourceFiles
    pythonPalLinearElastic.C

Original material model
    linearElasticMisesPlastic

Modified by
    Simon A. Rodriguez, UCD. All rights reserved
    Philip Cardiff, UCD. All rights reserved

\*---------------------------------------------------------------------------*/

#ifndef pythonPalLinearElastic_H
#define pythonPalLinearElastic_H

#include "mechanicalLaw.H"
#include "surfaceFields.H"
#include "pythonPal.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                         Class pythonPalLinearElastic Declaration
\*---------------------------------------------------------------------------*/

class pythonPalLinearElastic
:
    public mechanicalLaw
{
    // Private data

        pythonPal myPythonPal;

        //- Density
        dimensionedScalar rho_;

        //- Implicit stiffness used by the solid model
        //  Assuming convergence is achieved, this will not affect the final answer
        //  Setting it to the equivalent of 2*mu + lambda in elasticity is optimal
        dimensionedScalar impK_;

        //- Total strain field
        volSymmTensorField epsilon_;

    // Private Member Functions

        //- Disallow default bitwise copy construct
        pythonPalLinearElastic(const pythonPalLinearElastic&);

        //- Disallow default bitwise assignment
        void operator=(const pythonPalLinearElastic&);

        //- Update the strain field
        void updateStrain();

        void calculateStress
        (
            symmTensorField& sigmaI,
            symmTensorField& epsilonI
        );

public:

    //- Runtime type information
    TypeName("pythonPalLinearElastic");

    // Constructors

        //- Construct from dictionary
        pythonPalLinearElastic
        (
            const word& name,
            const fvMesh& mesh,
            const dictionary& dict,
            const nonLinearGeometry::nonLinearType& nonLinGeom
        );


    // Destructor

        virtual ~pythonPalLinearElastic();


    // Member Functions

        //- Return density
        virtual tmp<volScalarField> rho() const;

        //- Return the implicit stiffness
        //  This is the diffusivity for the Laplacian term
        virtual tmp<volScalarField> impK() const;

        //- Return the bulk modulus
        virtual tmp<volScalarField> K() const;

        //- Calculate the stress
        virtual void correct(volSymmTensorField& sigma);

        //- Calculate the stress
        virtual void correct(surfaceSymmTensorField& sigma);

        //- Return material residual i.e. a measured of how convergence of
        //  the material model
        virtual scalar residual();
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
