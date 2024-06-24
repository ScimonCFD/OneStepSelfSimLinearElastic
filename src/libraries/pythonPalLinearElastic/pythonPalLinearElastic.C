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

#include "pythonPalLinearElastic.H"
#include "addToRunTimeSelectionTable.H"
#include "zeroGradientFvPatchFields.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(pythonPalLinearElastic, 0);
    addToRunTimeSelectionTable
    (
        mechanicalLaw, pythonPalLinearElastic, linGeomMechLaw
    );
}


// * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * * * //

void Foam::pythonPalLinearElastic::updateStrain()
{
    if (incremental())
    {
        // Lookup gradient of displacement increment
        const volTensorField& gradDD =
            mesh().lookupObject<volTensorField>("grad(DD)");

        // Calculate the total strain
        epsilon_ = epsilon_.oldTime() + symm(gradDD);
    }
    else
    {
        // Lookup gradient of displacement
        const volTensorField& gradD =
            mesh().lookupObject<volTensorField>("grad(D)");

        // Calculate the total strain
        epsilon_ = symm(gradD);
    }
}

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

// Construct from dictionary
Foam::pythonPalLinearElastic::pythonPalLinearElastic
(
    const word& name,
    const fvMesh& mesh,
    const dictionary& dict,
    const nonLinearGeometry::nonLinearType& nonLinGeom
)
:
    mechanicalLaw(name, mesh, dict, nonLinGeom),
    myPythonPal("python_script.py", false),
    rho_(dict.lookup("rho")),
    impK_(dict.lookup("implicitStiffness")),
    epsilon_
    (
        IOobject
        (
            "epsilon",
            mesh.time().timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        mesh,
        dimensionedSymmTensor("zero", dimless, symmTensor::zero)
    )
{

    // Load the python file and evaluate it
    const word pythonMod =
        dict.lookupOrDefault<word>("pythonModule", "python_code.py");
    pythonPal myPythonPal(pythonMod, false);

    // Check impK is positive
    if (impK_.value() < SMALL)
    {
        FatalErrorIn
        (
            "Foam::pythonPalLinearElastic::pythonPalLinearElastic\n"
            "(\n"
            "    const word& name,\n"
            "    const fvMesh& mesh,\n"
            "    const dictionary& dict\n"
            ")"
        )   << "The implicitStiffness should be positive!"
            << abort(FatalError);
    }

    // Store the old time
    epsilon_.oldTime();
}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

Foam::pythonPalLinearElastic::~pythonPalLinearElastic()
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

Foam::tmp<Foam::volScalarField> Foam::pythonPalLinearElastic::rho() const
{
    tmp<volScalarField> tresult
    (
        new volScalarField
        (
            IOobject
            (
                "rho",
                mesh().time().timeName(),
                mesh(),
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ),
            mesh(),
            rho_,
            zeroGradientFvPatchScalarField::typeName
        )
    );

#ifdef OPENFOAMESIORFOUNDATION
    tresult.ref().correctBoundaryConditions();
#else
    tresult().correctBoundaryConditions();
#endif

    return tresult;
}


Foam::tmp<Foam::volScalarField> Foam::pythonPalLinearElastic::impK() const
{
    return tmp<volScalarField>
    (
        new volScalarField
        (
            IOobject
            (
                "impK",
                mesh().time().timeName(),
                mesh(),
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ),
            mesh(),
            impK_
        )
    );
}

Foam::tmp<Foam::volScalarField> Foam::pythonPalLinearElastic::K() const
{
    notImplemented("Foam::pythonPalLinearElastic::K()");

    // Keep the compiler happy
    return impK();
}


void Foam::pythonPalLinearElastic::calculateStress
(
    symmTensorField& sigma,
    symmTensorField& epsilon
)
{
    if (sigma.size() != 0)
    {
        // Pass epsilon and sigma to the Python side 
        myPythonPal.passToPython(sigma, "sigma");
        myPythonPal.passToPython(epsilon, "epsilon");

        // // Call the Python predict() function to calculate the stress field
        myPythonPal.execute("predict()");
    }
}

void Foam::pythonPalLinearElastic::correct(volSymmTensorField& sigma)
{
    // Update strain volSymmTensorField (epsilon)
    updateStrain();

    // Take references for brevity and efficiency
    #ifdef FOAMEXTEND
        symmTensorField& sigmaI = sigma.internalField();
    #else
        symmTensorField& sigmaI = sigma.primitiveFieldRef();
    #endif
        symmTensorField epsilonI = epsilon_.internalField();

    // Calculate stress in the internal field
    calculateStress(sigmaI, epsilonI);

    // Loop over all boundary patches
    forAll(sigma.boundaryField(), patchI)
    {
        // Take references for brevity and efficiency
        #ifdef FOAMEXTEND
            symmTensorField& sigmaP = sigma.boundaryField()[patchI];
        #else
            symmTensorField& sigmaP = sigma.boundaryFieldRef()[patchI];
        #endif
        symmTensorField  epsilonP = epsilon_.boundaryField()[patchI];

        // Calculate stress on the boundary patch
        calculateStress(sigmaP, epsilonP);
    }
}


void Foam::pythonPalLinearElastic::correct(surfaceSymmTensorField& sigma)
{
    notImplemented
    (
        "void Foam::pythonPalLinearElastic::correct(surfaceSymmTensorField&)"
    );
}


Foam::scalar Foam::pythonPalLinearElastic::residual()
{
    // For nowlinear laws, we can calculate this in some way so that the solid
    // model knows if the law has converged
    return 0.0;
}


// ************************************************************************* //
