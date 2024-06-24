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

   Application
    fieldsManipulator

   Description
    Utility used for the self-simulation algorithm.
    When performing the load-driven simulation, It saves the fields to 
    NumPy-readable files. 
    If performing the displacemnt-driven simulation, It sets the expected 
    displacemnents as boundary conditions. 
    Python-OpenFOAM interoperability is achieved via pythonPal4foam.

   Authors
    Simon A. Rodriguez, UCD. All rights reserved
    Philip Cardiff, UCD. All rights reserved
\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "pisoControl.H"
#include "pythonPal.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{

    #include "setRootCaseLists2.H"
    #include "createTime.H"
    #include "createMesh.H"

    pythonPal myPythonPal("python_script.py", true);

    // Read DExpected and D
    volVectorField DExpected
    (
        IOobject
        (
            "DExpected",
            runTime.timeName(),
            mesh,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        mesh
    );

    volVectorField D
    (
        IOobject
        (
            "D",
            runTime.timeName(),
            mesh,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        mesh
    );

    // Get references to the boundaries
    label Id_Up = mesh.boundaryMesh().findPatchID("up");
    label Id_Down = mesh.boundaryMesh().findPatchID("down");
    label Id_Hole = mesh.boundaryMesh().findPatchID("hole");
    label Id_Right = mesh.boundaryMesh().findPatchID("right");

    vectorField& D_Right = D.boundaryFieldRef()[Id_Right]; 
    vectorField& D_Up = D.boundaryFieldRef()[Id_Up];
    vectorField& D_Hole = D.boundaryFieldRef()[Id_Hole];   
    vectorField& D_Down = D.boundaryFieldRef()[Id_Down];

    // Read the "flag"
    myPythonPal.execute("exec('flag  = deserialise_flag()')");
    word flag = myPythonPal.retrieveWordFromPython("flag");

    // If flag == "BOUNDARY_CHANGE" replace D at the boundaries with DExpected 
    if (flag == "BOUNDARY_CHANGE") {
        D_Right = DExpected.boundaryField()[Id_Right];
        D_Up = DExpected.boundaryField()[Id_Up];
        D_Hole = DExpected.boundaryField()[Id_Hole];
        D_Down = DExpected.boundaryField()[Id_Down];
        D.write();
    }

    // else, just save the different fields as Python-readable (NumPy) files
    else {

        volSymmTensorField sigma
        (
            IOobject
            (
                "sigma",
                runTime.timeName(),
                mesh,
                IOobject::MUST_READ,
                IOobject::AUTO_WRITE
            ),
            mesh
        );

        volSymmTensorField sigmaExpected
        (
            IOobject
            (
                "sigmaExpected",
                runTime.timeName(),
                mesh,
                IOobject::MUST_READ,
                IOobject::AUTO_WRITE
            ),
            mesh
        );

        volSymmTensorField epsilon
        (
            IOobject
            (
                "epsilon",
                runTime.timeName(),
                mesh,
                IOobject::MUST_READ,
                IOobject::AUTO_WRITE
            ),
            mesh
        );

        volSymmTensorField epsilonExpected
        (
            IOobject
            (
                "epsilonExpected",
                runTime.timeName(),
                mesh,
                IOobject::MUST_READ,
                IOobject::AUTO_WRITE
            ),
            mesh
        );

        // Pass the "time" to Python side
        myPythonPal.passScalarToPython(runTime.value(), "time");

        // Pass the different fields to the Python side
        myPythonPal.passToPython(epsilon, "epsilon");
        myPythonPal.passToPython(epsilonExpected, "epsilonExpected");
        myPythonPal.passToPython(sigma, "sigma");
        myPythonPal.passToPython(sigmaExpected, "sigmaExpected");
        myPythonPal.passToPython(D, "D");
        myPythonPal.passToPython(DExpected, "DExpected");

        // Save the fields as NumPy-readable files
        myPythonPal.execute("serialise_fields()");

    }

return 0;

}