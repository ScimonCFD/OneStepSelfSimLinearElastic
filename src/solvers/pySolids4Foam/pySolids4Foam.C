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
    pySolids4Foam

   Original solver
    solids4Foam

   Modified by
    Simon A. Rodriguez, UCD. All rights reserved
    Philip Cardiff, UCD. All rights reserved

   Description
    General solver where the solved mathematical model (fluid, solid or
    fluid-solid) is chosen at run-time. It uses pythonPal4foam to save the 
    epsilon and sigma fields as Python-readable files.
\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "physicsModel.H"
#include "pythonPal.H"
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{

#   include "setRootCase2.H"
#   include "createTime.H"
#   include "solids4FoamWriteHeader.H"

    pythonPal myPythonPal("python_script.py", true);

    // Create the general physics class
    autoPtr<physicsModel> physics = physicsModel::New(runTime);

    while (runTime.run())
    {
        physics().setDeltaT(runTime);

        runTime++;

        Info<< "Time = " << runTime.timeName() << nl << endl;

        // Solve the mathematical model
        physics().evolve();

        // Let the physics model know the end of the time-step has been reached
        physics().updateTotalFields();

        if (runTime.outputTime())
        {
            physics().writeFields(runTime);

            // Pass the "time" variable to the Python side 
            myPythonPal.passScalarToPython(runTime.value(), "time");

            // Retrieve the "mesh" object from the mechanical law
            const fvMesh& mesh = runTime.lookupObject<fvMesh>("region0");
            
            // Retrieve epsilon and sigma fields from the mechanical law
            volSymmTensorField epsilon = 
                mesh.lookupObject<volSymmTensorField>("epsilon_");
            volSymmTensorField sigma = 
                mesh.lookupObject<volSymmTensorField>("sigma");

            // Pass the epsilon and sigma fields to the Python side 
            myPythonPal.passToPython(epsilon, "epsilon");
            myPythonPal.passToPython(sigma, "sigma");

            // Save epsilon and sigma as Python=readable (NumPy) files
            myPythonPal.execute("serialise_fields()");

        }

        Info<< "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
            << "  ClockTime = " << runTime.elapsedClockTime() << " s"
            << nl << endl;
    }

    physics().end();

    Info<< nl << "End" << nl << endl;

    return(0);
}


// ************************************************************************* //