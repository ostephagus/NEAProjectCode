# NEA Fluid Dynamics Simulation Code
This is a repository for the code I make as part of my technical solution of my A-Level Computer Science NEA.

## CPU Backend
The CPU Backend is the first part of the project, allowing me to get to grips with C++, and is mostly the same as the code in the Griebels Book. **DONE**

## GPU Backend
The GPU Backend is the evolution of the CPU Backend to work on a GPU, using parallelisation and SIMD (Single-Instruction Multiple-Data) for greater processing efficiency. **DONE**

## UI
The UI is written in C# using the WPF library and provides a GUI for the project. **INDEV**

## Visualisation
The visualisation is written in C# using OpenGL to display a colour gradient of a field. It takes the data directly from the backend to display in a portion of the UI. **INDEV**

## Current status
### CPU Backend
The CPU backend is finished. A few more optimisations could be carried out if there is time.

### GPU Backend
The GPU backend is working, although some profiling and optimisation would be very useful.

### Visualisation
The visualiusation in its current state is nearly complete, the only thing that needs adding is a way to differentiate obstacles.

### User Interface
The User Interface is now able to change and send parameters, although key areas still to work on are adding tooltips, greater control over the backend (starting and stopping), click-and-drag node system for defining obstacles and reading in a file for obstacles.

## Current areas of work (Jan/Feb 2024)
### GPU Backend
Profiling and optimisation (via Nsight profiler).

### User Interface
Click-and-drag node system.
