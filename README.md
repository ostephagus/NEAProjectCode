# NEA Fluid Dynamics Simulation Code
This is a repository for the code I make as part of my technical solution of my A-Level Computer Science NEA.

## CPU Backend
The CPU Backend is the first part of the project, allowing me to get to grips with C++, and is mostly the same as the code in the Griebels Book. **DONE**

## GPU Backend
The GPU Backend is the evolution of the CPU Backend to work on a GPU, using parallelisation and SIMD (Single-Instruction Multiple-Data) for greater processing efficiency. **DONE**

## UI
The UI is written in C# using the WPF library and provides a GUI for the project. **INDEV**

## Visualisation
The visualisation is written in C# using OpenGL to display a colour gradient of a field. It takes the data directly from the backend to display in a portion of the UI. **DONE**

## Current status
### CPU Backend
The CPU backend is finished. A few more optimisations could be carried out if there is time.

### GPU Backend
The GPU backend is working, although some profiling and optimisation would be very useful.

### Visualisation
The visualiusation is complete.

### User Interface
The User Interface is now able to change and send parameters and the click-and-drag node system works, although key areas still to work on are adding tooltips, and reading in a file for obstacles.

## Current areas of work (March 24)

### User Interface
- 2 modes for obstacles - self-drawn or using binary files (latter locks structure editing).
