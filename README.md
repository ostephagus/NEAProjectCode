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
The GPU backend is working, although further optimisation is needed, especially for pressure iteration. For example, the boundary conditions kernels could form part of a single kernel using boolean multiplication. CUDA graphs should also be used with one graph to represent a timestep.

### File Creator
The file creator is able to make files that can be used by the program. A GUI should be made to allow drawing with splines.

### Visualisation
The visualiusation is complete.

### User Interface
The User Interface is now able to change and send parameters and function as a user interface. Areas to work on are adding a new spline creation method based on Catmull-Rom splines or similar, and adding tooltips.

## Current areas of work (summer 24)
### User Interface
- Adding new splines to draw with
### File Creator
- Adding a GUI to draw obstacles using splines
### GPU Backend
- Optimisations as discussed above
