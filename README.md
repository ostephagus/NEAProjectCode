# NEA Fluid Dynamics Simulation Code
This is a repository for the code I make as part of my technical solution of my A-Level Computer Science NEA.

## CPU Backend
The CPU Backend is the first part of the project, allowing me to get to grips with C++, and is mostly the same as the code in the Griebels Book. **DONE**

## GPU Backend
The GPU Backend is the evolution of the CPU Backend to work on a GPU, using parallelisation and SIMD (Single-Instruction Multiple-Data) for greater processing efficiency. **INDEV**

## UI
The UI is written in C# using the WPF library and provides a GUI for the project. **INDEV**

## Visualisation
The visualisation is written in C# using OpenGL to display a colour gradient of a field. It takes the data directly from the backend to display in a portion of the UI. **INDEV**

## Current status
### CPU Backend
The CPU backend is finished. A few more optimisation could be carried out if there is to be no GPU backend

### GPU Backend
The GPU backend is optional, and should be implemented if there is time.

### Visualisation
The visualiusation in its current state is nearly complete, the only thing that needs adding is a way to differentiate obstacles.

### User Interface
The User Interface is now able to change and send parameters, although key areas still to work on are adding tooltips, greater control over the backend (starting and stopping), click-and-drag node system for defining obstacles and reading in a file for obstacles. In its current state, the User Interface does not follow the MVVM design pattern, and shoud be refactored to do so.

## Plan for over christmas
The work on the project will be split into backend and frontend.
### Backend
The GPU backend is underway, with some of the easier kernels finished. Through January and into February, I plan to finish this with the more difficult kernels (reduction kernels and SOR).

### Frontend
The frontend is made up of lots of smaller tasks, below is the order and how long approximately they should take. I will update this as I go.
Task | Duration | Days
-----|----------|-----
Improving backend control (pausing as well as stopping) | 1 hour | 27/12/23
Greying out obstacles in visualisation | 1 hour | 27/12/23
Choose file type(s) to accept and add file reading | 2 days | 28/12/23 - 29/12/23
Click-and-drag node system | 3 days | 2/12/23 - 4/12/23

**ALSO: Need to convert from half-arsed MVVM to proper MVVM**
