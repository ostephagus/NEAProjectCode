# NEA Fluid Dynamics Simulation Code
This is a repository for the code I make as part of my technical solution of my A-Level Computer Science NEA.

## CPU Backend
The CPU Backend is the first part of the project, allowing me to get to grips with C++, and is mostly the same as the code in the Griebels Book. **INDEV**

## GPU Backend
The GPU Backend is the evolution of the CPU Backend to work on a GPU, using parallelisation and SIMD (Single-Instruction Multiple-Data) for greater processing efficiency. **TODO**

## UI
The UI is written in C# using the WPF library and provides a GUI for the project. **INDEV**

## Visualisation
The visualisation is written in C# using OpenGL to display a colour gradient of a field. It takes the data directly from the backend to display in a portion of the UI. **INDEV**

## Current status
### CPU Backend
The CPU backend needs to be completed and, if the GPU backend will not be made, optimised so that it can run in close to real time. The inability to simulate obstacles needs to be sorted also, perhaps with some guard rails on the different quantities near boundaries.

### GPU Backend
The GPU backend is optional, and should be implemented if there is time.

### Visualisation
The visualiusation in its current state is nearly complete, the only thing that needs adding is a way to differentiate obstacles. The streamlines also move upwards as velocity decreases, this may turn out to be fine when obstacles are involved, or may need looking into.

### User Interface
The user interface is currently barebones, with few working features. Key areas are parameter sending and changing, greater control over the backend (starting and stopping), click-and-drag node system for defining obstacles, reading in a file for obstacles.

## Plan for over christmas
The work on the project will be split into backend and frontend.
### Backend
The first priority is ensuring that obstacles can be simulated. This will likely need a long session of debugging to figure out precisely where the simulation is going wrong. I hope to be able to get this done in 1 week (done before Christmas day). Next will be optimisation - tweaking parameters to ensure greatest stability and processing efficiency. The slowest part of the solver at the moment is the iterative pressure solution. If the CPU backend is to be the main backend, more work could be done on optimising the multi-threading there (Griebels chapter 8). If the GPU backend is to be used, the CPU backend will not need to be optimised beyond finding parameter values. If the CPU backend debugging is not finished by Christmas, the GPU backend will not be made.

### Frontend
The frontend is made up of lots of smaller tasks, below is the order and how long approximately they should take. I will update this as I go.
Task | Duration | Days
-----|----------|-----
Linking parameter inputs to a global state to send to backend | 1 day | 18/12/23
Sending parameters to backend | 1 day | 19/12/23
Improving backend control (pausing as well as stopping) | 1 day | 22/12/23
Greying out obstacles in visualisation | 1 hour | 22/12/23
Choose file type(s) to accept and add file reading | 2 days | 28/12/23 - 29/12/23
Click-and-drag node system | 3 days | 2/12/23 - 4/12/23
