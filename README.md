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
The CPU backend is now almost completly functional - the only thing that is left to do is fix obstacle send which is currently broken.

### GPU Backend
The GPU backend is optional, and should be implemented if there is time.

### Visualisation
The visualiusation in its current state is nearly complete, the only thing that needs adding is a way to differentiate obstacles. The streamlines also move upwards as velocity decreases, this may turn out to be fine when obstacles are involved, or may need looking into.

### User Interface
The user interface is currently barebones, with few working features. Key areas are parameter sending and changing, greater control over the backend (starting and stopping), click-and-drag node system for defining obstacles, reading in a file for obstacles.

## Plan for over christmas
The work on the project will be split into backend and frontend.
### Backend
Good progress was made on the CPU backend, such that it is now almost fully functional. The last thing to implement is obstacle sending, and possibly some optimisation. I plan not to start on the GPU backend until most of the UI is done, just to be totally sure that is finished in time.

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

**ALSO: Need to convert from half-arsed MVVM to proper MVVM**
