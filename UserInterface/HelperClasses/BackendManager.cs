#if DEBUG
//#define NO_GPU_BACKEND
#endif

using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Threading;
using System.Windows;


namespace UserInterface.HelperClasses
{
    public enum BackendStatus
    {
        /// <summary>
        /// Process created but not yet executing.
        /// </summary>
        NotStarted,

        /// <summary>
        /// Currently executing
        /// </summary>
        Running,

        /// <summary>
        /// Not executing, but in a paused state.
        /// </summary>
        Stopped,

        /// <summary>
        /// Not executing and the process has been destroyed or not yet created.
        /// </summary>
        Closed
    }

    /// <summary>
    /// Handler class for dealing with the backend
    /// </summary>
    public class BackendManager : INotifyPropertyChanged
    {
        private Process? backendProcess;
        private string filePath;
        private PipeManager? pipeManager;
        private int iMax;
        private int jMax;

        private BackendStatus backendStatus;

        private float[][]? fields;
        private FieldType[]? namedFields;

        private float frameTime;
        private float dragCoefficient;
        private Stopwatch frameTimer;

        private ResizableLinearQueue<ParameterChangedEventArgs> parameterSendQueue;
        private ParameterHolder parameterHolder;

        private readonly string pipeName = "NEAFluidDynamicsPipe";

        public int FieldLength { get => iMax * jMax; }
        public int IMax { get => iMax; set => iMax = value; }
        public int JMax { get => jMax; set => jMax = value; }

        public float FrameTime
        {
            get => frameTime;
            private set
            {
                frameTime = value;
                PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(nameof(FrameTime)));
            }
        }

        public float DragCoefficient
        {
            get => dragCoefficient;
            private set
            {
                dragCoefficient = value;
                PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(nameof(DragCoefficient)));
            }
        }

        public BackendStatus BackendStatus
        {
            get => backendStatus;
            private set
            {
                backendStatus = value;
                PropertyChanged?.Invoke(value, new PropertyChangedEventArgs(nameof(BackendStatus)));
            }
        }

        public event PropertyChangedEventHandler? PropertyChanged;

        private bool CreateBackend()
        {
            try
            {
                backendProcess = new Process();
                backendProcess.StartInfo.FileName = filePath;
                backendProcess.StartInfo.ArgumentList.Add(pipeName);
                //backendProcess.StartInfo.CreateNoWindow = true;
                backendProcess.Start();
                BackendStatus = BackendStatus.NotStarted;
                return true;
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
                return false;
            }
        }

        private bool PipeHandshake()
        {
            pipeManager = new PipeManager(pipeName);
            pipeManager.WaitForConnection();
            (iMax, jMax) = pipeManager.Handshake();
            return iMax > 0 && jMax > 0; // (0,0) is the error condition
        }

        private bool SendControlByte(byte controlByte)
        {
            return pipeManager.WriteByte(controlByte);
        }

        /// <summary>
        /// Initialises field arrays and constructs a request byte based on the null-ness of the field arguments.
        /// </summary>
        private byte CheckFieldParameters(float[]? horizontalVelocity, float[]? verticalVelocity, float[]? pressure, float[]? streamFunction)
        {
            if (pipeManager == null)
            {
                throw new InvalidOperationException("Cannot get data when pipe has not been opened");
            }

            int requestedFields = horizontalVelocity == null ? 0 : 1; // Sum up how many fields are not null
            requestedFields += verticalVelocity == null ? 0 : 1;
            requestedFields += pressure == null ? 0 : 1;
            requestedFields += streamFunction == null ? 0 : 1;

            if (requestedFields == 0)
            {
                throw new InvalidOperationException("No fields have been provided, cannot execute");
            }

            byte requestByte = PipeConstants.Request.CONTREQ;
            fields = new float[requestedFields][]; // A container for references to all the different fields
            int fieldNumber = 0;
            List<FieldType> namedFieldsList = new List<FieldType>();

            if (horizontalVelocity != null)
            {
                if (horizontalVelocity.Length < FieldLength)
                {
                    throw new InvalidOperationException("Field array is too small");
                }
                requestByte += PipeConstants.Request.HVEL;
                fields[fieldNumber] = horizontalVelocity;
                namedFieldsList.Add(FieldType.HorizontalVelocity);
                fieldNumber++;
            }
            if (verticalVelocity != null)
            {
                if (verticalVelocity.Length < FieldLength)
                {
                    throw new InvalidOperationException("Field array is too small");
                }
                requestByte += PipeConstants.Request.VVEL;
                fields[fieldNumber] = verticalVelocity;
                namedFieldsList.Add(FieldType.VerticalVelocity);
                fieldNumber++;
            }
            if (pressure != null)
            {
                if (pressure.Length < FieldLength)
                {
                    throw new InvalidOperationException("Field array is too small");
                }
                requestByte += PipeConstants.Request.PRES;
                fields[fieldNumber] = pressure;
                namedFieldsList.Add(FieldType.Pressure);
                fieldNumber++;
            }
            if (streamFunction != null)
            {
                if (streamFunction.Length < FieldLength)
                {
                    throw new InvalidOperationException("Field array is too small");
                }
                requestByte += PipeConstants.Request.STRM;
                fields[fieldNumber] = streamFunction;
                namedFieldsList.Add(FieldType.StreamFunction);
            }
            namedFields = namedFieldsList.ToArray();
            return requestByte;
        }

        private async void SendParameters()
        {
            while (!parameterSendQueue.IsEmpty)
            {
                ParameterChangedEventArgs args = parameterSendQueue.Dequeue();
                string parameterName = args.PropertyName;
                float parameterValue = args.NewValue;
                byte parameterBits = parameterName switch
                {
                    "Width" => PipeConstants.Marker.WIDTH,
                    "Height" => PipeConstants.Marker.HEIGHT,
                    "TimeStepSafetyFactor" => PipeConstants.Marker.TAU,
                    "RelaxationParameter" => PipeConstants.Marker.OMEGA,
                    "PressureResidualTolerance" => PipeConstants.Marker.RMAX,
                    "PressureMaxIterations" => PipeConstants.Marker.ITERMAX,
                    "ReynoldsNumber" => PipeConstants.Marker.REYNOLDS,
                    "InflowVelocity" => PipeConstants.Marker.INVEL,
                    "SurfaceFriction" => PipeConstants.Marker.CHI,
                    "FluidViscosity" => PipeConstants.Marker.MU,
                    "FluidDensity" => PipeConstants.Marker.DENSITY,
                    _ => 0,
                };

                if (parameterBits == 0) // Error case
                {
                    throw new InvalidOperationException("Parameter in queue was not recognised");
                }

                if (parameterBits == PipeConstants.Marker.ITERMAX) // Itermax is the only parameter that is an integer so needs special treatment
                {
                    pipeManager.SendParameter((int)parameterValue, parameterBits);
                }
                else
                {
                    pipeManager.SendParameter(parameterValue, parameterBits);
                }
                if (await pipeManager.ReadAsync() != PipeConstants.Status.OK)
                {
                    throw new IOException("Backend did not read parameters correctly");
                }
            }
        }

        private void HandleParameterChanged(object? sender, PropertyChangedEventArgs args)
        {
            parameterSendQueue.Enqueue((ParameterChangedEventArgs)args);
        }

        public BackendManager(ParameterHolder parameterHolder)
        {
            this.parameterHolder = parameterHolder;
            parameterHolder.PropertyChanged += HandleParameterChanged;

            fields = null;
            namedFields = null;

            parameterSendQueue = new();

            frameTimer = new Stopwatch();

#if NO_GPU_BACKEND
            if (File.Exists(".\\CPUBackend.exe"))
            {
                filePath = ".\\CPUBackend.exe"; // Look for CPUBackend in same directory...
            }
            else if (File.Exists("..\\..\\..\\..\\x64\\Debug\\CPUBackend.exe"))
            {
                filePath = "..\\..\\..\\..\\x64\\Debug\\CPUBackend.exe"; // ...then look in debug directory.
            }
            else
            {
                MessageBox.Show("Could not find backend executable. Make sure that CPUBackend.exe exists in the same folder as UserInterface.exe");
                throw new FileNotFoundException("Backend executable could not be found");
            }
#else // ^^ NO_GPU_BACKEND ^^ / vv !NO_GPU_BACKEND vv
            if (File.Exists(".\\GPUBackend.exe"))
            {
                filePath = ".\\GPUBackend.exe"; // First try to find GPU backend in same directory...
            }
            else if (File.Exists(".\\CPUBackend.exe"))
            {
                filePath = ".\\CPUBackend.exe"; // ...then look for CPU backend in same directory.
            }
            else if (File.Exists("..\\..\\..\\..\\x64\\Debug\\GPUBackend.exe"))
            {
                filePath = "..\\..\\..\\..\\x64\\Debug\\GPUBackend.exe"; // When debugging, backend executables are here. Try GPU backend first...
            }
            else if (File.Exists("..\\..\\..\\..\\x64\\Debug\\CPUBackend.exe"))
            {
                filePath = "..\\..\\..\\..\\x64\\Debug\\CPUBackend.exe"; // ...then try CPU backend.
            }
            else
            {
                MessageBox.Show("Could not find backend executable. Make sure that either GPUBackend.exe or CPUBackend.exe exists in the same folder as UserInterface.exe");
                throw new FileNotFoundException("Backend executable could not be found");
            }
#endif // !NO_GPU_BACKEND

            BackendStatus = BackendStatus.Closed;
        }

        /// <summary>
        /// Method to start and connect to the backend process
        /// </summary>
        /// <returns>Boolean result indicating whether the connection was successful</returns>
        public bool ConnectBackend()
        {
            return CreateBackend() && PipeHandshake(); // Return true only if both were successful. Also doesn't attempt handshake if backend did not start correctly
        }

        public async void SendAllParameters()
        {
            pipeManager.SendParameter(parameterHolder.Width.Value, PipeConstants.Marker.WIDTH);
            if (await pipeManager.ReadAsync() != PipeConstants.Status.OK) throw new IOException("Backend did not read parameters correctly");

            pipeManager.SendParameter(parameterHolder.Height.Value, PipeConstants.Marker.HEIGHT);
            if (await pipeManager.ReadAsync() != PipeConstants.Status.OK) throw new IOException("Backend did not read parameters correctly");

            pipeManager.SendParameter(parameterHolder.TimeStepSafetyFactor.Value, PipeConstants.Marker.TAU);
            if (await pipeManager.ReadAsync() != PipeConstants.Status.OK) throw new IOException("Backend did not read parameters correctly");

            pipeManager.SendParameter(parameterHolder.RelaxationParameter.Value, PipeConstants.Marker.OMEGA);
            if (await pipeManager.ReadAsync() != PipeConstants.Status.OK) throw new IOException("Backend did not read parameters correctly");

            pipeManager.SendParameter(parameterHolder.PressureResidualTolerance.Value, PipeConstants.Marker.RMAX);
            if (await pipeManager.ReadAsync() != PipeConstants.Status.OK) throw new IOException("Backend did not read parameters correctly");

            pipeManager.SendParameter((int)parameterHolder.PressureMaxIterations.Value, PipeConstants.Marker.ITERMAX);
            if (await pipeManager.ReadAsync() != PipeConstants.Status.OK) throw new IOException("Backend did not read parameters correctly");

            pipeManager.SendParameter(parameterHolder.ReynoldsNumber.Value, PipeConstants.Marker.REYNOLDS);
            if (await pipeManager.ReadAsync() != PipeConstants.Status.OK) throw new IOException("Backend did not read parameters correctly");

            pipeManager.SendParameter(parameterHolder.InflowVelocity.Value, PipeConstants.Marker.INVEL);
            if (await pipeManager.ReadAsync() != PipeConstants.Status.OK) throw new IOException("Backend did not read parameters correctly");

            pipeManager.SendParameter(parameterHolder.SurfaceFriction.Value, PipeConstants.Marker.CHI);
            if (await pipeManager.ReadAsync() != PipeConstants.Status.OK) throw new IOException("Backend did not read parameters correctly");

            pipeManager.SendParameter(parameterHolder.FluidDensity.Value, PipeConstants.Marker.DENSITY);
            if (await pipeManager.ReadAsync() != PipeConstants.Status.OK) throw new IOException("Backend did not read parameters correctly");

            pipeManager.SendParameter(parameterHolder.FluidViscosity.Value, PipeConstants.Marker.MU);
            if (await pipeManager.ReadAsync() != PipeConstants.Status.OK) throw new IOException("Backend did not read parameters correctly");
        }

        private async void ReceiveParameter(byte parameterBits)
        {
            if (parameterBits == PipeConstants.Marker.DRAGCOEF)
            {
                DragCoefficient = await pipeManager.ReadParameterAsync();
            }
        }

        /// <summary>
        /// Asynchronous method to repeatedly receive fields from the backend, for visualisation
        /// </summary>
        /// <param name="horizontalVelocity">Array to store horizontal velocity data</param>
        /// <param name="verticalVelocity">Array to store vertical velocity data</param>
        /// <param name="pressure">Array to store pressure data</param>
        /// <param name="streamFunction">Array to store stream function data</param>
        /// <param name="token">A cancellation token to stop the method and backend</param>
        /// <exception cref="InvalidOperationException">Thrown when parameters are invalid</exception>
        /// <exception cref="IOException">Thrown when backend does not respond as expected</exception>
        public async void GetFieldStreamsAsync(float[]? horizontalVelocity, float[]? verticalVelocity, float[]? pressure, float[]? streamFunction, CancellationToken token)
        {
            switch (BackendStatus)
            {
                case BackendStatus.NotStarted:
                    byte requestByte = CheckFieldParameters(horizontalVelocity, verticalVelocity, pressure, streamFunction); // Abstract the parameter checking into its own function

                    SendParameters(); // Send the parameters that were set before the simulation started

                    SendControlByte(requestByte); // Start the backend executing
                    byte receivedByte = await pipeManager.ReadAsync();
                    if (receivedByte != PipeConstants.Status.OK) // Should receive OK, then the backend will start executing
                    {
                        if ((receivedByte & PipeConstants.CATEGORYMASK) == PipeConstants.Error.GENERIC) // Throw an exception with the provided error code
                        {
                            throw new IOException($"Backend did not receive data correctly. Exception code {receivedByte}.");
                        }
                        throw new IOException("Result from backend not understood"); // Throw a generic error if it was not understood at all
                    }

                    break;

                case BackendStatus.Stopped: // Resuming from a paused state
                    if (parameterSendQueue.IsEmpty)
                    {
                        SendControlByte(PipeConstants.Status.OK);
                    }
                    else
                    {
                        SendParameters();
                        SendControlByte(PipeConstants.Status.OK);
                    }
                    break;
                case BackendStatus.Closed:
                    throw new IOException("Backend must be created and connected before calling GetFieldStreamsAsync.");
                default:
                    break;
            }

            byte[] tmpByteBuffer = new byte[FieldLength * sizeof(float)]; // Temporary buffer for pipe output

            frameTimer.Start(); // Start the timer and create a variable to hold the previous time.
            TimeSpan iterationStartTime = frameTimer.Elapsed;

            bool cancellationRequested = token.IsCancellationRequested;
            BackendStatus = BackendStatus.Running;

            while (!cancellationRequested) // Repeat until the task is cancelled
            {
                if (await pipeManager.ReadAsync() != PipeConstants.Marker.ITERSTART) throw new IOException("Backend did not send data correctly"); // Each timestep iteration should start with an ITERSTART

                for (int fieldNum = 0; fieldNum < fields.Length; fieldNum++)
                {
                    byte fieldBits = (byte)namedFields[fieldNum];
                    byte startMarker = await pipeManager.ReadAsync();
                    if (startMarker != (PipeConstants.Marker.FLDSTART | fieldBits)) throw new IOException($"Backend did not send data correctly. Bits were {startMarker}"); // Each field should start with a FLDSTART with the relevant field bits

                    await pipeManager.ReadAsync(tmpByteBuffer, FieldLength * sizeof(float)); // Read the stream of bytes into the temporary buffer
                    Buffer.BlockCopy(tmpByteBuffer, 0, fields[fieldNum], 0, FieldLength * sizeof(float)); // Copy the bytes from the temporary buffer into the double array
                    if (await pipeManager.ReadAsync() != (PipeConstants.Marker.FLDEND | fieldBits))  throw new IOException("Backend did not send data correctly"); // Each field should start with a FLDEND with the relevant field bits
                }
                byte nextByte = await pipeManager.ReadAsync();
                while ((nextByte & ~PipeConstants.Marker.PRMMASK) == PipeConstants.Marker.PRMSTART)
                {
                    byte parameterBits = (byte)(nextByte & PipeConstants.Marker.PRMMASK);
                    ReceiveParameter(parameterBits);
                    if (await pipeManager.ReadAsync() != (PipeConstants.Marker.PRMEND | parameterBits)) throw new IOException("Backend did not send data correctly");
                    nextByte = await pipeManager.ReadAsync();
                }

                if (nextByte != PipeConstants.Marker.ITEREND) throw new IOException("Backend did not send data correctly"); // Each timestep iteration should end with an ITEREND

                if (token.IsCancellationRequested)
                {
                    cancellationRequested = true;
                }
                else if (parameterSendQueue.IsEmpty)
                {
                    SendControlByte(PipeConstants.Status.OK);
                }
                else
                {
                    SendParameters();
                    SendControlByte(PipeConstants.Status.OK);
                }
                TimeSpan iterationLength = frameTimer.Elapsed - iterationStartTime;
                FrameTime = (float)iterationLength.TotalSeconds;

                iterationStartTime = frameTimer.Elapsed; // Set the new iteration start time once FPS processing is done.
            }

            SendControlByte(PipeConstants.Status.STOP); // Upon cancellation, stop (pause) the backend.
            BackendStatus = BackendStatus.Stopped;

            if (await pipeManager.ReadAsync() != PipeConstants.Status.OK) throw new IOException("Backend did not stop correctly");
        }

        public bool SendObstacles(bool[] obstacles)
        {
            return pipeManager.SendObstacles(obstacles);
        }

        public bool CloseBackend()
        {
            SendControlByte(PipeConstants.Status.CLOSE);
            if (pipeManager.AttemptRead().data[0] != PipeConstants.Status.OK)
            {
                return false;
            }
            if (!backendProcess.HasExited)
            {
                return false;
            }
            backendProcess.Close();
            return true;
        }

        public void ForceCloseBackend()
        {
            backendProcess.Kill();
        }
    }
}