using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;

namespace UserInterface
{

    /// <summary>
    /// Handler class for dealing with the backend
    /// </summary>
    public class BackendManager
    {
        private Process? backendProcess;
        private string filePath;
        private PipeManager? pipeManager;
        private int iMax;
        private int jMax;

        public int FieldLength { get => iMax * jMax; }
        public int IMax { get => iMax; set => iMax = value; }
        public int JMax { get => jMax; set => jMax = value; }

        private bool StartBackend()
        {
            try
            {
                backendProcess = new Process();
                backendProcess.StartInfo.FileName = filePath;
                backendProcess.StartInfo.CreateNoWindow = true;
                backendProcess.Start();
                return true;
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
                return false;
            }
        }

        private bool PipeHandshake()
        {
            pipeManager = new PipeManager("NEAFluidDynamicsPipe");
            pipeManager.WaitForConnection();
            (iMax, jMax) = pipeManager.Handshake();
            return iMax > 0 && jMax > 0; // (0,0) is the error condition
        }

        private bool SendControlByte(byte controlByte)
        {
            return pipeManager.WriteByte(controlByte);
        }

        private (byte, double[][], FieldType[]) CheckFieldParameters(double[]? horizontalVelocity, double[]? verticalVelocity, double[]? pressure, double[]? streamFunction)
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
            double[][] fields = new double[requestedFields][]; // A container for references to all the different fields
            int fieldNumber = 0;
            List<FieldType> namedFields = new List<FieldType>();

            if (horizontalVelocity != null)
            {
                if (horizontalVelocity.Length < FieldLength)
                {
                    throw new InvalidOperationException("Field array is too small");
                }
                requestByte += PipeConstants.Request.HVEL;
                fields[fieldNumber] = horizontalVelocity;
                namedFields.Add(FieldType.HorizontalVelocity);
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
                namedFields.Add(FieldType.VerticalVelocity);
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
                namedFields.Add(FieldType.Pressure);
                fieldNumber++;
            }
            if (streamFunction != null)
            {
                if (pressure.Length < FieldLength)
                {
                    throw new InvalidOperationException("Field array is too small");
                }
                requestByte += PipeConstants.Request.STRM;
                fields[fieldNumber] = streamFunction;
                namedFields.Add(FieldType.StreamFunction);
            }
            return (requestByte, fields, namedFields.ToArray());
        }

        public BackendManager()
        {
            filePath = "..\\..\\..\\..\\x64\\Debug\\CPUBackend.exe"; // Relative filepath to CPUBackend.exe
        }

        public BackendManager(string executableFilePath)
        {
            filePath = executableFilePath;
        }

        /// <summary>
        /// Method to start and connect to the backend process
        /// </summary>
        /// <returns>Boolean result indicating whether the connection was successful</returns>
        public bool ConnectBackend()
        {
            return StartBackend() && PipeHandshake(); // Return true only if both were successful. Also doesn't attempt handshake if backend did not start correctly
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
        public async void GetFieldStreamsAsync(double[]? horizontalVelocity, double[]? verticalVelocity, double[]? pressure, double[]? streamFunction, CancellationToken token)
        {
            (byte requestByte, double[][] fields, FieldType[] namedFields) = CheckFieldParameters(horizontalVelocity, verticalVelocity, pressure, streamFunction); // Abstract the parameter checking into its own function

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
            
            byte[] tmpByteBuffer = new byte[FieldLength * 8]; // Temporary buffer for pipe output

            while (!token.IsCancellationRequested) // Repeat until the task is cancelled
            {
                if (await pipeManager.ReadAsync() != PipeConstants.Marker.ITERSTART) { throw new IOException("Backend did not send data correctly"); } // Each timestep iteration should start with an ITERSTART

                for (int fieldNum = 0; fieldNum < fields.Length; fieldNum++)
                {
                    byte fieldBits = (byte)namedFields[fieldNum];
                    if (await pipeManager.ReadAsync() != (PipeConstants.Marker.FLDSTART | fieldBits)) { throw new IOException("Backend did not send data correctly"); } // Each field should start with a FLDSTART with the relevant field bits
                    
                    await pipeManager.ReadAsync(tmpByteBuffer, FieldLength * 8); // Read the stream of bytes into the temporary buffer
                    for (int i = 0; i < tmpByteBuffer.Length; i++)
                    {
                        Console.WriteLine($"{i}: {tmpByteBuffer[i]}");
                    }
                    Buffer.BlockCopy(tmpByteBuffer, 0, fields[fieldNum], 0, FieldLength * 8); // Copy the bytes from the temporary buffer into the double array
                    if (await pipeManager.ReadAsync() != (PipeConstants.Marker.FLDEND | fieldBits)) { throw new IOException("Backend did not send data correctly"); } // Each field should start with a FLDEND with the relevant field bits
                }

                if (await pipeManager.ReadAsync() != PipeConstants.Marker.ITEREND) { throw new IOException("Backend did not send data correctly"); } // Each timestep iteration should end with an ITEREND
                SendControlByte(PipeConstants.Status.OK);
            }

            SendControlByte(PipeConstants.Status.STOP); // Send a request to stop the backend, and make sure its stops ok

            // May need to have logic to read the last field transmission before OK byte received

            if (await pipeManager.ReadAsync() != PipeConstants.Status.OK)
            {
                throw new IOException("Backend did not stop correctly");
            }
            // Backend stopped correctly, so exit.
        }

        public async Task<bool> StopBackend()
        {
            SendControlByte(PipeConstants.Status.CLOSE);
            if (await pipeManager.ReadAsync() != PipeConstants.Status.OK)
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

        public void ForceClose()
        {
            backendProcess.Close();
        }
    }
}