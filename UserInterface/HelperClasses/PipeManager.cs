using System;
using System.Diagnostics;
using System.IO.Pipes;
using System.Threading.Tasks;

namespace UserInterface.HelperClasses
{
    /// <summary>
    /// Struct enclosing a bool, specifying whether a read operation happened, and a buffer for the read operation output (if applicable)
    /// </summary>
    public struct ReadResults
    {
        public bool anythingRead;
        public byte[] data;
    }

    public enum FieldType
    {
        HorizontalVelocity = 1,
        VerticalVelocity = 2,
        Pressure = 3,
        StreamFunction = 4
    }

    /// <summary>
    /// Helper class for managing the pipe communication with the C++ backend
    /// </summary>
    public class PipeManager
    {
        private readonly NamedPipeServerStream pipeStream;

        public PipeManager(string pipeName)
        {
            pipeStream = new NamedPipeServerStream(pipeName);
        }

        /// <summary>
        /// Serialises an integer into part of a buffer.
        /// </summary>
        /// <param name="buffer">The <c>byte[] to store the result in.</c></param>
        /// <param name="offset">The index in which to store the first element.</param>
        /// <param name="datum">The datum to store.</param>
        private static void SerialisePrimitive(byte[] buffer, int offset, int datum)
        {
            for (int i = 0; i < sizeof(int); i++)
            {
                buffer[i + offset] = (byte)(datum >> i * 8);
            }
        }

        /// <summary>
        /// Serialises a float into part of a buffer.
        /// </summary>
        /// <param name="buffer">The <c>byte[] to store the result in.</c></param>
        /// <param name="offset">The index in which to store the first element.</param>
        /// <param name="datum">The datum to store.</param>
        private static void SerialisePrimitive(byte[] buffer, int offset, float datum)
        {
            byte[] serialisedPrimitive = BitConverter.GetBytes(datum);
            Buffer.BlockCopy(serialisedPrimitive, 0, buffer, offset, sizeof(float));
        }

        /// <summary>
        /// Reads one byte asynchronously
        /// </summary>
        /// <returns>A task to read the byte from the pipe, when one is available</returns>
        public Task<byte> ReadAsync()
        {
            TaskCompletionSource<byte> taskCompletionSource = new TaskCompletionSource<byte>();

            byte[] buffer = new byte[1];
            pipeStream.Read(buffer, 0, 1); // Read one byte. ReadByte method is not used because that returns -1 if there is nothing to read, whereas we want to wait until there is data available which Read does

            taskCompletionSource.SetResult(buffer[0]);
            return taskCompletionSource.Task;
        }

        public Task<bool> ReadAsync(byte[] buffer, int count)
        {
            TaskCompletionSource<bool> taskCompletionSource = new TaskCompletionSource<bool>();
            pipeStream.Read(buffer, 0, count);
            taskCompletionSource.SetResult(true);
            return taskCompletionSource.Task;
        }

        /// <summary>
        /// Attempts a read operation of the pipe stream
        /// </summary>
        /// <returns>A ReadResults struct, including whether any data was read and the data (if applicable)</returns>
        public ReadResults AttemptRead()
        {
            byte[] buffer = new byte[1024]; // Start by reading 1kiB of the pipe
            int bytesRead = pipeStream.Read(buffer, 0, buffer.Length);
            if (bytesRead == 0)
            {
                return new ReadResults { anythingRead = false, data = new byte[1] };
            }
            int offset = 1;
            while (bytesRead == 1024) // While the buffer gets filled
            {
                Array.Resize(ref buffer, 1024 * (offset + 1)); // Resize the buffer by 1kiB
                bytesRead = pipeStream.Read(buffer, offset * 1024, 1024); // Read the next 1k bytes
                offset++;
            }
            Array.Resize(ref buffer, (offset - 1) * 1024 + bytesRead); // Resize the buffer to the actual length of data
            return new ReadResults { anythingRead = true, data = buffer };
        }

        public async Task<ReadResults> ReadFieldAsync(FieldType field, int fieldLength)
        {
            ReadResults readResults = new ReadResults();
            byte[] buffer = new byte[fieldLength];
            if (await ReadAsync() != (PipeConstants.Marker.GENERIC | (byte)field)) // If the received byte is not a marker with the correct field
            {
                readResults.anythingRead = false;
                return readResults;
            }

            pipeStream.Read(buffer, 0, fieldLength);
            readResults.anythingRead = true;
            readResults.data = buffer;

            return readResults;
        }

        /// <summary>
        /// Writes a single byte to the pipe
        /// </summary>
        /// <param name="b">The byte to write</param>
        /// <returns></returns>
        public bool WriteByte(byte b)
        {
            try
            {
                pipeStream.WriteByte(b);
                return true;
            }
            catch (Exception e)
            {
                Trace.WriteLine(e.Message);
                return false;
            }
        }

        /// <summary>
        /// Performs a handshake with the client where server dictates the field length
        /// </summary>
        /// <param name="fieldLength">The size of the simulation domain</param>
        /// <returns>true if successful, false if handshake failed</returns>
        public bool Handshake(int iMax, int jMax)
        {
            byte[] buffer = new byte[12];
            WriteByte(PipeConstants.Status.HELLO); // Send a HELLO byte
            if (AttemptRead().data[0] != PipeConstants.Status.HELLO) // Handshake not completed
            {
                return false;
            }

            pipeStream.WaitForPipeDrain();

            buffer[0] = (byte)(PipeConstants.Marker.PRMSTART | PipeConstants.Marker.IMAX); // Send PRMSTART with iMax
            SerialisePrimitive(buffer, 1, iMax);
            buffer[5] = (byte)(PipeConstants.Marker.PRMEND | PipeConstants.Marker.IMAX); // Send corresponding PRMEND

            buffer[6] = (byte)(PipeConstants.Marker.PRMSTART | PipeConstants.Marker.JMAX); // Send PRMSTART with jMax
            SerialisePrimitive(buffer, 7, jMax);
            buffer[11] = (byte)(PipeConstants.Marker.PRMEND | PipeConstants.Marker.IMAX); // Send PRMEND

            pipeStream.Write(new ReadOnlySpan<byte>(buffer));

            pipeStream.WaitForPipeDrain();

            ReadResults readResults = AttemptRead();
            if (readResults.anythingRead == false || readResults.data[0] != PipeConstants.Status.OK) // If nothing was read or no OK byte, param read was unsuccessful
            {
                return false;
            }
            return true;

        }
        /// <summary>
        /// Performs a handshake with the client where the client dictates the field length
        /// </summary>
        /// <returns>The field length, or 0 if handshake failed</returns>
        public (int, int) Handshake()
        {
            pipeStream.WriteByte(PipeConstants.Status.HELLO); // Write a HELLO byte, backend dictates field dimensions
            pipeStream.WaitForPipeDrain();

            ReadResults readResults = AttemptRead();
            if (!readResults.anythingRead || readResults.data[0] != PipeConstants.Status.HELLO)
            {
                return (0, 0); // Error case
            }

            if (readResults.data[1] != (PipeConstants.Marker.PRMSTART | PipeConstants.Marker.IMAX)) { return (0, 0); } // Should start with PRMSTART
            int iMax = BitConverter.ToInt32(readResults.data, 2);
            if (readResults.data[6] != (PipeConstants.Marker.PRMEND | PipeConstants.Marker.IMAX)) { return (0, 0); } // Should end with PRMEND

            if (readResults.data[7] != (PipeConstants.Marker.PRMSTART | PipeConstants.Marker.JMAX)) { return (0, 0); }
            int jMax = BitConverter.ToInt32(readResults.data, 8);
            if (readResults.data[12] != (PipeConstants.Marker.PRMEND | PipeConstants.Marker.JMAX)) { return (0, 0); }

            WriteByte(PipeConstants.Status.OK); // Send an OK byte to show the transmission was successful

            return (iMax, jMax);
        }

        /// <summary>
        /// Wrapper method that waits for the backend to connect to the pipe.
        /// </summary>
        public void WaitForConnection()
        {
            pipeStream.WaitForConnection();
        }

        public void CloseConnection()
        {
            pipeStream.Close();
        }

        /// <summary>
        /// Sends a parameter to the backend
        /// </summary>
        /// <param name="parameter">The value of the parameter to send</param>
        /// <param name="bits">The bits corresponding to the parameter, as read from <c>PipeConstants</c></param>
        public void SendParameter(float parameter, byte bits)
        {
            byte[] buffer = new byte[6];
            buffer[0] = (byte)(PipeConstants.Marker.PRMSTART | bits);
            SerialisePrimitive(buffer, 1, parameter);
            buffer[5] = (byte)(PipeConstants.Marker.PRMEND | bits);
            pipeStream.Write(buffer, 0, buffer.Length);
        }

        /// <summary>
        /// Sends a parameter to the backend
        /// </summary>
        /// <param name="parameter">The value of the parameter to send</param>
        /// <param name="bits">The bits corresponding to the parameter, as read from <c>PipeConstants</c></param>
        public void SendParameter(int parameter, byte bits)
        {
            byte[] buffer = new byte[2 + sizeof(float)];
            buffer[0] = (byte)(PipeConstants.Marker.PRMSTART | bits);
            SerialisePrimitive(buffer, 1, parameter);
            buffer[1 + sizeof(float)] = (byte)(PipeConstants.Marker.PRMEND | bits);
            pipeStream.Write(buffer, 0, buffer.Length);
        }

        public async Task<float> ReadParameterAsync()
        {
            byte[] buffer = new byte[sizeof(float)];
            await ReadAsync(buffer, sizeof(float));
            return BitConverter.ToSingle(buffer, 0);
        }

        /// <summary>
        /// Serialises and sends obstacle data through the pipe.
        /// </summary>
        /// <param name="obstacles">A boolean array indicating whether each cell is an obstacle cell or fluid cell.</param>
        /// <returns>A boolean indicating whether the transmission was successful.</returns>
        public bool SendObstacles(bool[] obstacles)
        {
            byte[] buffer = new byte[obstacles.Length / 8 + (obstacles.Length % 8 == 0 ? 0 : 1) + 1]; // Divide the length by 8 and add one if the length does not divide evenly. Also add 1 byte for FLDEND
            WriteByte((byte)(PipeConstants.Marker.FLDSTART | PipeConstants.Marker.OBST)); // Put a FLDSTART marker at the start
            int index = 0;
            for (int i = 0; i < obstacles.Length; i++)
            {
                buffer[index] |= (byte)((obstacles[i] ? 1 : 0) << i % 8); // Convert the bool to 1 or 0, shift it left the relevant amount of times and OR it with the current value in the buffer
                if (i % 8 == 7) // Add one to the index if the byte is full
                {
                    index++;
                }
            }
            buffer[^1] = (byte)(PipeConstants.Marker.FLDEND | PipeConstants.Marker.OBST); // And put a FLDEND at the end (^1 gets the last element of the array)

            pipeStream.Write(buffer, 0, buffer.Length);

            ReadResults readResults = AttemptRead();

            return readResults.anythingRead && readResults.data[0] == PipeConstants.Status.OK;
        }

    }
}
