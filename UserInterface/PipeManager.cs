using System;
using System.Collections.Generic;
using System.IO.Pipes;
using System.Linq;
using System.Numerics;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Media.Imaging;

namespace UserInterface
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
        private NamedPipeServerStream pipeStream;

        public PipeManager(string pipeName)
        {
            pipeStream = new NamedPipeServerStream(pipeName);
        }
        /// <summary>
        /// Reads one byte asynchronously
        /// </summary>
        /// <returns>A task to read the byte from the pipe, when one is available</returns>
        public Task<byte> ReadAsync()
        {
            TaskCompletionSource<byte> taskCompletionSource = new TaskCompletionSource<byte>();

            byte[] buffer = new byte[1];
            pipeStream.Read(buffer, 0, 1); //Read one byte. ReadByte method is not used because that returns -1 if there is nothing to read, whereas we want to wait until there is data available which Read does

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
            int currentByte = pipeStream.ReadByte();
            if (currentByte == -1) // Nothing to read - return nothing read
            {
                return new ReadResults { anythingRead = false, data = new byte[1] };
            }
            byte[] buffer = new byte[1024]; // Create a new buffer 1kiB
            int index = 0;
            do
            {
                buffer[index] = (byte)currentByte; // Put the current byte in the buffer
                index++;
                if (index == buffer.Length) // Double the size of the array if it becomes inadequate
                {
                    Array.Resize(ref buffer, 2 * index);
                }
                currentByte = pipeStream.ReadByte();
            } while (currentByte != -1);

            Array.Resize(ref buffer, index); // Resize the array to be only as long as is needed
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
                Console.WriteLine(e.Message);
                return false;
            }
        }

        /// <summary>
        /// Performs a handshake with the client where server dictates the field length
        /// </summary>
        /// <param name="fieldLength">The size of the simulation domain</param>
        /// <returns>true if successful, false if handshake failed</returns>
        public bool Handshake(int fieldLength)
        {
            byte[] buffer = new byte[5];
            buffer[0] = (byte)(PipeConstants.Status.HELLO | 4); // Send a HELLO bit with parameter 4 (next 4 bytes are relevant)
            for (int i = 0; i < sizeof(uint); i++)
            {
                buffer[i+1] = (byte)(fieldLength >> (i * 8));
            }
            pipeStream.Write(new ReadOnlySpan<byte>(buffer));
            pipeStream.WaitForPipeDrain();
            ReadResults readResults = AttemptRead();
            if (readResults.anythingRead == false || readResults.data[0] != PipeConstants.Status.HELLO) // If nothing was read, handshake not completed properly
            {
                return false;
            }
            return true; // If a HELLO byte is read, as expected, handshake was successful

        }
        /// <summary>
        /// Performs a handshake with the client where the client dictates the field length
        /// </summary>
        /// <returns>The field length, or 0 if handshake failed</returns>
        public int Handshake()
        {
            pipeStream.WriteByte(PipeConstants.Status.HELLO); // Write a HELLO byte with no parameters, backend dictates the field length
            pipeStream.WaitForPipeDrain();
            ReadResults readResults = AttemptRead(); // First byte should be HELLO with parameter 4, should be 4 more bytes with the uint
            if (!readResults.anythingRead)
            {
                return 0;
            }
            int fieldLength = 0;
            int followingBytes = readResults.data[0] & PipeConstants.Status.PARAMMASK;
            if (followingBytes > 0) // If the HELLO byte has parameter other than 0
            {
                for (int i = 0; i < followingBytes; i++) // Read the next n bytes
                {
                    fieldLength += readResults.data[i + 1] << (i * 8);
                }
            }
            return fieldLength;
        }

        public void WaitForConnection()
        {
            pipeStream.WaitForConnection();
        }

        public bool SendObstacles(bool[] obstacles)
        {
            byte[] buffer = new byte[obstacles.Length / 8 + (obstacles.Length % 8 == 0 ? 0 : 1)]; // Divide the length by 8 and add one if the length does not divide evenly
            int index = 0;
            for (int i = 0; i < obstacles.Length; i++)
            {
                buffer[index] |= (byte)((obstacles[i] ? 1 : 0) << (i % 8)); // Convert the bool to 1 or 0, shift it left the relevant amount of times and OR it with the current value in the buffer
                if (i % 8 == 7) // Add one to the index if the byte is full
                {
                    index++;
                }
            }
            WriteByte((byte)(PipeConstants.Marker.FLDSTART | PipeConstants.Marker.OBST)); // Put a FLDSTART marker at the start
            pipeStream.Write(new ReadOnlySpan<byte>(buffer)); // Write the data
            WriteByte((byte)(PipeConstants.Marker.FLDEND | PipeConstants.Marker.OBST)); // And put a FLDEND at the end
            int readByte = -1;
            while (readByte == -1) // Wait until there is a response
            {
                readByte = pipeStream.ReadByte();
            }
            return (byte)readByte == PipeConstants.Status.OK;
        }

    }
}
