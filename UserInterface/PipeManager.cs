using System;
using System.Collections.Generic;
using System.IO.Pipes;
using System.Linq;
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

    /// <summary>
    /// Helper class for managing the pipe communication with the C++ backend
    /// </summary>
    public class PipeManager
    {
        private NamedPipeServerStream pipeStream;

        private void Write(ReadOnlySpan<byte> data)
        {
            pipeStream.Write(data);
        }

        public PipeManager(string pipeName)
        {
            pipeStream = new NamedPipeServerStream(pipeName);
        }

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

        /// <summary>
        /// Performs a handshake with the client where server dictates the field length
        /// </summary>
        /// <param name="fieldLength">The size of the simulation domain</param>
        /// <returns>true if successful, false if handshake failed</returns>
        public bool Handshake(uint fieldLength)
        {
            byte[] buffer = new byte[5];
            buffer[0] = (byte)(PipeConstants.Status.HELLO | 4); // Send a HELLO bit with parameter 4 (next 4 bytes are relevant)
            for (int i = 0; i < sizeof(uint); i++)
            {
                buffer[i+1] = (byte)(fieldLength >> (i * 8));
            }
            Write(new ReadOnlySpan<byte>(buffer));
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
        public uint Handshake()
        {
            pipeStream.WriteByte(PipeConstants.Status.HELLO); // Write a HELLO byte with no parameters, backend dictates the field length
            pipeStream.WaitForPipeDrain();
            ReadResults readResults = AttemptRead(); // First byte should be HELLO with parameter 4, should be 4 more bytes with the uint
            if (!readResults.anythingRead)
            {
                return 0;
            }
            uint fieldLength = 0;
            int followingBytes;
            if ((followingBytes = readResults.data[0] & PipeConstants.Status.PARAMMASK) > 0) // If the HELLO byte has parameter other than 0
            {
                for (int i = 0; i < followingBytes; i++) // Read the next n bytes
                {
                    fieldLength += (uint)(readResults.data[i + 1] << (i * 8));
                }
            }
            return fieldLength;
        }

        public void WaitForConnection()
        {
            pipeStream.WaitForConnection();
        }
    }
}
