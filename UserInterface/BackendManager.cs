using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

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
        private uint fieldLength;

        private bool StartBackend()
        {
            try
            {
                backendProcess = new Process();
                backendProcess.StartInfo.FileName = filePath;
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
            fieldLength = pipeManager.Handshake();
            return fieldLength > 0; // FieldLength == 0 is the error condition for Handshake()
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

            return true;
        }
    }
}
