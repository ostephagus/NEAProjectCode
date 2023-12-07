using OpenTK.Graphics.OpenGL4;

namespace Visualisation
{
    public class ShaderManager : IDisposable
    {
        private int programHandle;
        private bool isDisposed = false;

        public int Handle { get => programHandle; set => programHandle = value; }

        private static void ExtractShaderSources(string vertexPath, string fragmentPath, out int hVertexShader, out int hFragmentShader)
        {
            string vertexShaderSourceCode = File.ReadAllText(vertexPath);
            string fragmentShaderSourceCode = File.ReadAllText(fragmentPath);

            hVertexShader = GL.CreateShader(ShaderType.VertexShader);
            GL.ShaderSource(hVertexShader, vertexShaderSourceCode);

            hFragmentShader = GL.CreateShader(ShaderType.FragmentShader);
            GL.ShaderSource(hFragmentShader, fragmentShaderSourceCode);
        }

        private static void ExtractShaderSource(string path, ShaderType type, out int shaderHandle)
        {
            string shaderSource = File.ReadAllText(path);

            shaderHandle = GL.CreateShader(type);
            GL.ShaderSource(shaderHandle, shaderSource);
        }

        private static void CompileShaders(int hVertexShader, int hFragmentShader)
        {
            GL.CompileShader(hVertexShader);
            GL.GetShader(hVertexShader, ShaderParameter.CompileStatus, out int success);
            if (success == 0)
            {
                Console.WriteLine(GL.GetShaderInfoLog(hVertexShader));
            }

            GL.CompileShader(hFragmentShader);
            GL.GetShader(hFragmentShader, ShaderParameter.CompileStatus, out success);
            if (success == 0)
            {
                Console.WriteLine(GL.GetShaderInfoLog(hFragmentShader));
            }
        }

        private static void CompileShader(int shaderHandle)
        {
            GL.CompileShader(shaderHandle);
            GL.GetShader(shaderHandle, ShaderParameter.CompileStatus, out int success);
            if (success == 0) // Error in compilation
            {
                Console.WriteLine(GL.GetShaderInfoLog(shaderHandle));
            }
        }

        private void LinkShaders(int hVertexShader, int hFragmentShader)
        {
            programHandle = GL.CreateProgram();

            GL.AttachShader(programHandle, hVertexShader);
            GL.AttachShader(programHandle, hFragmentShader);

            GL.LinkProgram(programHandle);

            GL.GetProgram(programHandle, GetProgramParameterName.LinkStatus, out int success);
            if (success == 0)
            {
                Console.WriteLine(GL.GetProgramInfoLog(programHandle));
            }
        }

        private void LinkShaders(int[] shaderHandles)
        {
            programHandle = GL.CreateProgram();

            foreach (int shaderHandle in shaderHandles)
            {
                GL.AttachShader(programHandle, shaderHandle);
            }

            GL.LinkProgram(programHandle);

            GL.GetProgram(programHandle, GetProgramParameterName.LinkStatus, out int success);
            if (success == 0) // Error case
            {
                Console.WriteLine(GL.GetProgramInfoLog(programHandle));
            }
        }


        private void BuildProgram((string, ShaderType)[] shadersWithPaths)
        {
            int[] handles = new int[shadersWithPaths.Length];
            for (int i = 0; i < shadersWithPaths.Length; i++)
            {
                ExtractShaderSource(shadersWithPaths[i].Item1, shadersWithPaths[i].Item2, out handles[i]);
                CompileShader(handles[i]);
            }
            LinkShaders(handles);

            foreach(int shaderHandle in handles)
            {
                GL.DetachShader(programHandle, shaderHandle);
                GL.DeleteShader(shaderHandle);
            }
        }

        public ShaderManager((string, ShaderType)[] shadersWithPaths)
        {
            BuildProgram(shadersWithPaths);
        }

        ~ShaderManager()
        {
            if (!isDisposed)
            {
                Console.WriteLine("Object not disposed of correctly");
                throw new InvalidOperationException("Object was not disposed of correctly.");
            }
        }

        public void Use()
        {
            GL.UseProgram(programHandle);
        }

        public int GetUniformLocation(string uniformName)
        {
            return GL.GetUniformLocation(programHandle, uniformName);
        }

        public void SetUniform(int uniformLocation, float value)
        {
            GL.Uniform1(uniformLocation, value);
        }

        public void Dispose()
        {
            if (!isDisposed)
            {
                GL.DeleteProgram(programHandle);
                isDisposed = true;
            }
            GC.SuppressFinalize(this);
        }
    }
}
