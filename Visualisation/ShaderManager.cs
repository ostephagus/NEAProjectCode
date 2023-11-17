using OpenTK.Graphics.OpenGL4;

namespace Visualisation
{
    public class ShaderManager : IDisposable
    {
        private int hProgram;
        private bool isDisposed = false;

        public int Handle { get => hProgram; set => hProgram = value; }

        private static void ExtractShaderSources(string vertexPath, string fragmentPath, out int hVertexShader, out int hFragmentShader)
        {
            string vertexShaderSourceCode = File.ReadAllText(vertexPath);
            string fragmentShaderSourceCode = File.ReadAllText(fragmentPath);

            hVertexShader = GL.CreateShader(ShaderType.VertexShader);
            GL.ShaderSource(hVertexShader, vertexShaderSourceCode);

            hFragmentShader = GL.CreateShader(ShaderType.FragmentShader);
            GL.ShaderSource(hFragmentShader, fragmentShaderSourceCode);
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
        private void LinkShaders(int hVertexShader, int hFragmentShader)
        {
            hProgram = GL.CreateProgram();

            GL.AttachShader(hProgram, hVertexShader);
            GL.AttachShader(hProgram, hFragmentShader);

            GL.LinkProgram(hProgram);

            GL.GetProgram(hProgram, GetProgramParameterName.LinkStatus, out int success);
            if (success == 0)
            {
                Console.WriteLine(GL.GetProgramInfoLog(hProgram));
            }
        }


        private void BuildProgram(string vertexPath, string fragmentPath)
        {
            ExtractShaderSources(vertexPath, fragmentPath, out int hVertexShader, out int hFragmentShader);
            CompileShaders(hVertexShader, hFragmentShader);
            LinkShaders(hVertexShader, hFragmentShader);

            GL.DetachShader(hProgram, hVertexShader);
            GL.DetachShader(hProgram, hFragmentShader);

            GL.DeleteShader(hVertexShader);
            GL.DeleteShader(hFragmentShader);
        }

        public ShaderManager(string vertexPath, string fragmentPath)
        {
            BuildProgram(vertexPath, fragmentPath);
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
            GL.UseProgram(hProgram);
        }

        public int GetUniformLocation(string uniformName)
        {
            return GL.GetUniformLocation(hProgram, uniformName);
        }

        public void SetUniform(int uniformLocation, float value)
        {
            GL.Uniform1(uniformLocation, value);
        }

        public void Dispose()
        {
            if (!isDisposed)
            {
                GL.DeleteProgram(hProgram);
                isDisposed = true;
            }
            GC.SuppressFinalize(this);
        }
    }
}
