using OpenTK.Graphics.OpenGL4;

namespace Visualisation
{
    public class ComputeShaderManager : ShaderManager
    {
        public ComputeShaderManager(string path) : base(new (string, ShaderType)[] { (path, ShaderType.ComputeShader) }) { }
    }
}
