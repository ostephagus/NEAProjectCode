using OpenTK.Graphics.OpenGL4;
using OpenTK.Wpf;
using System;
using System.Windows.Controls;
using Visualisation;

namespace UserInterface
{
    /// <summary>
    /// Interaction logic for VisualisationControl.xaml
    /// </summary>
    public partial class VisualisationControl : UserControl
    {
        Random rng = new Random(); // Testing

        ShaderManager shaderManager;

        float[] fieldValues;
        int width;
        int height;

        float[] vertices;
        uint[] indices;

        int hVertexBuffer;
        int hVertexArrayObject;
        int hElementBuffer;
        public VisualisationControl()
        {
            InitializeComponent();
            DataContext = this;

            width = 5;
            height = 5;
            fieldValues = new float[width * height];
            for (int i = 0; i < width; i++)
            {
                for (int j = 0; j < height; j++)
                {
                    fieldValues[i * width + j] = 1f;
                }
            }

            GLWpfControlSettings settings = new GLWpfControlSettings { MajorVersion = 4, MinorVersion = 3 };
            GLControl.Start(settings);

            shaderManager = new("shader.vert", "shader.frag");
            SetUpGL();
        }

        private void SetUpGL()
        {
            GL.ClearColor(0.1f, 0.7f, 0.5f, 1.0f);

            vertices = OpenGLHelper.FillVertices(fieldValues, width, height);
            hVertexBuffer = OpenGLHelper.CreateVBO(vertices);
            hVertexArrayObject = OpenGLHelper.CreateVAO();

            OpenGLHelper.CreateAttribPointer(0, 2, 3, 0);
            OpenGLHelper.CreateAttribPointer(1, 1, 3, 1);

            indices = OpenGLHelper.FillIndices(width, height);
            hElementBuffer = OpenGLHelper.CreateEBO(indices);

            shaderManager.Use();
        }

        public void GLControl_OnRender(TimeSpan delta)
        {
            GL.Clear(ClearBufferMask.ColorBufferBit);

            shaderManager.Use();

            OpenGLHelper.Draw(hVertexArrayObject, indices);
        }
    }
}
