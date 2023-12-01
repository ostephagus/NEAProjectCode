using OpenTK.Graphics.OpenGL4;
using OpenTK.Wpf;
using System;
using System.Diagnostics;
using System.Windows.Controls;
using Visualisation;

namespace UserInterface
{
    /// <summary>
    /// Interaction logic for VisualisationControl.xaml
    /// </summary>
    public partial class VisualisationControl : UserControl
    {
        private ShaderManager shaderManager;

        private float[] vertices;
        private uint[] indices;

        private int hVertexBuffer;
        private int hVertexArrayObject;
        private int hElementBuffer;
        private int hSubtrahend;
        private int hScalar;

        private float[] fieldValues;
        private float[] streamFunction;

        private int dataWidth;
        private int dataHeight;

        private float min;
        private float max;

        public float[] FieldValues { get => fieldValues; set => fieldValues = value; }
        public int DataWidth { get => dataWidth; set => dataWidth = value; }
        public int DataHeight { get => dataHeight; set => dataHeight = value; }
        public float Min { get => min; set => min = value; }
        public float Max { get => max; set => max = value; }
        public float[] StreamFunction { get => streamFunction; set => streamFunction = value; }

        public VisualisationControl()
        {
            InitializeComponent();
            DataContext = this;

            //width = 10;
            //height = 10;
            //fieldValues = new float[width * height];
            //for (int i = 0; i < width; i++)
            //{
            //    for (int j = 0; j < height; j++)
            //    {
            //        fieldValues[i * width + j] = (float)i / width;
            //    }
            //}

            //SetUpGL();
        }

        public VisualisationControl(float[] fieldValues, float[] streamFunction, int dataWidth, int dataHeight, float min, float max)
        {
            FieldValues = fieldValues;
            StreamFunction = streamFunction;
            DataWidth = dataWidth;
            DataHeight = dataHeight;
            Min = min;
            Max = max;

            InitializeComponent();
            DataContext = this;

            SetUpGL();
        }

        public void Start()
        {
            SetUpGL();
        }

        private void SetUpGL()
        {
            GLWpfControlSettings settings = new() { MajorVersion = 3, MinorVersion = 1 };
            GLControl.Start(settings);

            shaderManager = new("shader.vert", "shader.frag");

            //GL.ClearColor(0.1f, 0.7f, 0.5f, 1.0f);
            vertices = GLHelper.FillVertices(dataWidth, dataHeight);
            hVertexBuffer = GLHelper.CreateVBO(vertices.Length + fieldValues.Length);
            hVertexArrayObject = GLHelper.CreateVAO();

            GLHelper.BufferSubData(vertices, 0);
            //Trace.WriteLine(GL.GetError().ToString());
            GLHelper.BufferSubData(fieldValues, vertices.Length);
            //Trace.WriteLine(GL.GetError().ToString());

            GLHelper.CreateAttribPointer(0, 2, 2, 0);
            GLHelper.CreateAttribPointer(1, 1, 1, vertices.Length);

            indices = GLHelper.FillIndices(dataWidth, dataHeight);
            hElementBuffer = GLHelper.CreateEBO(indices);

            hSubtrahend = shaderManager.GetUniformLocation("subtrahend");
            hScalar = shaderManager.GetUniformLocation("scalar");
            shaderManager.Use();
            shaderManager.SetUniform(hSubtrahend, min);
            shaderManager.SetUniform(hScalar, 1 / (max - min));
        }

        public void GLControl_OnRender(TimeSpan delta)
        {
            GL.Clear(ClearBufferMask.ColorBufferBit);

            GLHelper.BufferSubData(fieldValues, vertices.Length); // Update the field values

            shaderManager.Use();

            GLHelper.Draw(hVertexArrayObject, indices);

            ErrorCode errorCode = GL.GetError();
            if (errorCode != ErrorCode.NoError)
            {
                Trace.WriteLine("\x1B[31m" + errorCode.ToString() + "\033[0m");
            }
        }
    }
}
