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
        private ShaderManager fieldShaderManager;
        private ShaderManager contourShaderManager;
        private ComputeShaderManager computeShaderManager;

        private float[] vertices;
        private uint[] fieldIndices;
        private uint[] contourIndices;

        private const float contourTolerance = 0.001f;
        private const float contourSpacingMultiplier = 0.1f;

        private int hVBO;
        private int hFieldVAO;
        private int hFieldEBO;

        private int hContourVAO;
        private int hContourEBO;

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

        ~VisualisationControl()
        {
            fieldShaderManager.Dispose();
        }

        public void Start()
        {
            SetUpGL();
        }

        // Compute shader plan:
        // Take the coordinates of a point, take the field value at that point, if the field value is a multiple of some spacing multiplier then add the coordinate to an EBO to be drawn using GL_LINE_STRIP

        private void SetUpGL()
        {
            GLWpfControlSettings settings = new() { MajorVersion = 3, MinorVersion = 1 };
            GLControl.Start(settings);

            fieldShaderManager = new ShaderManager(new (string, ShaderType)[] { ("fieldShader.frag", ShaderType.FragmentShader), ("fieldShader.vert", ShaderType.VertexShader) });
            contourShaderManager = new ShaderManager(new (string, ShaderType)[] { ("contourShader.frag", ShaderType.FragmentShader), ("contourShader.vert", ShaderType.VertexShader) });
            //computeShaderManager = new ComputeShaderManager("shader.comp");
            HandleData();
        }

        private void HandleData()
        {
            //GL.ClearColor(0.1f, 0.7f, 0.5f, 1.0f);

            vertices = GLHelper.FillVertices(dataWidth, dataHeight);
            fieldIndices = GLHelper.FillIndices(dataWidth, dataHeight);
            contourIndices = GLHelper.FindContourIndices(streamFunction, contourTolerance, contourSpacingMultiplier, dataWidth, dataHeight);

            // Setting up data for field visualisation
            hFieldVAO = GLHelper.CreateVAO();
            hVBO = GLHelper.CreateVBO(vertices.Length + fieldValues.Length);

            GLHelper.BufferSubData(vertices, 0);
            //Trace.WriteLine(GL.GetError().ToString());
            GLHelper.BufferSubData(fieldValues, vertices.Length);
            //Trace.WriteLine(GL.GetError().ToString());

            GLHelper.CreateAttribPointer(0, 2, 2, 0); // Vertex pointer
            GLHelper.CreateAttribPointer(1, 1, 1, vertices.Length); // Field value pointer

            hFieldEBO = GLHelper.CreateEBO(fieldIndices);

            // Setting up data for contour line plotting
            hContourVAO = GLHelper.CreateVAO();
            GL.BindBuffer(BufferTarget.ArrayBuffer, hVBO); // Bind the same VBO

            GLHelper.CreateAttribPointer(0, 2, 2, 0); // And the same for attribute pointers
            GLHelper.CreateAttribPointer(1, 1, 1, vertices.Length);

            hContourEBO = GLHelper.CreateEBO(fieldIndices);

            // Return to field context
            GL.BindVertexArray(hFieldVAO);

            hSubtrahend = fieldShaderManager.GetUniformLocation("subtrahend");
            hScalar = fieldShaderManager.GetUniformLocation("scalar");
        }

        public void GLControl_OnRender(TimeSpan delta)
        {
            GL.Clear(ClearBufferMask.ColorBufferBit);

            // For each draw command, need to bind the program, set uniforms, bind VAO, draw

            // Drawing field value spectrum
            fieldShaderManager.Use();

            fieldShaderManager.SetUniform(hSubtrahend, min);
            fieldShaderManager.SetUniform(hScalar, 1 / (max - min));

            GL.BindVertexArray(hFieldVAO);
            GLHelper.BufferSubData(fieldValues, vertices.Length); // Update the field values

            GLHelper.Draw(fieldIndices, PrimitiveType.Triangles);

            // Drawing contour lines over the top
            // NOTE: may need to use line primitives, may need to increase z coordinate of points via vertex shader
            contourIndices = GLHelper.FindContourIndices(streamFunction, contourTolerance, contourSpacingMultiplier, dataWidth, dataHeight);
            contourShaderManager.Use();

            GL.BindVertexArray(hContourVAO);

            GLHelper.Draw(contourIndices, PrimitiveType.Points);

            ErrorCode errorCode = GL.GetError();
            if (errorCode != ErrorCode.NoError)
            {
                Trace.WriteLine("\x1B[31m" + errorCode.ToString() + "\033[0m");
            }
        }
    }
}
