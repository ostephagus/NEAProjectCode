#define HOLLOW_TRIANGLES

using OpenTK.Graphics.OpenGL4;
using OpenTK.Wpf;
using System;
using System.ComponentModel;
using System.Diagnostics;
using System.Windows.Controls;
using UserInterface.HelperClasses;
using Visualisation;

namespace UserInterface
{
    /// <summary>
    /// Interaction logic for VisualisationControl.xaml
    /// </summary>
    public partial class VisualisationControl : UserControl, INotifyPropertyChanged
    {
        private readonly ShaderManager fieldShaderManager;
        private readonly ShaderManager contourShaderManager;
        //private ComputeShaderManager computeShaderManager;

        private readonly float[] vertices;
        private readonly uint[] fieldIndices;
        private uint[] contourIndices;

        private const uint primitiveRestartIndex = uint.MaxValue;

        private int hVBO;
        private int hFieldVAO;
        //private int hFieldEBO;

        private int hContourVAO;
        //private int hContourEBO;

        private int hSubtrahend;
        private int hScalar;

        private ParameterHolder parameterHolder;
        private float[] streamFunction;

        private int dataWidth;
        private int dataHeight;

        private float frameTime;

        public int DataWidth { get => dataWidth; set => dataWidth = value; }
        public int DataHeight { get => dataHeight; set => dataHeight = value; }
        public float[] StreamFunction { get => streamFunction; set => streamFunction = value; }
        public float FrameTime { get => frameTime; }

        public event PropertyChangedEventHandler? PropertyChanged;

        public VisualisationControl(ParameterHolder parameterHolder, float[] streamFunction, int dataWidth, int dataHeight)
        {
            this.parameterHolder = parameterHolder;
            this.streamFunction = streamFunction;
            this.dataWidth = dataWidth;
            this.dataHeight = dataHeight;

            InitializeComponent();
            DataContext = this;

            SetUpGL(out fieldShaderManager, out contourShaderManager, out vertices, out fieldIndices, out contourIndices); // Using out parameters so that these can be returned to the control of the constructor and not generate warnings
        }

        ~VisualisationControl()
        {
            fieldShaderManager.Dispose();
            contourShaderManager.Dispose();
        }


        private void SetUpGL(out ShaderManager fieldShaderManager, out ShaderManager contourShaderManager, out float[] vertices, out uint[] fieldIndices, out uint[] contourIndices)
        {
            GLWpfControlSettings settings = new() { MajorVersion = 3, MinorVersion = 1 };
            GLControl.Start(settings);

            fieldShaderManager = new ShaderManager(new (string, ShaderType)[] { ("fieldShader.frag", ShaderType.FragmentShader), ("fieldShader.vert", ShaderType.VertexShader) });
            contourShaderManager = new ShaderManager(new (string, ShaderType)[] { ("contourShader.frag", ShaderType.FragmentShader), ("contourShader.vert", ShaderType.VertexShader) });
            //computeShaderManager = new ComputeShaderManager("shader.comp");

            GL.Enable(EnableCap.PrimitiveRestart);
            GL.PrimitiveRestartIndex(primitiveRestartIndex);

            HandleData(out vertices, out fieldIndices, out contourIndices);
        }

        private void HandleData(out float[] vertices, out uint[] fieldIndices, out uint[] contourIndices)
        {
            //GL.ClearColor(0.1f, 0.7f, 0.5f, 1.0f);

            vertices = GLHelper.FillVertices(dataWidth, dataHeight);
            fieldIndices = GLHelper.FillIndices(dataWidth, dataHeight);
            contourIndices = GLHelper.FindContourIndices(streamFunction, parameterHolder.ContourTolerance.Value, parameterHolder.ContourSpacing.Value, primitiveRestartIndex, dataWidth, dataHeight);

            FieldParameters fieldParameters = parameterHolder.FieldParameters.Value;

            // Setting up data for field visualisation
            hFieldVAO = GLHelper.CreateVAO();
            hVBO = GLHelper.CreateVBO(vertices.Length + fieldParameters.field.Length);

            GLHelper.BufferSubData(vertices, 0);
            //Trace.WriteLine(GL.GetError().ToString());
            GLHelper.BufferSubData(fieldParameters.field, vertices.Length);
            //Trace.WriteLine(GL.GetError().ToString());

            GLHelper.CreateAttribPointer(0, 2, 2, 0); // Vertex pointer
            GLHelper.CreateAttribPointer(1, 1, 1, vertices.Length); // Field value pointer

            _ = GLHelper.CreateEBO(fieldIndices); // EBO handle is never used because it is bound to the VAO

            // Setting up data for contour line plotting
            hContourVAO = GLHelper.CreateVAO();
            GL.BindBuffer(BufferTarget.ArrayBuffer, hVBO); // Bind the same VBO

            GLHelper.CreateAttribPointer(0, 2, 2, 0); // And the same for attribute pointers
            GLHelper.CreateAttribPointer(1, 1, 1, vertices.Length);

            _ = GLHelper.CreateEBO(contourIndices);

            // Return to field context
            GL.BindVertexArray(hFieldVAO);

            hSubtrahend = fieldShaderManager.GetUniformLocation("subtrahend");
            hScalar = fieldShaderManager.GetUniformLocation("scalar");
        }

        public void GLControl_OnRender(TimeSpan delta)
        {
            GL.Clear(ClearBufferMask.ColorBufferBit);
            FieldParameters fieldParameters = parameterHolder.FieldParameters.Value; // Get the most recent field parameters

            // For each draw command, need to bind the program, set uniforms, bind VAO, draw

            // Drawing field value spectrum
            fieldShaderManager.Use();

            fieldShaderManager.SetUniform(hSubtrahend, fieldParameters.min);
            fieldShaderManager.SetUniform(hScalar, 1 / (fieldParameters.max - fieldParameters.min));

            GL.BindVertexArray(hFieldVAO);
            GLHelper.BufferSubData(fieldParameters.field, vertices.Length); // Update the field values

#if HOLLOW_TRIANGLES
            GLHelper.Draw(fieldIndices, PrimitiveType.LineStrip); // For alignment testing - don't fill in triangles.
#else
            GLHelper.Draw(fieldIndices, PrimitiveType.Triangles);
#endif

            // Drawing contour lines over the top
            if (parameterHolder.DrawContours.Value)
            {
                contourIndices = GLHelper.FindContourIndices(streamFunction, parameterHolder.ContourTolerance.Value, parameterHolder.ContourSpacing.Value, primitiveRestartIndex, dataWidth, dataHeight);
                contourShaderManager.Use();

                GL.BindVertexArray(hContourVAO);

                GLHelper.UpdateEBO(contourIndices, BufferUsageHint.DynamicDraw);

                GLHelper.Draw(contourIndices, PrimitiveType.LineStrip);
            }

            frameTime = (float)delta.TotalSeconds;
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(nameof(FrameTime)));

            ErrorCode errorCode = GL.GetError();
            if (errorCode != ErrorCode.NoError)
            {
                Trace.WriteLine("\x1B[31m" + errorCode.ToString() + "\033[0m");
            }
        }
    }
}
