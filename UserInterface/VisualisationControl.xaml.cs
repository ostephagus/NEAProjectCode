﻿using OpenTK.Graphics.OpenGL4;
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

            width = 10;
            height = 10;
            fieldValues = new float[width * height];
            for (int i = 0; i < width; i++)
            {
                for (int j = 0; j < height; j++)
                {
                    fieldValues[i * width + j] = (float)i / width;
                }
            }

            GLWpfControlSettings settings = new GLWpfControlSettings { MajorVersion = 3, MinorVersion = 1 };
            GLControl.Start(settings);

            shaderManager = new("shader.vert", "shader.frag");
            SetUpGL();
        }

        private void SetUpGL()
        {
            GL.ClearColor(0.1f, 0.7f, 0.5f, 1.0f);
            vertices = GLHelper.FillVertices(width, height);
            hVertexBuffer = GLHelper.CreateVBO();
            hVertexArrayObject = GLHelper.CreateVAO();

            GLHelper.BufferSubData(vertices, 0);
            Trace.WriteLine(GL.GetError().ToString());
            GLHelper.BufferSubData(fieldValues, vertices.Length);
            Trace.WriteLine(GL.GetError().ToString());

            GLHelper.CreateAttribPointer(0, 2, 2, 0);
            GLHelper.CreateAttribPointer(1, 1, 1, vertices.Length);

            indices = GLHelper.FillIndices(width, height);
            hElementBuffer = GLHelper.CreateEBO(indices);

            shaderManager.Use();
        }

        public void GLControl_OnRender(TimeSpan delta)
        {
            GL.Clear(ClearBufferMask.ColorBufferBit);

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