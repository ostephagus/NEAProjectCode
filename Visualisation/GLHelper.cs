using OpenTK.Graphics.OpenGL4;

namespace Visualisation
{
    public static class GLHelper
    {
        /// <summary>
        /// Creates an array of vertices, with values for each coordinate in the field.
        /// </summary>
        /// <param name="fieldValues">A flattened array of values for the field.</param>
        /// <param name="width">The number of vertices in the x direction.</param>
        /// <param name="height">The number of vertices in the y direction</param>
        /// <returns>An array of floats to be passed to the vertex shader.s</returns>
        public static float[] FillVertices(int width, int height)
        {
            float[] vertices = new float[2 * width * height];

            float horizontalStep = 2f / (width - 1);
            float verticalStep = 2f / (height - 1);

            for (int i = 0; i < width; i++)
            {
                for (int j = 0; j < height; j++)
                {
                    // Need to start at bottom left (-1, -1) and go vertically then horizontally to top right (1, 1)
                    vertices[i * height * 2 + j * 2 + 0] = i * horizontalStep - 1; // Starting at -1, increase x coordinate each iteration of outer loop
                    vertices[i * height * 2 + j * 2 + 1] = j * verticalStep - 1;  // Starting at -1, increase y coordinate after each iteration of inner loop
                }
            }
            return vertices;
        }

        /// <summary>
        /// Creates an index array for triangles to be drawn into a grid
        /// </summary>
        /// <param name="width">The width of the simulation space</param>
        /// <param name="height">The height of the simulation space</param>
        /// <returns>An array of unsigned integers representing the indices of each triangle, flattened</returns>
        public static uint[] FillIndices(int width, int height)
        {
            // Note that the given data has first data point bottom left, then moving upwards (in the positive y direction) then moving left (positive x direction)
            uint[] indices = new uint[(height - 1) * (width - 1) * 6];
            // For each 2x2 square of vertices, we need 2 triangles with the hypotenuses on the leading diagonal.
            for (int i = 0; i < width - 1; i++)
            {
                for (int j = 0; j < height - 1; j++)
                {
                    indices[i * (height - 1) * 6 + j * 6 + 0] = (uint)(i * height + j);           // Top left
                    indices[i * (height - 1) * 6 + j * 6 + 1] = (uint)(i * height + j + 1);       // Top right
                    indices[i * (height - 1) * 6 + j * 6 + 2] = (uint)((i + 1) * height + j + 1); // Bottom right
                    indices[i * (height - 1) * 6 + j * 6 + 3] = (uint)(i * height + j);           // Top left
                    indices[i * (height - 1) * 6 + j * 6 + 4] = (uint)((i + 1) * height + j + 1); // Bottom right
                    indices[i * (height - 1) * 6 + j * 6 + 5] = (uint)((i + 1) * height + j);     // Bottom left
                }
            }
            return indices;
        }

        public static uint[] FindContourIndices(float[] streamFunction, float contourTolerance, float spacingMultiplier, int width, int height)
        {
            List<uint> indices = new List<uint>();
            for (int i = 0; i < width; i++)
            {
                for (int j = 0; j < height; j++)
                {
                    float streamFunctionValue = streamFunction[i * width + j];
                    if (streamFunctionValue % spacingMultiplier < contourTolerance) // If the stream function value is close to an integer multiple of the spacing multipliers
                    {
                        indices.Add((uint)(i * width + j));
                    }
                }
            }
            return indices.ToArray();
        }

        /// <summary>
        /// Creates an element buffer object, and buffers the indices array
        /// </summary>
        /// <param name="indices">An array representing the indices of the triangles that are to be drawn.</param>
        /// <returns>A handle to the created EBO</returns>
        public static int CreateEBO(uint[] indices)
        {
            int EBOHandle = GL.GenBuffer();
            GL.BindBuffer(BufferTarget.ElementArrayBuffer, EBOHandle);
            GL.BufferData(BufferTarget.ElementArrayBuffer, indices.Length * sizeof(uint), indices, BufferUsageHint.StaticDraw);
            return EBOHandle;
        }

        /// <summary>
        /// Creates a vertex array object, which will hold the data to be passed to the vertex shader.
        /// </summary>
        /// <returns>A handle to the created VAO</returns>
        public static int CreateVAO()
        {
            int VAOHandle = GL.GenVertexArray();
            GL.BindVertexArray(VAOHandle);
            return VAOHandle;
        }

        /// <summary>
        /// Creates an attribute pointer, providing metadata to OpenGL when passing data to the vertex shader.
        /// </summary>
        /// <param name="pointerNumber">The number of this pointer - this is the number passed to layout in the vertex shader.</param>
        /// <param name="length">The dimension of the resulting vector</param>
        /// <param name="stride">The width (in number of floats) of the subsections of the vertex array</param>
        /// <param name="offset">The position (in number of floats) of the first element to include in the resulting vector</param>
        public static void CreateAttribPointer(int pointerNumber, int length, int stride, int offset)
        {
            GL.VertexAttribPointer(pointerNumber, length, VertexAttribPointerType.Float, false, stride * sizeof(float), offset * sizeof(float));
            GL.EnableVertexAttribArray(pointerNumber);
        }

        /// <summary>
        /// Creates a buffer and binds it.
        /// </summary>
        /// <returns>A handle to the created VBO.</returns>
        public static int CreateVBO()
        {
            int VBOHandle = GL.GenBuffer();
            GL.BindBuffer(BufferTarget.ArrayBuffer, VBOHandle);
            return VBOHandle;
        }

        /// <summary>
        /// Creates a buffer and binds it, filling it with blank data to ensure it is the correct size.
        /// </summary>
        /// <param name="size">The length, in number of floats, of the desired buffer.</param>
        /// <returns>A handle to the created VBO.</returns>
        public static int CreateVBO(int size)
        {
            int VBOHandle = GL.GenBuffer();
            GL.BindBuffer(BufferTarget.ArrayBuffer, VBOHandle);
            GL.BufferData(BufferTarget.ArrayBuffer, size * sizeof(float), new float[size], BufferUsageHint.StreamDraw);
            return VBOHandle;
        }

        /// <summary>
        /// Copies a <c>float[]</c> into part of a buffer, starting at <paramref name="offset"/>
        /// </summary>
        /// <param name="data">The <c>float[]</c> to be copied into the buffer</param>
        /// <param name="offset">The desired index of the first float to be copied</param>
        public static void BufferSubData(float[] data, int offset)
        {
            GL.BufferSubData(BufferTarget.ArrayBuffer, offset * sizeof(float), data.Length * sizeof(float), data);
        }
        
        /// <summary>
        /// Draws the grid, using triangles with indices specified in <paramref name="indices"/>.
        /// </summary>
        /// <param name="VAOHandle">The handle of the vertex array object for the shaders to use.</param>
        /// <param name="indices">An array of unsigned integers specifying the indices of the </param>
        public static void Draw(int VAOHandle, uint[] indices)
        {
            GL.BindVertexArray(VAOHandle);
            GL.DrawElements(PrimitiveType.Triangles, indices.Length, DrawElementsType.UnsignedInt, 0);
        }
    }
}