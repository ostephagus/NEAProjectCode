using OpenTK.Graphics.OpenGL4;
using System.Diagnostics;

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

        /// <summary>
        /// Creates an array of <c>uint</c>s, representing the indices of where contour vertices should be, with each level set separated by <paramref name="primitiveRestartSentinel"/>.
        /// </summary>
        /// <param name="streamFunction">The values of the stream function for the simulation domain.</param>
        /// <param name="contourTolerance">The tolerance for accepting a vertex into the level set.</param>
        /// <param name="spacingMultiplier">A multiplier, such that vertices that have a stream function value that is an integer multiple of this multiplier will be included into the level set</param>
        /// <param name="primitiveRestartSentinel">The sentinel value, such as <c>uint.MaxValue</c></param>
        /// <param name="width">The width of the simulation space</param>
        /// <param name="height">The height of the simulation space</param>
        /// <returns>An array of <c>uint</c>s, to be passed to the EBO</returns>
        public static uint[] FindContourIndices(float[] streamFunction, float contourTolerance, float spacingMultiplier, uint primitiveRestartSentinel, int width, int height)
        {
            List<List<uint>> levelSets = new();
            for (int j = 0; j < height; j++) // Find level sets 
            {
                float streamFunctionValue = streamFunction[j];
                if (streamFunctionValue == 0)
                {
                    continue;
                }
                float distanceFromMultiple = streamFunctionValue % spacingMultiplier;
                int levelSet;
                if (distanceFromMultiple < contourTolerance || spacingMultiplier - distanceFromMultiple < contourTolerance)
                {
                    levelSet = (int)Math.Round(streamFunctionValue / spacingMultiplier); // Round the value to get the correct level set
                }
                else
                {
                    continue;
                }

                while (levelSet >= levelSets.Count) // Add level set lists until there is one for the current level set
                {
                    levelSets.Add(new List<uint>());
                }

                levelSets[levelSet].Add((uint)j); // Add the current index
            }

            List<uint> indices = new();

            for (int levelSetNum = 1; levelSetNum < levelSets.Count; levelSetNum++) // Go through each level set, finding coordinates that belong to the level set. Start at 1 because the 0 level set is not drawn.
            {
                if (levelSets[levelSetNum].Count == 0) continue; // The level set does not exist
                int currentHeight = (int)levelSets[levelSetNum][0]; // Get the starting height of the level set
                float targetValue = levelSetNum * spacingMultiplier;
                for (int i = 1; i < width; i++)
                {
                    if (!(streamFunction[i * width + currentHeight] - targetValue > contourTolerance) && !(targetValue - streamFunction[i * width + currentHeight] > contourTolerance)) // Add in another condition to avoid floating point error (which should be always less than contour tolerance)
                    {
                        levelSets[levelSetNum].Add((uint)(i * width + currentHeight));
                        continue;
                    }
                    if (streamFunction[i * width + currentHeight] > targetValue) // Possibilities: current value is too big, need to move down; or current value is too small, need to move up. For both cases, either there exists a member of the level set or there does not.
                    { // Stream function greater than target, need to move downwards
                        while (currentHeight > 0 && streamFunction[i * width + currentHeight] - targetValue > contourTolerance) // While we are still too big, decrease height until 0
                        {
                            currentHeight--;
                        }
                        // Now, current height is either larger than target but within tolerance, below target but within tolerance, or neither
                        if (streamFunction[i * width + currentHeight] > targetValue || targetValue - streamFunction[i * width + currentHeight] < contourTolerance) // Within tolerance either side of target
                        {
                            levelSets[levelSetNum].Add((uint)(i * width + currentHeight));
                        }
                        // If it is not within the tolerance, there does not exist a stream function value at this x coordinate in the level set.
                    }
                    else // Current height's contour value is too small
                    {
                        while (currentHeight < height - 1 && streamFunction[i * width + currentHeight] < targetValue) // While we are still too small, increase height until limit
                        {
                            currentHeight++;
                        }
                        // Now, current height is either smaller than target but within tolerance, above target but within tolerance, or neither
                        if (targetValue > streamFunction[i * width + currentHeight] || streamFunction[i * width + currentHeight] - targetValue < contourTolerance)
                        {
                            levelSets[levelSetNum].Add((uint)(i * width + currentHeight));
                        }
                    }
                }
                indices.AddRange(levelSets[levelSetNum]);
                indices.Add(primitiveRestartSentinel);
            }
            return indices.ToArray();
        }

        /// <summary>
        /// Creates an element buffer object, and buffers the indices array.
        /// </summary>
        /// <param name="indices">An array representing the indices of the primitives that are to be drawn.</param>
        /// <returns>A handle to the created EBO.</returns>
        public static int CreateEBO(uint[] indices)
        {
            int EBOHandle = GL.GenBuffer();
            GL.BindBuffer(BufferTarget.ElementArrayBuffer, EBOHandle);
            GL.BufferData(BufferTarget.ElementArrayBuffer, indices.Length * sizeof(uint), indices, BufferUsageHint.StaticDraw);
            return EBOHandle;
        }

        /// <summary>
        /// Creates an element buffer object, and buffers the indices array.
        /// </summary>
        /// <param name="indices">An array representing the indices of the primitives that are to be drawn.</param>
        /// <param name="bufferUsageHint">The enum value to tell the GPU which type of memory it should use.</param>
        /// <returns>A handle to the created EBO.</returns>
        public static int CreateEBO(uint[] indices, BufferUsageHint bufferUsageHint)
        {
            int EBOHandle = GL.GenBuffer();
            GL.BindBuffer(BufferTarget.ElementArrayBuffer, EBOHandle);
            GL.BufferData(BufferTarget.ElementArrayBuffer, indices.Length * sizeof(uint), indices, bufferUsageHint);
            return EBOHandle;
        }

        /// <summary>
        /// Buffers new data into the currently bound EBO.
        /// </summary>
        /// <param name="indices">An array representing the indices of the primitives that are to be drawn.</param>
        /// <param name="bufferUsageHint">The enum value to tell the GPU which type of memory it should use.</param>
        public static void UpdateEBO(uint[] indices, BufferUsageHint bufferUsageHint)
        {
            GL.BufferData(BufferTarget.ElementArrayBuffer, indices.Length * sizeof(uint), indices, bufferUsageHint);
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
        /// <param name="indices">An array of unsigned integers specifying the order in which to link vertices together.</param>
        /// <param name="primitiveType">Which type of primitive type to use for drawing.</param>
        public static void Draw(uint[] indices, PrimitiveType primitiveType)
        {
            GL.DrawElements(primitiveType, indices.Length, DrawElementsType.UnsignedInt, 0);
        }
    }
}