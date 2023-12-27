using System;
using System.Collections.Generic;

namespace UserInterface
{
    /// <summary>
    /// A memory-efficient linear queue implementation.
    /// </summary>
    /// <typeparam name="T">The type of the objects that will be stored in the queue.</typeparam>
    public class ResizableLinearQueue<T>
    {
        private T[] array;
        private int back;
        private int front;
        private int length;

        /// <summary>
        /// Initialises the linear queue with a known start length.
        /// </summary>
        /// <param name="startLength">The initial length of the queue.</param>
        public ResizableLinearQueue(int startLength)
        {
            length = startLength;
            array = new T[length];
            back = 0;
            front = 0;
        }

        /// <summary>
        /// Initialises the linear queue with a default initial length of 1.
        /// </summary>
        public ResizableLinearQueue() : this(1) { } //If no length is specified, start at 1.

        /// <summary>
        /// Moves all the elements forwards in the array such the front of the queue is at position 0 and the rest are stored contiguously.
        /// </summary>
        private void Reshuffle()
        {
            for (int i = front; i < length; i++)
            {
                array[i - front] = array[i]; //Reshuffle the array by copying each item back a certain number of spaces
            }
            back -= front; //Reshuffle the pointers
            front = 0;
        }

        /// <summary>
        /// Adds an item to the back of the queue.
        /// </summary>
        /// <param name="value">The item to add.</param>
        public void Enqueue(T value)
        {
            if (back == length)
            {
                if (front > 0) //If the queue is full and the front is not at 0, there is unused space at the start of the array
                {
                    Reshuffle();
                }
                else //If the queue is completely full (front pointer at 0), make the queue 2x longer (don't reshuffle because this will have no effect)
                {
                    length *= 2;
                    Array.Resize(ref array, length);
                }
            }
            array[back] = value; //Put an item at the back and increase the back pointer
            back++;
        }

        /// <summary>
        /// Adds multiple items to the back of the queue.
        /// </summary>
        /// <param name="values">The items to add to the queue.</param>
        public void EnqueueRange(IEnumerable<T> values)
        {
            foreach (T value in values)
            {
                Enqueue(value);
            }
        }

        /// <summary>
        /// Removes an item from the front of the queue and returns it
        /// </summary>
        /// <returns>The item that used to be at the front of the queue.</returns>
        /// <exception cref="InvalidOperationException">If the queue was empty before Dequeue was called</exception>
        public T Dequeue()
        {
            if (IsEmpty)
            {
                throw new InvalidOperationException("Queue was empty when Dequeue was called");
            }
            T removedItem = array[front];
            front++;

            if (back - front < length / 4) //If only 1/4 of the array now is used, reshuffle it and halve the size
            {
                Reshuffle();
                length /= 2;
                Array.Resize(ref array, length);
            }
            return removedItem; //Return the item that has been "removed"
        }

        /// <summary>
        /// Removes an item from the front of the queue, if the queue is not empty.
        /// </summary>
        /// <param name="removedItem">The item removed if the operation succeeded, or the default value for <typeparamref name="T" /> if it failed.</param>
        /// <returns>A <c>bool</c> indicating whether the item was successfully removed.</returns>
        public bool TryDequeue(out T? removedItem)
        {
            if (IsEmpty)
            {
                removedItem = default;
                return false;
            }
            removedItem = Dequeue();
            return true;
        }

        public bool IsEmpty
        {
            get { return front == back; } //Empty if the front pointer is at the back pointer
        }
    }
}
