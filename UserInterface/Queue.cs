namespace UserInterface
{
    /// <summary>
    /// An abstract class to represent the methods for a queue.
    /// </summary>
    /// <typeparam name="T">The type of objects that will be stored in the queue.</typeparam>
    public abstract class Queue<T>
    {
        protected T[] array;

        protected int front;
        protected int back;
        protected int length;

        /// <summary>
        /// Initialises a queue with length <paramref name="length"/>
        /// </summary>
        /// <param name="length">The length of the queue</param>
        public Queue(int length)
        {
            array = new T[length];
            this.length = length;
        }

        /// <summary>
        /// Adds <paramref name="item"/> to the back of the queue.
        /// </summary>
        /// <param name="item">The item to add to the queue.</param>
        public abstract void Enqueue(T item);

        /// <summary>
        /// Removes one item from the front of the queue, and returns it.
        /// </summary>
        /// <returns>The item that used to be at the front of the queue.</returns>
        public abstract T Dequeue();

        /// <summary>
        /// Returns whether the queue is full.
        /// </summary>
        public abstract bool IsFull { get; }

        /// <summary>
        /// Returns whether the queue is empty.
        /// </summary>
        public abstract bool IsEmpty { get; }

        /// <summary>
        /// Returns the number of items in the queue.
        /// </summary>
        public abstract int Count { get; }
    }
}
