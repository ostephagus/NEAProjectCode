using System;

namespace UserInterface.HelperClasses
{
    public class CircularQueue<T> : Queue<T>
    {
        protected int count;

        public CircularQueue(int length) : base(length)
        {
            count = 0;
        }

        public override void Enqueue(T value)
        {
            if (IsFull)
            {
                throw new InvalidOperationException("Queue was full when Enqueue was called");
            }

            array[back] = value;
            back++;
            if (back >= length)
            {
                back = 0;
            }
            count++;
        }

        public override T Dequeue()
        {
            if (IsEmpty)
            {
                throw new InvalidOperationException("Queue was empty when Dequeue was called");
            }

            T removedItem = array[front];
            front++;
            if (front >= length)
            {
                front = 0;
            }
            count--;

            return removedItem;
        }

        public override bool IsEmpty
        {
            get { return count == 0; }
        }

        public override bool IsFull
        {
            get { return count == length; }
        }

        public override int Count => count;
    }
}
