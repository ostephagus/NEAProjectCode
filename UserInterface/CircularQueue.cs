using System;

namespace UserInterface
{
    public class CircularQueue<T> : Queue<T>
    {
        //protected bool isFull;
        //protected bool isEmpty;
        protected int count;

        public CircularQueue(int length) : base(length)
        {
            //isEmpty = true;
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
            //isEmpty = false; //Set isEmpty to false, and IsEmpty to true if the front is equal to the back
            //isFull = (front == back);
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
            //isFull = false; //Set isFull to false, and isEmpty to true if the front is equal to the back
            //isEmpty = (front == back);
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
