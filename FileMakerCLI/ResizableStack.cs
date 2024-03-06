namespace FileMakerCLI
{
    public class ResizableStack<T>
    {
        T[] array;

        protected int length;
        protected int pointerPosition = -1;

        public ResizableStack() //The default constructor for the method - useful for when the number of items to be pushed on the stack is not known (sets the initial size of the array to 1 item)
        {
            array = new T[1];
            length = 1;
        }

        public ResizableStack(int length) //A constructor in case the minimum size the stack needs to be is known - therefore it would be more efficient (both time and space) to declare this at the start.
        {
            array = new T[length];
            this.length = length;
        }

        public void Push(T value) //Add a new item to the stack
        {
            if (IsFull)
            {
                length *= 2;
                Array.Resize(ref array, length); //If the array is full, resize the stack
            }
            pointerPosition = pointerPosition + 1;
            array[pointerPosition] = value;

        }

        public T Peek() //Peek the item from the top of the stack
        {
            if (IsEmpty)
            {
                throw new InvalidOperationException("Stack was empty when Peek was called");
            }
            return array[pointerPosition];
        }

        public T Pop() //Take an item off the stack, returning it.
        {
            if (IsEmpty)
            {
                throw new InvalidOperationException("Stack was empty when Pop was called");
            }
            pointerPosition--;
            if (pointerPosition < length / 4 && length > 1)
            {
                length /= 2;
                Array.Resize(ref array, length); //If less than 1/4 of the array is in use (calcalated by pointer being less than 1/4 of the length of the array)
            }
            return array[pointerPosition + 1];
        }

        public bool IsFull
        {
            get { return pointerPosition + 1 == length; } //Full if pointer is 1 less than length
        }

        public bool IsEmpty
        {
            get { return pointerPosition < 0; } //Empty if pointer is at -1 (below 1st element)
        }
    }
}
