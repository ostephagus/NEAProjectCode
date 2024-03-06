using FileMakerHardCoded;

namespace FileMakerCLI
{
    public static class ConstraintParser
    {
        private static readonly char[] operators = ['^', '/', '*', '+', '-']; // In order of precendence
        private static readonly Dictionary<char, (int, bool)> operatorData = new Dictionary<char, (int, bool)>() // lowest int means highest precendence, true means right associativity, false means left associativity
        {
            {'^', (1, true) },
            {'/', (2, false) },
            {'*', (2, false) },
            {'-', (3, false) },
            {'+', (3, false) }
        };
        private static readonly string[] mathsFunctions = ["cos", "sin", "tan"];
        private static readonly int functionLength = 3;

        private static string[] Tokenise(string input)
        {
            input = input.ToLower();
            List<string> tokens = new List<string>();
            int stringPos = 0;
            //int tokenPos = 0;
            char lastChar = input[0];
            char currentChar;
            tokens.Add(input[0].ToString());
            while (stringPos < input.Length - 1)
            {
                stringPos++;
                currentChar = input[stringPos];
                if (currentChar == ' ')
                {
                    continue; // Skip spaces
                }
                
                if (char.IsDigit(currentChar))
                {
                    if (char.IsDigit(lastChar)) // If the last char was a digit, use same token.
                    {
                        tokens[^1] += currentChar.ToString();
                    }
                    else // If not, start a new token.
                    {
                        tokens.Add(currentChar.ToString());
                    }
                }
                else if (currentChar == 'x' || currentChar == 'y') // variables
                {
                    if (char.IsDigit(lastChar) || currentChar == 'x' || currentChar == 'y') // last char was a number or another variable
                    {
                        tokens.Add("*"); // Add a multiplication sign between the number and variable
                    }
                    tokens.Add(currentChar.ToString());
                }
                else if (operators.Contains(currentChar) || currentChar == '(' || currentChar == ')') // operators
                {
                    tokens.Add(currentChar.ToString());
                }
                else if (input.Length - stringPos > functionLength && mathsFunctions.Contains(input[stringPos..(stringPos + functionLength)])) // Check there is enough characters left and then see if the next characters are a function
                {
                    tokens.Add(input[stringPos..(stringPos + functionLength)]);
                }
                else
                {
                    throw new FormatException("Input was not in the correct format.");
                }
                lastChar = currentChar;
            }
            return tokens.ToArray();
        }

        public static string[] ConvertToRPN(string infix)
        {
            string[] tokens = Tokenise(infix);
            List<string> output = new List<string>();
            ResizableStack<string> operatorStack = new ResizableStack<string>();
            int tokenPos = 0;
            while (tokenPos < tokens.Length)
            {
                string currentToken = tokens[tokenPos];
                if (currentToken.Length == functionLength) // Current token is a function
                {
                    operatorStack.Push(currentToken);
                }
                else
                {
                    char symbol = currentToken[0]; // All other tokens have only 1 character.
                    if (char.IsDigit(symbol) || symbol == 'x' || symbol == 'y')
                    {
                        output.Add(currentToken);
                    }
                    else if (operators.Contains(symbol))
                    {
                        if (operatorStack.IsEmpty)
                        {
                            operatorStack.Push(currentToken);
                        }
                        else
                        {
                            char topOfStack = operatorStack.Peek()[0];
                            while (topOfStack != '(' && (operatorData[topOfStack].Item1 < operatorData[symbol].Item1 || (operatorData[topOfStack].Item1 == operatorData[symbol].Item1 && !operatorData[symbol].Item2)))
                            {
                                output.Add(operatorStack.Pop());
                                topOfStack = operatorStack.Peek()[0];
                            }
                            operatorStack.Push(currentToken);
                        }
                    }
                    else if (symbol == '(')
                    {
                        operatorStack.Push(currentToken);
                    }
                    else if (symbol == ')')
                    {
                        string topOfStack;
                        do
                        {
                            topOfStack = operatorStack.Pop();
                            output.Add(topOfStack);
                        }
                        while (topOfStack != "(");
                        output.RemoveAt(output.Count - 1); // Remove the last element (a "(").
                    }
                }
                tokenPos++;
            }

            do // Pop the rest of the stack onto the output list
            {
                output.Add(operatorStack.Pop());
            } while (!operatorStack.IsEmpty);
            return output.ToArray();
        }

        //public static Constraint Parse(string stringConstraint)
        //{
        //    string[] postFix = ConvertToRPN(stringConstraint);
        //}
    }
}
