using FileMakerBackend;

namespace FileMakerCLI
{
    public static class ConstraintParser
    {
        public static readonly char[] Operators = ['^', '/', '*', '+', '-']; // In order of precendence
        private static readonly Dictionary<char, (int, bool)> operatorData = new Dictionary<char, (int, bool)>() // lowest int means highest precendence, true means right associativity, false means left associativity
        {
            {'^', (1, true) },
            {'/', (2, false) },
            {'*', (2, false) },
            {'-', (3, false) },
            {'+', (3, false) }
        };
        public static readonly string[] MathsFunctions = ["cos", "sin", "tan"];
        public static readonly string[] OperatorsAndFunctions = Operators.Select(x => x.ToString()).Concat(MathsFunctions).ToArray();
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
                    if (char.IsDigit(lastChar) || lastChar == '.') // If the last char was a digit, use same token.
                    {
                        tokens[^1] += currentChar.ToString();
                    }
                    else // If not, start a new token.
                    {
                        tokens.Add(currentChar.ToString());
                    }
                }
                else if (currentChar == '.')
                {
                    if (char.IsDigit(lastChar)) // Most common use case: decimal point after a number
                    {
                        tokens[^1] += currentChar.ToString();
                    }
                    else // Less common use case: decimal point implying a zero
                    {
                        tokens.Add("0.");
                    }
                }
                else if (currentChar == 'x' || currentChar == 'y') // variables
                {
                    if (!Operators.Contains(lastChar) && lastChar != '(') // last char was not an operator or open bracket
                    {
                        tokens.Add("*"); // Add a multiplication sign between the 2 tokens if it was not an operator before.
                    }
                    tokens.Add(currentChar.ToString());
                }
                else if (Operators.Contains(currentChar) || currentChar == '(' || currentChar == ')') // Operators
                {
                    tokens.Add(currentChar.ToString());
                }
                else if (input.Length - stringPos > functionLength && MathsFunctions.Contains(input[stringPos..(stringPos + functionLength)])) // Check there is enough characters left and then see if the next characters are a function
                {
                    if (!Operators.Contains(lastChar) && lastChar != '(') // last char was not an operator or open bracket
                    {
                        tokens.Add("*"); // Add a multiplication sign between the 2 tokens if it was not an operator before.
                    }
                    tokens.Add(input[stringPos..(stringPos + functionLength)]);
                    stringPos += functionLength - 1;
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
                if (float.TryParse(currentToken, out _)) // Token is a number
                {
                    output.Add(currentToken);
                }
                else if (currentToken.Length == functionLength) // Current token is a function
                {
                    operatorStack.Push(currentToken);
                }
                else
                {
                    char symbol = currentToken[0]; // All other tokens have only 1 character.
                    if (symbol == 'x' || symbol == 'y')
                    {
                        output.Add(currentToken);
                    }
                    else if (Operators.Contains(symbol))
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
                                if (operatorStack.IsEmpty) break;
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
                        if (!operatorStack.IsEmpty && operatorStack.Peek().Length == functionLength) // Function at the top of the stack
                        {
                            output.Add(operatorStack.Pop());
                        }
                    }
                }
                tokenPos++;
            }
            if (!operatorStack.IsEmpty)
            {
                do // Pop the rest of the stack onto the output list
                {
                    output.Add(operatorStack.Pop());
                } while (!operatorStack.IsEmpty);
            }
            return output.ToArray();
        }

        public static RPNConstraint? Parse(string stringConstraint)
        {
            string leftHandSide;
            Inequality inequality;
            float constant;
            if (stringConstraint.Contains("<="))
            {
                string[] equationSides = stringConstraint.Split("<=");
                leftHandSide = equationSides[0];
                inequality = Inequality.LessThanOrEqual;
                constant = float.Parse(equationSides[1]);
            }
            else if (stringConstraint.Contains('<'))
            {
                string[] equationSides = stringConstraint.Split('<');
                leftHandSide = equationSides[0];
                inequality = Inequality.LessThan;
                constant = float.Parse(equationSides[1]);
            }
            else if (stringConstraint.Contains(">="))
            {
                string[] equationSides = stringConstraint.Split(">=");
                leftHandSide = equationSides[0];
                inequality = Inequality.GreaterThanOrEqual;
                constant = float.Parse(equationSides[1]);
            }
            else if (stringConstraint.Contains('>'))
            {
                string[] equationSides = stringConstraint.Split('>');
                leftHandSide = equationSides[0];
                inequality = Inequality.GreaterThan;
                constant = float.Parse(equationSides[1]);
            }
            else
            {
                return null;
            }
            string[] postFix = ConvertToRPN(leftHandSide);
            return new RPNConstraint(postFix, inequality, constant);
        }
    }
}
