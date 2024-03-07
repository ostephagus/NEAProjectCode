using System;
using System.Collections.Generic;
using System.Linq;
using FileMakerHardCoded;

namespace FileMakerCLI
{
    public class RPNConstraint : Constraint
    {
        protected readonly string[] postFixExpression;

        public float RPNProcess(float x, float y)
        {
            int tokenPos = 0;
            ResizableStack<float> evaluationStack = new ResizableStack<float>();
            while (tokenPos < postFixExpression.Length)
            {
                string token = postFixExpression[tokenPos];
                if (ConstraintParser.OperatorsAndFunctions.Contains(token))
                {
                    float operand1, operand2;
                    if (ConstraintParser.Operators.Contains(token[0]))
                    {
                        operand2 = evaluationStack.Pop();
                        operand1 = evaluationStack.Pop();
                    }
                    else // Function rather than operand
                    {
                        operand1 = evaluationStack.Pop();
                        operand2 = 0; // Not used.
                    }
                    evaluationStack.Push(token switch
                    {
                        "+" => operand1 + operand2,
                        "-" => operand1 - operand2,
                        "*" => operand1 * operand2,
                        "/" => operand1 / operand2,
                        "^" => (float)Math.Pow(operand1, operand2),
                        "sin" => (float)Math.Sin(operand1),
                        "cos" => (float)Math.Cos(operand1),
                        "tan" => (float)Math.Tan(operand1),
                        _ => 0
                    });
                }
                else
                {
                    switch (token)
                    {
                        case "x":
                            evaluationStack.Push(x);
                            break;
                        case "y":
                            evaluationStack.Push(y);
                            break;
                        default:
                            evaluationStack.Push(float.Parse(token));
                            break;
                    }
                }
                
                tokenPos++;
            }
            return evaluationStack.Pop();
        }

        public override bool Evaluate(float x, float y)
        {
            return InequalityCompare(RPNProcess(x, y));
        }

        public RPNConstraint(string[] postFixExpression, Inequality inequality, float constant) : base((float _, float _) => 0, inequality, constant)
        {
            this.postFixExpression = postFixExpression;
        }
    }
}
