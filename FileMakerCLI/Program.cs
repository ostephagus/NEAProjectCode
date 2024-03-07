﻿using FileMakerHardCoded;

namespace FileMakerCLI
{
    internal class Program
    {
        static string[] GetConstraints()
        {
            List<string> constraints = new List<string>();
            string? userInput = "";
            Console.WriteLine("Type a constraint in the format f(x) + g(y) ? k, where f and g are functions, k is a constant, and ? is an inequality symbol.\nFor example, you could enter 2x^2+3y>=5.\nTo stop entering constraints, type q and press enter.");
            userInput = Console.ReadLine();
            while (userInput != "q" && userInput is not null)
            {
                constraints.Add(userInput);
                Console.WriteLine("Type a constraint in the format f(x) + g(y) ? k, where f and g are functions, k is a constant, and ? is an inequality symbol.\nTo stop entering constraints, type q and press enter.");
                userInput = Console.ReadLine();
            }
            return constraints.ToArray();
        }
        static void RunProgram(string[] args)
        {
            Console.Write("Enter number of cells in x direction: ");
            int xLength = int.Parse(Console.ReadLine());
            Console.Write("Enter number of cells in y direction: ");
            int yLength = int.Parse(Console.ReadLine());
            Console.Write("Enter file path to output to: ");
            string filePath = Console.ReadLine();
            string[] stringConstraints = GetConstraints();
            Constraint[] constraints = new Constraint[stringConstraints.Length];
            for (int i = 0; i < stringConstraints.Length; i++)
            {
                RPNConstraint? constraint = ConstraintParser.Parse(stringConstraints[i]);
                if (constraint is null)
                {
                    Console.WriteLine("One of your inequalities was incorrectly formatted.");
                }
                else
                {
                    constraints[i] = constraint;
                }
            }
            FileMaker.CreateFile(filePath, xLength, yLength, constraints);
        }

        static void Main(string[] args)
        {
            RunProgram(args);
            //string input = "8tan(3x)+2y^(x-4)";
            //Console.WriteLine("Input: " + input);
            //string[] postfixOutput = ConstraintParser.ConvertToRPN(input);
            //foreach (string s in postfixOutput)
            //{
            //    Console.Write(s + " ");
            //}
            //Console.WriteLine();
            //RPNConstraint constraint = new RPNConstraint(postfixOutput, Inequality.LessThan, 1);
            //Console.WriteLine(constraint.RPNProcess(-4, 60));
        }
    }
}
