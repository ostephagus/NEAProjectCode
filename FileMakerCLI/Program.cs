using FileMakerBackend;

namespace FileMakerCLI
{
    internal class Program
    {
        static T ConstrainedInput<T>(string prompt, Predicate<string> requirements, Func<string, T> conversionFunction)
        {
            Console.WriteLine(prompt);
            string? userInput = Console.ReadLine();
            while (userInput is null || !requirements(userInput))
            {
                Console.WriteLine("Input was not valid.");
                Console.WriteLine(prompt);
                userInput = Console.ReadLine();
            }
            return conversionFunction(userInput);
        }

        static int IntInput(string prompt, bool requirePositive = false)
        {
            Predicate<string> requirements;
            if (requirePositive)
            {
                requirements = input => int.TryParse(input, out int value) && value > 0;
            }
            else
            {
                requirements = input => int.TryParse(input, out _);                 
            }
            return ConstrainedInput(prompt, requirements, int.Parse);
        }

        static string NonNullInput(string prompt)
        {
            return ConstrainedInput(prompt, input => input is not null, input => input);
        }

        static Constraint[] GetConstraints()
        {
            List<Constraint> constraints = new List<Constraint>();
            Console.WriteLine("Type a constraint in the format f(x) + g(y) ? k, where f and g are functions, k is a constant, and ? is an inequality symbol.\nFor example, you could enter 2x^2+3y>=5.\nTo stop entering constraints, type q and press enter.");
            string? userInput = Console.ReadLine();
            while (userInput != "q" && userInput is not null)
            {
                try
                {
                    constraints.Add(ConstraintParser.Parse(userInput));
                } catch (FormatException)
                {
                    Console.WriteLine("Constraint was incorrectly formatted and has not been added.");
                }
                Console.WriteLine("Type a constraint in the format f(x) + g(y) ? k, where f and g are functions, k is a constant, and ? is an inequality symbol.\nTo stop entering constraints, type q and press enter.");
                userInput = Console.ReadLine();
            }
            return constraints.ToArray();
        }

        static void RunProgram(string[] args)
        {
            int xLength = IntInput("Enter number of cells in x direction: ", true);
            int yLength = IntInput("Enter number of cells in y direction: ", true);
            string filePath = NonNullInput("Enter file path to output to.");
            Constraint[] constraints = GetConstraints();

            FileMaker.CreateFile(filePath, xLength, yLength, constraints);
        }

        static void Main(string[] args)
        {
            RunProgram(args);
        }
    }
}
