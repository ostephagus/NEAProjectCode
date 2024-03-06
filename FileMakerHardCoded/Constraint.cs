namespace FileMakerHardCoded
{
    public enum Inequality
    {
        LessThan,
        LessThanOrEqual,
        GreaterThan,
        GreaterThanOrEqual
    }

    /// <summary>
    /// Represents a constraint on x and y in the form f(x) + g(y) ? k, where f and g are functions, k is a constant, and ? is an inequality symbol.
    /// </summary>
    public class Constraint
    {
        private readonly Func<float, float, float> function;
        private readonly Inequality inequality;
        private readonly float constant;

        public bool Evaluate(float x, float y)
        {
            float leftHandSide = function(x, y);
            return inequality switch
            {
                Inequality.LessThan => leftHandSide < constant,
                Inequality.LessThanOrEqual => leftHandSide <= constant,
                Inequality.GreaterThan => leftHandSide > constant,
                Inequality.GreaterThanOrEqual => leftHandSide >= constant,
                _ => false
            };
        }

        public Constraint(Func<float, float, float> function, Inequality inequality, float constant)
        {
            this.function = function;
            this.inequality = inequality;
            this.constant = constant;
        }
    }
}
