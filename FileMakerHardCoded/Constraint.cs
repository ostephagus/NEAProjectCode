namespace FileMakerBackend
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
        protected readonly Func<float, float, float> function;
        protected readonly Inequality inequality;
        protected readonly float constant;

        protected bool InequalityCompare(float left)
        {
            return inequality switch
            {
                Inequality.LessThan => left < constant,
                Inequality.LessThanOrEqual => left <= constant,
                Inequality.GreaterThan => left > constant,
                Inequality.GreaterThanOrEqual => left >= constant,
                _ => false
            };
        }

        public virtual bool Evaluate(float x, float y)
        {
            float leftHandSide = function(x, y);
            return InequalityCompare(leftHandSide);
        }

        public Constraint(Func<float, float, float> function, Inequality inequality, float constant)
        {
            this.function = function;
            this.inequality = inequality;
            this.constant = constant;
        }
    }
}
