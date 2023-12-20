using System;
using System.Windows.Controls;

namespace UserInterface
{
    public abstract class SwappableScreen : UserControl
    {
        protected ParameterHolder? parameterHolder;

        public ParameterHolder? ParameterHolder { get => parameterHolder; set => parameterHolder = value; }

        public SwappableScreen()
        {
            ParameterHolder = null;
        }

        public SwappableScreen(ParameterHolder parameterHolder)
        {
            ParameterHolder = parameterHolder;
        }
    }
}
