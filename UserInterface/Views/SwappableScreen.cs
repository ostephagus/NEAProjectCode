using System;
using System.Windows.Controls;
using UserInterface.HelperClasses;

namespace UserInterface.Views
{
    public abstract class SwappableScreen : UserControl
    {
        protected ParameterHolder? parameterHolder;

        public SwappableScreen(ParameterHolder parameterHolder)
        {
            this.parameterHolder = parameterHolder;
        }
    }
}
