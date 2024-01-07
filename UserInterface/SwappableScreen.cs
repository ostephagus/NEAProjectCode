using System;
using System.Windows.Controls;
using UserInterface.HelperClasses;

namespace UserInterface
{
    public abstract class SwappableScreen : UserControl
    {
        protected ParameterHolder? parameterHolder;

        protected ParameterStruct<T> ModifyParameterValue<T>(ParameterStruct<T> parameterStruct, T newValue)
        {
            parameterStruct.Value = newValue;
            return parameterStruct;
        }

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
