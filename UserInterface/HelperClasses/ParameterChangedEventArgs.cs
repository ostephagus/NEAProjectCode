using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace UserInterface.HelperClasses
{
    class ParameterChangedEventArgs : PropertyChangedEventArgs
    {
        public float NewValue { get; private set; }

        public ParameterChangedEventArgs(string propertyName, float newValue) : base(propertyName)
        {
            NewValue = newValue;
        }
    }
}
