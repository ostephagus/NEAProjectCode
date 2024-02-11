using System.ComponentModel;
using UserInterface.Converters;

namespace UserInterface.HelperClasses
{
    public class UnitChangedEventArgs : PropertyChangedEventArgs
    {
        public UnitClasses.Unit NewValue { get; private set; }

        public UnitChangedEventArgs(string propertyName, UnitClasses.Unit newValue) : base(propertyName)
        {
            NewValue = newValue;
        }
    }
}
