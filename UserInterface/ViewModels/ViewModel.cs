using System.ComponentModel;
using UserInterface.HelperClasses;

namespace UserInterface.ViewModels
{
    public abstract class ViewModel : INotifyPropertyChanged // The equivalent of SwappableScreen for Views, provides handling of ParameterHolder, as well as some VM-specific features
    {
        public event PropertyChangedEventHandler? PropertyChanged;

        protected void OnPropertyChanged(object? sender, string propertyName)
        {
            PropertyChanged?.Invoke(sender, new PropertyChangedEventArgs(propertyName));
        }

        protected ParameterHolder? parameterHolder;

        protected ParameterStruct<T> ModifyParameterValue<T>(ParameterStruct<T> parameterStruct, T newValue)
        {
            parameterStruct.Value = newValue;
            return parameterStruct;
        }

        public ParameterHolder? ParameterHolder { get => parameterHolder; set => parameterHolder = value; }

        public ViewModel()
        {
            ParameterHolder = null;
        }

        public ViewModel(ParameterHolder parameterHolder)
        {
            ParameterHolder = parameterHolder;
        }
    }
}
