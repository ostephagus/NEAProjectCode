using UserInterface.HelperClasses;

namespace UserInterface.ViewModels
{
    public class ConfigScreenVM : ViewModel
    {
        private float inVel;
        private float chi;
        private float width;
        private float height;


        public float InVel
        {
            get => inVel;
            set { inVel = value; OnPropertyChanged(this, nameof(InVel)); parameterHolder.InflowVelocity = ModifyParameterValue(parameterHolder.InflowVelocity, inVel); }
        }

        public float Chi
        {
            get => chi;
            set { chi = value; OnPropertyChanged(this, nameof(Chi)); parameterHolder.SurfaceFriction = ModifyParameterValue(parameterHolder.SurfaceFriction, chi); }
        }

        public float Width
        {
            get => width;
            set { width = value; OnPropertyChanged(this, nameof(Width)); parameterHolder.Width = ModifyParameterValue(parameterHolder.Width, width); }
        }

        public float Height
        {
            get => height;
            set { height = value; OnPropertyChanged(this, nameof(Height)); parameterHolder.Height = ModifyParameterValue(parameterHolder.Height, height); }
        }

        public Commands.ConfigScreenReset ResetCommand { get; set; }

        public Commands.ChangeWindow ChangeWindow { get; set; }

        public ConfigScreenVM(ParameterHolder parameterHolder) : base(parameterHolder)
        {
            InVel = parameterHolder.InflowVelocity.Value;
            Chi = parameterHolder.SurfaceFriction.Value;
            Width = parameterHolder.Width.Value;
            Height = parameterHolder.Height.Value;
            ResetCommand = new Commands.ConfigScreenReset(this, parameterHolder);
            ChangeWindow = new Commands.ChangeWindow();
        }
    }
}
