using UserInterface.HelperClasses;
using UserInterface.HelperControls;

namespace UserInterface.ViewModels
{
    public class ConfigScreenVM : ViewModel
    {
        private readonly UnitConversionPanel unitsPanel;

        public float InVel
        {
            get => parameterHolder.InflowVelocity.Value;
            set
            {
                parameterHolder.InflowVelocity = ModifyParameterValue(parameterHolder.InflowVelocity, value);
                OnPropertyChanged(this, nameof(InVel));
            }
        }
        public float Chi
        {
            get => parameterHolder.SurfaceFriction.Value;
            set
            {
                parameterHolder.SurfaceFriction = ModifyParameterValue(parameterHolder.SurfaceFriction, value);
                OnPropertyChanged(this, nameof(Chi));
            }
        }
        public float Width
        {
            get => parameterHolder.Width.Value;
            set
            {
                parameterHolder.Width = ModifyParameterValue(parameterHolder.Width, value);
                OnPropertyChanged(this, nameof(Width));
            }
        }
        public float Height
        {
            get => parameterHolder.Height.Value;
            set
            {
                parameterHolder.Height = ModifyParameterValue(parameterHolder.Height, value);
                OnPropertyChanged(this, nameof(Height));
            }
        }
        public float ReynoldsNo
        {
            get => parameterHolder.ReynoldsNumber.Value;
            set
            {
                parameterHolder.ReynoldsNumber = ModifyParameterValue(parameterHolder.ReynoldsNumber, value);
                OnPropertyChanged(this, nameof(ReynoldsNo));
            }
        }
        public float Viscosity
        {
            get => parameterHolder.FluidViscosity.Value;
            set {
                parameterHolder.FluidViscosity = ModifyParameterValue(parameterHolder.FluidViscosity, value);
                OnPropertyChanged(this, nameof(Viscosity));
            }
        }
        public float Density
        {
            get => parameterHolder.FluidDensity.Value;
            set
            {
                parameterHolder.FluidDensity = ModifyParameterValue(parameterHolder.FluidDensity, value);
                OnPropertyChanged(this, nameof(Density));
            }
        }

        public UnitConversionPanel UnitsPanel => unitsPanel;

        public Commands.ConfigScreenReset ResetCommand { get; set; }
        public Commands.SetAirParameters SetAirCommand { get; set; }
        public Commands.ChangeWindow ChangeWindowCommand { get; set; }
        public Commands.CreatePopup CreatePopupCommand { get; set; }

        public ConfigScreenVM(ParameterHolder parameterHolder, UnitHolder unitHolder) : base(parameterHolder, unitHolder)
        {
            InVel = parameterHolder.InflowVelocity.Value;
            Chi = parameterHolder.SurfaceFriction.Value;
            Width = parameterHolder.Width.Value;
            Height = parameterHolder.Height.Value;
            ReynoldsNo = parameterHolder.ReynoldsNumber.Value;
            Viscosity = parameterHolder.FluidViscosity.Value;
            Density = parameterHolder.FluidDensity.Value;
            unitsPanel = new UnitConversionPanel(unitHolder);
            ResetCommand = new Commands.ConfigScreenReset(this, parameterHolder);
            SetAirCommand = new Commands.SetAirParameters(this);
            ChangeWindowCommand = new Commands.ChangeWindow();
            CreatePopupCommand = new Commands.CreatePopup();
        }
    }
}
