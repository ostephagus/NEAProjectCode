﻿using UserInterface.HelperClasses;
using UserInterface.HelperControls;

namespace UserInterface.ViewModels
{
    public class ConfigScreenVM : ViewModel
    {
        private float inVel;
        private float chi;
        private float width;
        private float height;
        private float reynoldsNo;
        private float viscosity;
        private float density;

        private readonly UnitConversionPanel unitsPanel;

        public float InVel
        {
            get => inVel;
            set
            {
                inVel = value;
                OnPropertyChanged(this, nameof(InVel));
                parameterHolder.InflowVelocity = ModifyParameterValue(parameterHolder.InflowVelocity, inVel);
            }
        }

        public float Chi
        {
            get => chi;
            set
            {
                chi = value;
                OnPropertyChanged(this, nameof(Chi));
                parameterHolder.SurfaceFriction = ModifyParameterValue(parameterHolder.SurfaceFriction, chi);
            }
        }

        public float Width
        {
            get => width;
            set
            {
                width = value;
                OnPropertyChanged(this, nameof(Width));
                parameterHolder.Width = ModifyParameterValue(parameterHolder.Width, width);
            }
        }

        public float Height
        {
            get => height;
            set
            {
                height = value;
                OnPropertyChanged(this, nameof(Height));
                parameterHolder.Height = ModifyParameterValue(parameterHolder.Height, height);
            }
        }

        public float ReynoldsNo
        {
            get => reynoldsNo;
            set
            {
                reynoldsNo = value;
                OnPropertyChanged(this, nameof(ReynoldsNo));
                parameterHolder.ReynoldsNumber = ModifyParameterValue(parameterHolder.ReynoldsNumber, reynoldsNo);
            }
        }

        public float Viscosity
        {
            get => viscosity;
            set {
                viscosity = value;
                OnPropertyChanged(this, nameof(Viscosity));
                parameterHolder.FluidViscosity = ModifyParameterValue(parameterHolder.FluidViscosity, viscosity);
            }
        }

        public float Density
        {
            get => density;
            set
            {
                density = value;
                OnPropertyChanged(this, nameof(Density));
                parameterHolder.FluidDensity = ModifyParameterValue(parameterHolder.FluidDensity, density);
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
