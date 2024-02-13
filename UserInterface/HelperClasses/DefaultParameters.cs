namespace UserInterface.HelperClasses
{
    public static class DefaultParameters
    {
        public static readonly float WIDTH = 1f;
        public static readonly float HEIGHT = 1f;
        public static readonly float TIMESTEP_SAFETY_FACTOR = 0.8f;
        public static readonly float RELAXATION_PARAMETER = 1.7f;
        public static readonly float PRESSURE_RESIDUAL_TOLERANCE = 2f;
        public static readonly float PRESSURE_MAX_ITERATIONS = 1000f;
        public static readonly float REYNOLDS_NUMBER = 2000f;
        public static readonly float FLUID_VISCOSITY = 1.983E-5f;
        public static readonly float FLUID_VELOCITY = 5f;
        public static readonly float FLUID_DENSITY = 1.293f;
        public static readonly float SURFACE_FRICTION = 0f;

        public static readonly bool DRAW_CONTOURS = true;
        public static readonly float CONTOUR_TOLERANCE = 0.01f;
        public static readonly float CONTOUR_SPACING = 0.071f;
        public static readonly int FPS_WINDOW_SIZE = 500;
        public static readonly int DRAG_COEF_WINDOW_SIZE = 10;

        public static readonly float VELOCITY_MIN = 0f;
        public static readonly float VELOCITY_MAX = 18f;
        public static readonly float PRESSURE_MIN = 1000f;
        public static readonly float PRESSURE_MAX = 100000f;
    }
}
