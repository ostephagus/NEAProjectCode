using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using UserInterface.Converters;
using UserInterface.HelperClasses;
using UserInterface.HelperControls;
using UserInterface.ViewModels;

namespace UserInterface.Views
{
    /// <summary>
    /// Interaction logic for SimulationScreen.xaml
    /// </summary>
    public partial class SimulationScreen : SwappableScreen
    {
        private readonly SimulationScreenVM viewModel;
        private readonly AbsoluteRectToRelativePol RecToPolConverter;
        private readonly RelativePolToAbsoluteRect PolToRecConverter;
        public SimulationScreenVM ViewModel => viewModel;

        public SimulationScreen(ParameterHolder parameterHolder) : base(parameterHolder)
        {
            InitializeComponent();
            viewModel = new SimulationScreenVM(parameterHolder);
            DataContext = viewModel;
            RecToPolConverter = new();
            PolToRecConverter = new();
            //PlaceInitialPoints();
        }
        private VisualPoint? draggedPoint;
        private Point mousePosition;

        private void PlaceInitialPoints()
        {
            foreach (PolarPoint polarPoint in ViewModel.ControlPoints)
            {
                SimulationCanvas.Children.Add(new VisualPoint((Point)PolToRecConverter.Convert([polarPoint, SimulationCanvas.ActualWidth, SimulationCanvas.ActualHeight], typeof(Point), ViewModel.ObstacleCentre, System.Globalization.CultureInfo.CurrentCulture)));
            }
        }

        private void AddPoint(VisualPoint point)
        {
            SimulationCanvas.Children.Add(point);
            ViewModel.ControlPoints.Add(ConvertToPolar(point.Point));
        }

        private void RemovePoint(VisualPoint point)
        {
            SimulationCanvas.Children.Remove(point);
            ViewModel.ControlPoints.Remove(ConvertToPolar(point.Point));
        }

        private void CanvasMouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            if (e.Source is VisualPoint point && SimulationCanvas.CaptureMouse())
            {
                mousePosition = e.GetPosition(SimulationCanvas);
                draggedPoint = point;

                Panel.SetZIndex(draggedPoint, 1); // Make point go in front of everything else while is is dragged
            }
            else
            {
                Point mousePosition = e.GetPosition(SimulationCanvas);
                AddPoint(new VisualPoint(new Point(mousePosition.X / 2, mousePosition.Y / 2))); // Halve the point position.
            }
        }

        private void CanvasMouseLeftButtonUp(object sender, MouseButtonEventArgs e)
        {
            if (draggedPoint is not null)
            {
                SimulationCanvas.ReleaseMouseCapture();
                Panel.SetZIndex(draggedPoint, 0);
                draggedPoint = null;
            }
        }

        /// <summary>
        /// Fills in the common parameters and calls <see cref="AbsoluteRectToRelativePol.Convert(object[], Type, object, System.Globalization.CultureInfo)"/>, like a partial function application.
        /// </summary>
        /// <param name="rectangularPoint">The absolute rectangular point to convert</param>
        /// <returns>A relative polar coordinate.</returns>
        private PolarPoint ConvertToPolar(Point rectangularPoint)
        {
            return (PolarPoint)RecToPolConverter.Convert([rectangularPoint, SimulationCanvas.ActualWidth, SimulationCanvas.ActualHeight], typeof(PolarPoint), new Point(0.5, 0.5), System.Globalization.CultureInfo.CurrentCulture);
        }

        private void CanvasMouseRightButtonDown(object sender, MouseButtonEventArgs e)
        {
            if (e.Source is VisualPoint point)
            {
                RemovePoint(point);
            }
        }

        private void CanvasMouseMove(object sender, MouseEventArgs e)
        {
            if (draggedPoint != null)
            {
                Point position = e.GetPosition(SimulationCanvas);
                Vector offset = position - mousePosition;
                mousePosition = position;
                PolarPoint oldPoint = (PolarPoint)RecToPolConverter.Convert(new object[] { draggedPoint.Point, SimulationCanvas.ActualWidth, SimulationCanvas.ActualHeight }, typeof(PolarPoint), new Point(0.5, 0.5), System.Globalization.CultureInfo.CurrentCulture);
                draggedPoint.Point += offset;
                PolarPoint newPoint = (PolarPoint)RecToPolConverter.Convert(new object[] { draggedPoint.Point, SimulationCanvas.ActualWidth, SimulationCanvas.ActualHeight }, typeof(PolarPoint), new Point(0.5, 0.5), System.Globalization.CultureInfo.CurrentCulture);
                ViewModel.ControlPoints[ViewModel.ControlPoints.IndexOf(oldPoint)] = newPoint; // Replace the point.
            }
        }
    }
}
