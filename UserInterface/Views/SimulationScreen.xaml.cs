using System;
using System.ComponentModel;
using System.Diagnostics;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Shapes;
using UserInterface.Converters;
using UserInterface.HelperClasses;
using UserInterface.HelperControls;
using UserInterface.ViewModels;

namespace UserInterface.Views
{
    /// <summary>
    /// Interaction logic for SimulationScreen.xaml
    /// </summary>
    public partial class SimulationScreen : UserControl
    {
        private const float CELL_EXTRA_SIZE = 0.2f;

        private readonly SimulationScreenVM viewModel;
        private readonly RectangularToPolar RecToPolConverter;
        public SimulationScreenVM ViewModel => viewModel;

        private VisualPoint? draggedPoint;
        private bool isCentreMoved;
        private Point mousePosition;
        private bool pointsPlaced;

        private static readonly double POINT_TOLERANCE = 0.1f;

        public SimulationScreen(ParameterHolder parameterHolder, UnitHolder unitHolder, ObstacleHolder obstacleHolder)
        {
            InitializeComponent();

            viewModel = new SimulationScreenVM(parameterHolder, unitHolder, obstacleHolder);
            DataContext = viewModel;

            RecToPolConverter = new();
            pointsPlaced = false;
            isCentreMoved = false;
            ViewModel.PropertyChanged += OnViewModelPropertyChanged;
            if (obstacleHolder.UsingObstacleFile)
            {
                PlaceObstacleCells();
            }
        }

        #region Private helper methods
        private Point MakeAbsolute(Point point) => point + (Vector)ViewModel.ObstacleCentre;
        private Point MakeRelative(Point point) => point - (Vector)ViewModel.ObstacleCentre;

        private void PlaceObstacleCells()
        {
            foreach (ObstacleCell cell in ViewModel.ObstacleCells)
            {
                Rectangle rectangle = new Rectangle
                {
                    Width = cell.Width + CELL_EXTRA_SIZE,
                    Height = cell.Height + CELL_EXTRA_SIZE,
                    Fill = Brushes.Black,
                    Stroke = null
                };

                Canvas.SetLeft(rectangle, cell.X);
                Canvas.SetBottom(rectangle, cell.Y);
                SimulationCanvas.Children.Add(rectangle);
            }
        }

        private void PlaceInitialPoints()
        {
            foreach (Point point in ViewModel.ControlPoints)
            {
                SimulationCanvas.Children.Add(new VisualPoint(MakeAbsolute(point)));
            }
            SimulationCanvas.Children.Add(new VisualPoint(ViewModel.ObstacleCentre));
        }

        private void MoveControlPoints(Vector translation)
        {
            for (int i = 0; i < SimulationCanvas.Children.Count; i++)
            {
                if (SimulationCanvas.Children[i] is VisualPoint point)
                {
                    point.Point += translation;
                }
            }
        }

        private void AddPoint(VisualPoint point)
        {
            SimulationCanvas.Children.Add(point);
            ViewModel.ControlPoints.Add(MakeRelative(point.Point));
        }

        private void RemovePoint(VisualPoint point)
        {
            SimulationCanvas.Children.Remove(point);
            int pointIndex = FindIndexOfPoint(point.Point);
            ViewModel.ControlPoints.RemoveAt(pointIndex);
        }

        private int FindIndexOfPoint(Point point)
        {
            int pointIndex = ViewModel.ControlPoints.IndexOf(MakeRelative(point));
            if (pointIndex == -1)
            {
                for (int i = 0; i < ViewModel.ControlPoints.Count; i++)
                {
                    Point comparisonPoint = MakeAbsolute(ViewModel.ControlPoints[i]);
                    if ((point - comparisonPoint).LengthSquared < POINT_TOLERANCE * POINT_TOLERANCE) // Allow some inexactness in distance from comparison point
                    {
                        pointIndex = i;
                    }
                }
                if (pointIndex == -1) // If still not found
                {
                    throw new InvalidOperationException("Could not find index of polar point.");
                }
            }

            return pointIndex;
        }
        #endregion

        #region Event handlers
        private void OnViewModelPropertyChanged(object? sender, PropertyChangedEventArgs e)
        {
            switch (e.PropertyName)
            {
                case nameof(ViewModel.EditingObstacles):
                    OnEditingObstalesChanged();
                    break;
                default:
                    break;
            }
        }

        private void OnEditingObstalesChanged()
        {
            if (ViewModel.EditingObstacles && !pointsPlaced) // Place the points the first time app enters obstacle editing.
            {
                PlaceInitialPoints();
                pointsPlaced = true;
            }

            foreach (UIElement element in SimulationCanvas.Children) // Set the visibility of all of the control points on entering/leaving obstacle editing
            {
                if (element is VisualPoint)
                {
                    element.Visibility = ViewModel.EditingObstacles ? Visibility.Visible : Visibility.Hidden;
                }
            }
        }

        private void CanvasMouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            if (e.Source is VisualPoint point && SimulationCanvas.CaptureMouse())
            {
                mousePosition = e.GetPosition(SimulationCanvas);
                draggedPoint = point;
                draggedPoint.IsDragged = true;
                //Trace.WriteLine($"difference in X coordinate: {Math.Abs(draggedPoint.Point.X - ViewModel.ObstacleCentre.X)}");
                //Trace.WriteLine($"difference in Y coordinate: {Math.Abs(draggedPoint.Point.Y - ViewModel.ObstacleCentre.Y)}");
                isCentreMoved = (draggedPoint.Point - ViewModel.ObstacleCentre).LengthSquared < POINT_TOLERANCE * POINT_TOLERANCE;

                Panel.SetZIndex(draggedPoint, 1); // Make point go in front of everything else while is is dragged
            }
            else
            {
                Point clickPosition = e.GetPosition(SimulationCanvas); // Check whether mouse positions map correctly or not.
                AddPoint(new VisualPoint(clickPosition.X, 100 - clickPosition.Y));
            }
        }

        private void CanvasMouseLeftButtonUp(object sender, MouseButtonEventArgs e)
        {
            if (draggedPoint is not null)
            {
                SimulationCanvas.ReleaseMouseCapture();
                Panel.SetZIndex(draggedPoint, 0);
                draggedPoint.IsDragged = false;
                draggedPoint = null;
                isCentreMoved = false;
            }
        }

        private void CanvasMouseRightButtonDown(object sender, MouseButtonEventArgs e)
        {
            if (e.Source is VisualPoint point && point.Point != ViewModel.ObstacleCentre)
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
                Vector offsetYFlip = new Vector(offset.X, -offset.Y);
                if (!isCentreMoved) // Normal control points
                {
                    int draggedPointIndex = FindIndexOfPoint(draggedPoint.Point);
                    draggedPoint.Point += offsetYFlip;
                    ViewModel.ControlPoints[draggedPointIndex] = MakeRelative(draggedPoint.Point);
                }
                else
                {
                    MoveControlPoints(offsetYFlip);

                    ViewModel.ObstacleCentre = draggedPoint.Point;
                }
            }
        }
        #endregion
    }
}
