using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Shapes;

namespace UserInterface.HelperControls
{
    public class VisualPoint : Shape
    {
        private const int circleRadiusRatio = 3;
        private const int size = 10;

        private Point point;

        protected override Geometry DefiningGeometry => new EllipseGeometry(Point, size, size);

        public Point Point
        {
            get => point;
            set
            {
                point = value;
                Canvas.SetLeft(this, point.X);
                Canvas.SetTop(this, point.Y);
            }
        }

        public VisualPoint(Point point)
        {
            DrawingGroup drawingGroup = new DrawingGroup();
            drawingGroup.Children.Add(new GeometryDrawing // Outer circle
            {
                Brush = Brushes.DarkGreen,
                Geometry = new EllipseGeometry(new Point(0, 0), circleRadiusRatio, circleRadiusRatio)
            });
            drawingGroup.Children.Add(new GeometryDrawing // Inner, lighter circle
            {
                Brush = Brushes.LightGreen,
                Geometry = new EllipseGeometry(new Point(0, 0), 1, 1)
            });
            Fill = new DrawingBrush(drawingGroup);
            Point = point;
        }

        public VisualPoint() : this(new Point(0, 0)) { }

        public override string ToString()
        {
            return point.ToString();
        }
    }
}
