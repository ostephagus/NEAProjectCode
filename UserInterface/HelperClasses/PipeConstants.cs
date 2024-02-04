namespace UserInterface.HelperClasses
{
    /// <summary>
    /// Constants for pipe communication, containing all the control bytes as defined in Documentation D.3 Precise Specification
    /// </summary>
    internal static class PipeConstants
    {
        public static readonly byte NULL = 0;
        public static readonly byte CATEGORYMASK = 0b11000000;

        /// <summary>
        /// STATUS bytes, providing information to the client or commands to do with program state
        /// </summary>
        public static class Status
        {
            public static readonly byte GENERIC = 0b00000000;
            public static readonly byte HELLO = 0b00001000;
            public static readonly byte BUSY = 0b00010000;
            public static readonly byte OK = 0b00011000;
            public static readonly byte STOP = 0b00100000;
            public static readonly byte CLOSE = 0b00101000;

            public static readonly byte PARAMMASK = 0b00000111;
        }

        /// <summary>
        /// REQUEST bytes, to request data to be calculated and sent by the client
        /// </summary>
        public static class Request
        {
            public static readonly byte GENERIC = 0b01000000;
            public static readonly byte FIXLENREQ = 0b01000000;
            public static readonly byte CONTREQ = 0b01100000;

            public static readonly byte PARAMMASK = 0b00011111;

            public static readonly byte HVEL = 0b00010000;
            public static readonly byte VVEL = 0b00001000;
            public static readonly byte PRES = 0b00000100;
            public static readonly byte STRM = 0b00000010;
        }

        /// <summary>
        /// MARKER bytes, to denote start and end of fields, timestep iterations, or parameters
        /// </summary>
        public static class Marker
        {
            public static readonly byte GENERIC = 0b10000000;
            public static readonly byte ITERSTART = 0b10000000;
            public static readonly byte ITEREND = 0b10001000;
            public static readonly byte FLDSTART = 0b10010000;
            public static readonly byte FLDEND = 0b10011000;

            public static readonly byte ITERPRMMASK = 0b00000111;

            public static readonly byte HVEL = 0b00000001;
            public static readonly byte VVEL = 0b00000010;
            public static readonly byte PRES = 0b00000011;
            public static readonly byte STRM = 0b00000100;
            public static readonly byte OBST = 0b00000101;

            public static readonly byte PRMSTART = 0b10100000;
            public static readonly byte PRMEND = 0b10101000;

            public static readonly byte PRMMASK = 0b00001111;

            public static readonly byte IMAX = 0b00000001;
            public static readonly byte JMAX = 0b00000010;
            public static readonly byte WIDTH = 0b00000011;
            public static readonly byte HEIGHT = 0b00000100;
            public static readonly byte TAU = 0b00000101;
            public static readonly byte OMEGA = 0b00000110;
            public static readonly byte RMAX = 0b00000111;
            public static readonly byte ITERMAX = 0b00001000;
            public static readonly byte REYNOLDS = 0b00001001;
            public static readonly byte INVEL = 0b00001010;
            public static readonly byte CHI = 0b00001011;
            public static readonly byte MU = 0b00001100;
        }

        /// <summary>
        /// ERROR bytes, sent due to errors in data or internal stop codes
        /// </summary>
        public static class Error
        {
            public static readonly byte GENERIC = 0b11000000;
            public static readonly byte BADREQ = 0b11000001;
            public static readonly byte BADPARAM = 0b11000010;
            public static readonly byte INTERNAL = 0b11000011;
            public static readonly byte TIMEOUT = 0b11000100;
            public static readonly byte BADTYPE = 0b11000101;
            public static readonly byte BADLEN = 0b11000110;
        }
    }
}
