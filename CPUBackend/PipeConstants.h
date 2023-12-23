#ifndef PIPE_CONSTANTS_H
#define PIPE_CONSTANTS_H
#include "Definitions.h"

namespace PipeConstants {
    constexpr BYTE CATEGORYMASK = 0b11000000;

    namespace Status
    {
        constexpr BYTE GENERIC = 0b00000000;
        constexpr BYTE HELLO = 0b00001000;
        constexpr BYTE BUSY = 0b00010000;
        constexpr BYTE OK = 0b00011000;
        constexpr BYTE STOP = 0b00100000;
        constexpr BYTE CLOSE = 0b00101000;

        constexpr BYTE PARAMMASK = 0b00000111;
    }
    namespace Request
    {
        constexpr BYTE GENERIC = 0b01000000;
        constexpr BYTE FIXLENREQ = 0b01000000;
        constexpr BYTE CONTREQ = 0b01100000;

        constexpr BYTE PARAMMASK = 0b00011111;

        constexpr BYTE HVEL = 0b00010000;
        constexpr BYTE VVEL = 0b00001000;
        constexpr BYTE PRES = 0b00000100;
        constexpr BYTE STRM = 0b00000010;
    }
    namespace Marker
    {
        constexpr BYTE GENERIC = 0b10000000;
        constexpr BYTE ITERSTART = 0b10000000;
        constexpr BYTE ITEREND = 0b10001000;
        constexpr BYTE FLDSTART = 0b10010000;
        constexpr BYTE FLDEND = 0b10011000;

        constexpr BYTE ITERPRMMASK = 0b00000111;

        constexpr BYTE HVEL = 0b00000001;
        constexpr BYTE VVEL = 0b00000010;
        constexpr BYTE PRES = 0b00000011;
        constexpr BYTE STRM = 0b00000100;
        constexpr BYTE OBST = 0b00000101;

        constexpr BYTE PRMSTART = 0b10100000;
        constexpr BYTE PRMEND = 0b10101000;

        constexpr BYTE PRMMASK = 0b00001111;

        constexpr BYTE IMAX = 0b00000001;
        constexpr BYTE JMAX = 0b00000010;
        constexpr BYTE WIDTH = 0b00000011;
        constexpr BYTE HEIGHT = 0b00000100;
        constexpr BYTE TAU = 0b00000101;
        constexpr BYTE OMEGA = 0b00000110;
        constexpr BYTE RMAX = 0b00000111;
        constexpr BYTE ITERMAX = 0b00001000;
        constexpr BYTE REYNOLDS = 0b00001001;
        constexpr BYTE INVEL = 0b00001010;
        constexpr BYTE CHI = 0b00001011;
    }
    namespace Error
    {
        constexpr BYTE GENERIC = 0b11000000;
        constexpr BYTE BADREQ = 0b11000001;
        constexpr BYTE BADPARAM = 0b11000010;
        constexpr BYTE INTERNAL = 0b11000011;
        constexpr BYTE TIMEOUT = 0b11000100;
        constexpr BYTE BADTYPE = 0b11000101;
        constexpr BYTE BADLEN = 0b11000110;
    }
}


#endif // !PIPE_CONSTANTS_H