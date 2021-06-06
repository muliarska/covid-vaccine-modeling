//
// Created by Yaroslav Morozevych on 4/21/21.
//

#ifndef COVID_MODEL_CONSTANTS_HPP
#define COVID_MODEL_CONSTANTS_HPP

// INCLUDE ERROR CODES
#define OK 0
#define IO_READ_ERR 1
#define INCORRECT_INPUT 2
#define CFG_VALUE_ERROR 3

#define HAS_FLAG(STATE) ((STATE == E_STATE) || (STATE == I_STATE) || (STATE == R_STATE)) ? (true) : (false)

// INCLUDE STATIC VAR DEFINITIONS
#define INFECT_PROB 0.1
#define WINDOW_SIZE 32
#define BUILD_TIMESTEP 10
#define LOCKDOWN_DURATION 21

// INCLUDE STATIC STATE DEFINES
#define S_STATE 's'
#define E_STATE 'e'
#define I_STATE 'i'
#define R_STATE 'r'
#define V_STATE 'v'
#define D_STATE 'd'
#define M_STATE 'm'

#endif //COVID_MODEL_CONSTANTS_HPP
