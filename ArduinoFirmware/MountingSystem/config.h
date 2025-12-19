/**
******************************************************************************
  @file      config.h
  @author    Scott Thomason
  @date      19 Nov 2021
  @brief     Mounting system controller for the VNA mounting system on the 
             MudMaster.
  REFERENCE:
******************************************************************************
  VERSION HISTORY
******************************************************************************
* 
******************************************************************************

******************************************************************************
*/

#ifndef CONFIG_H
#define CONFIG_H

// Controller modes
#define CONTROLLER_IDLE     0
#define CONTROLLER_CAL      1
#define CONTROLLER_ACTIVE   2

// Actuator limits
#define ACTUATOR_LENGTH     140.0 // mm
#define ACTUATOR_MIN        0.0 // mm
#define ACTUATOR_MAX        140.0 // mm
#define ACTUATOR_ALLOWANCE  0.2 // mm
#define ACTUATOR_SPEED_HIGH 255 // high speed for when the distance difference is high
#define ACTUATOR_SPEED_LOW  100 // low speed for when the distance difference is small
#define ACTUATOR_SPEED_DIST 2   // mm - distance used to change the speed to the slow one

// Use Time of Flight sensor for height
#define USE_TOF             false

// Commands for data over serial
// Setters use upper case
#define SET_HEIGHT          'H'
#define SET_HEIGHT_CURRENT  'F' // height feedback when it is provided by the python program
#define SET_MODE            'M'
#define SET_ACTUATOR_POS    'P'

#define SET_DUTY            'D'
#define MOVE_OUT            'A'
#define MOVE_IN             'a'

// Getters use lower case
#define GET_HEIGHT          'h'
#define GET_TEMP            't'
#define GET_VIBRATION       'v'
#define GET_MODE            'm'
#define GET_ACTUATOR_POS    'p'


// Pins
// Motor is run using PWM
#define ACTUATOR_PWM        3
#define ACTUATOR_DIR        12
#define ACTUATOR_BRAKE      9
#define ACTUATOR_FEED_POS   A2 // feedback from potentiometer on actuator
#define ACTUATOR_FEED_CUR   A0 // feedback from current sensing on the motor driver

#endif
