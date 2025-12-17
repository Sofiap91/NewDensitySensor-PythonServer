/**
 ******************************************************************************
   @file      MountingSystem
   @author    Scott Thomason
   @date      19 Nov 2021
   @brief     Mounting System controller for MudMaster.
   REFERENCE:
 ******************************************************************************
*/

/* Includes ------------------------------------------------------------------*/
#include <stdio.h>
#include <stdint.h>
#include <ctype.h>

#include "config.h"


/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
#define ACTUATOR_IDLE     1
#define ACTUATOR_OUT      2
#define ACTUATOR_IN       3
/* Private functions ---------------------------------------------------------*/
void actuatorControl(int control);
float actuatorPosition(void);

/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
int currentMode = CONTROLLER_IDLE;

// height
float height_current = 0.0;
float height_target = 0.0;
bool  height_changing = false;

// temperature
float temperature_current = 0.0;

// vibrations
float vibration_x = 0.0;
float vibration_y = 0.0;
float vibration_z = 0.0;

int actuator_state = ACTUATOR_IDLE;
float actuator_position = 0.0;
float actuator_position_target = 0.0;
float actuator_position_diff = 0.0;

int actuator_duty = 100;

char command;
int commandValue_int;
float commandValue_float;
float height_diff;

void setup() {
  Serial.begin(9600);
  pinMode(ACTUATOR_PWM, OUTPUT);
  pinMode(ACTUATOR_DIR, OUTPUT);
  pinMode(ACTUATOR_BRAKE, OUTPUT);
  
  digitalWrite(ACTUATOR_DIR, HIGH);
  digitalWrite(ACTUATOR_BRAKE, LOW);

}

void loop() {
  actuator_position = get_actuatorPosition();
  // check whether the target height is different to the current height
  if(actuator_position != actuator_position_target) {
    // either the current height or the target height has changed: try using 
    // the actuator to compensate.
    actuator_position_diff = actuator_position_target - actuator_position;

    if(actuator_position_diff > ACTUATOR_ALLOWANCE || actuator_position_diff < -ACTUATOR_ALLOWANCE) {
      //Serial.print("Diff: ");
      //Serial.print(actuator_position_diff);
      //Serial.print("\t");
      //Serial.print("\t\tCurrent: ");
      Serial.println(get_actuatorCurrent());
      //Serial.print("\t\tActuator State: ");
      //Serial.println(actuator_state);

    }
    

    // negative difference means that the current height is greater (higher off 
    // the ground) than the target height - see if we can use the actuator to 
    // lower it
    if(actuator_position_diff > ACTUATOR_ALLOWANCE && actuator_position < ACTUATOR_MAX) {
      // tell the actuator to start lowering
      actuatorControl(ACTUATOR_OUT, actuator_position_diff);
    }

    // positive difference means that the current height is lower (closer to 
    // the ground) than the target height - see if we can use the actuator to 
    // raise it
    else if(actuator_position_diff < -ACTUATOR_ALLOWANCE && actuator_position > ACTUATOR_MIN) {
      // tell the actuator to start raising
      actuatorControl(ACTUATOR_IN, actuator_position_diff);
      
    } 
    
    else {
      actuatorControl(ACTUATOR_IDLE, 0.0);
    }
    
  } else {
    actuatorControl(ACTUATOR_IDLE, 0.0);
  }

  

  // Now process commands
  if(Serial.available() > 0) {
    command = Serial.read();

    switch(command) {
      // height stuff
      case SET_HEIGHT: 
        // this is separated by a comma, read to remove it
        Serial.read();
        
        // Need to set the current target height to the new height
        commandValue_float = Serial.parseFloat();
        Serial.println(commandValue_float);
        Serial.println(commandValue_float == 0.0);
        if(commandValue_float != 0.0 && commandValue_float > 0.0) {
          height_target = commandValue_float;
          // set the height changing flag to instruct it to move
          height_changing = true;
        }
        
        break;
        
      case SET_HEIGHT_CURRENT: 
        // this is separated by a comma, read to remove it
        Serial.read();
        
        // Need to set the current target height to the new height
        commandValue_float = Serial.parseFloat();
        if(commandValue_float >= 0.0 && USE_TOF == false) {
          height_current = commandValue_float;
        }
        
        break;
        
      case GET_HEIGHT:
        // returns the current height, target height, and whether it is currently moving, separated by commas
        Serial.print(height_current);
        Serial.print(",");
        Serial.print(height_target);
        Serial.print(",");
        Serial.println(height_changing);
        break;

      // device mode
      case SET_MODE:
        // sets the current operating state of the system
        // this is separated by a comma, read to remove it
        Serial.read();

        commandValue_int = Serial.parseInt();
        if(commandValue_int == CONTROLLER_IDLE || commandValue_int == CONTROLLER_CAL || commandValue_int == CONTROLLER_ACTIVE) {
          currentMode = commandValue_int;
        }
        break;

      // get mode
      case GET_MODE:
        // returns the current mode
        Serial.println(currentMode);
        break;

      // temperature
      case GET_TEMP:
        // returns the current temperature
        Serial.println(temperature_current);
        break;

      // vibrations
      case GET_VIBRATION:
        // returns the vibration_x, vibration_x, and vibration_x, separated by commas
        Serial.print(vibration_x);
        Serial.print(",");
        Serial.print(vibration_y);
        Serial.print(",");
        Serial.println(vibration_z);
        break;

      // actuator position
      case GET_ACTUATOR_POS:
        // returns the current temperature
        Serial.print(actuator_position);
        Serial.print(",");
        Serial.println(actuator_position_target);
        break;

      // actuator increment 10
      case MOVE_OUT:
        actuator_position_target += 10;
        if(actuator_position_target > ACTUATOR_MAX) {
          actuator_position_target = ACTUATOR_MAX;
        }

        Serial.println(actuator_position_target);
        break;

      // actuator decrement 10
      case MOVE_IN:
        actuator_position_target -= 10;
        if(actuator_position_target < ACTUATOR_MIN) {
          actuator_position_target = ACTUATOR_MIN;
        }

        Serial.println(actuator_position_target);
        break;

      // device mode
      case SET_DUTY:
        // sets the current duty cycle of the actuator
        // this is separated by a comma, read to remove it
        Serial.read();

        actuator_duty = Serial.parseInt();
        if(actuator_duty > 100) {
          actuator_duty = 100;
          
        } else if(actuator_duty < 0) {
          actuator_duty = 0;
        }
        break;
    }
  }

  delay(10);

}

/**
  @brief  Set the state of the relays that control the actuator.
  @param  control - which direction to move the actuator.
  @retval None
*/
void actuatorControl(int control, float distance) {
  int actuator_speed = (int) (ACTUATOR_SPEED_HIGH * actuator_duty / 100);

  if(distance <= ACTUATOR_SPEED_DIST || distance >= -ACTUATOR_SPEED_DIST) {
    actuator_speed = ACTUATOR_SPEED_LOW;
  }
  
  switch(control) {
    case ACTUATOR_OUT:
      // lengthen the actuator
      if(actuator_state != ACTUATOR_OUT) {
        digitalWrite(ACTUATOR_DIR, HIGH);
        analogWrite(ACTUATOR_PWM, actuator_speed);

        actuator_state = ACTUATOR_OUT;
      }
      

      break;
      
    case ACTUATOR_IN:
      // retract the actuator
      if(actuator_state != ACTUATOR_IN) {
        digitalWrite(ACTUATOR_DIR, LOW);
        analogWrite(ACTUATOR_PWM, actuator_speed);

        actuator_state = ACTUATOR_IN;
      }

      break;
      
    default:
      // idle
      if(actuator_state != ACTUATOR_IDLE) {
        digitalWrite(ACTUATOR_DIR, HIGH);
        analogWrite(ACTUATOR_PWM, 0);

        actuator_state = ACTUATOR_IDLE;
      }
      

      break;
  }
}


/**
  @brief  Gets the amount of current being drawn by the linear actuator.
  @param  None
  @retval None
*/
float get_actuatorCurrent(void) {
  return analogRead(ACTUATOR_FEED_CUR) * (3.03 / 1023.0);
}


/**
  @brief  Gets the current extension length of the linear actuator.
  @param  control - which direction to move the actuator.
  @retval None
*/
float get_actuatorPosition(void) {
  return analogRead(ACTUATOR_FEED_POS) * (ACTUATOR_LENGTH / 1023.0);
}
