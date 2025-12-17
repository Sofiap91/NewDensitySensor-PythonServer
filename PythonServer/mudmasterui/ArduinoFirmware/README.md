# Arduino Firmware for Mounting System

## Linear Actuator
The linear actuator is controlled using a positive or negative signal on the power connections. Using no signal will keep it constant.

An actuator with a potentiometer gives feedback on the current position of the actuator. This could be used as part of the control loop for the actuator.

## Accelerometer
This is used to measure the amount of vibrations that are transferring from the MudMaster to the VNA and antenna system. 

## Digital Temperature Sensor
This is used to measure the temperature in the VNA enclosure. This will be used to track how hot it is getting. If it is getting too hot, the main Python script could revert to a lower data acquisition rate.