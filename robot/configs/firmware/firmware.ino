#include <CommandHandler.h>
#include <CommandManager.h>

CommandManager cmdMng;
#include <AccelStepper.h>
#include <LinearAccelStepperActuator.h>
#include <CommandLinearAccelStepperActuator.h>

AccelStepper stp_X(AccelStepper::DRIVER, 54, 55);
CommandLinearAccelStepperActuator X(stp_X, 2, 38);

AccelStepper stp_Y(AccelStepper::DRIVER, 60, 61);
CommandLinearAccelStepperActuator Y(stp_Y, 3, 56);

AccelStepper stp_Z(AccelStepper::DRIVER, 46, 48);
CommandLinearAccelStepperActuator Z(stp_Z, 15, 62);

AccelStepper stp_CX(AccelStepper::DRIVER, 26, 28);
CommandLinearAccelStepperActuator CX(stp_CX, 19, 24);

AccelStepper stp_CY(AccelStepper::DRIVER, 36, 34);
CommandLinearAccelStepperActuator CY(stp_CY, 18, 30);

#include <CommandAnalogWrite.h>
CommandAnalogWrite Fan(9);


void setup()
{
  Serial.begin(115200);
  X.registerToCommandManager(cmdMng, "X");
  Y.registerToCommandManager(cmdMng, "Y");
  Z.registerToCommandManager(cmdMng, "Z");
  CX.registerToCommandManager(cmdMng, "CX");
  CY.registerToCommandManager(cmdMng, "CY");
  Fan.registerToCommandManager(cmdMng, "Fan");
  cmdMng.init();
}

void loop()
{
   cmdMng.update();
}
