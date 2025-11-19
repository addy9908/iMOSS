//-------------------------------------------------------------------------------------
// zy_tst_NIDA for tail suspension test with 4 load cells
//Author: Zengyou Ye at NIDA/IRP
//Date: 10/23/2024

// Settling time (number of samples) and data filtering can be adjusted in the config.h file of HX711_ADC.h
//---------------------------------

#include <HX711_ADC.h>

//pins:
const int HX711_dout_1 = 2; //mcu > HX711 no 1 dout pin
const int HX711_sck_1 = 3; //mcu > HX711 no 1 sck pin
const int HX711_dout_2 = 4; //mcu > HX711 no 2 dout pin
const int HX711_sck_2 = 5; //mcu > HX711 no 2 sck pin
const int HX711_dout_3 = 6; //mcu > HX711 no 3 dout pin
const int HX711_sck_3 = 7; //mcu > HX711 no 3 sck pin
const int HX711_dout_4 = 8; //mcu > HX711 no 4 dout pin
const int HX711_sck_4 = 9; //mcu > HX711 no 4 sck pin


//HX711 constructor (dout pin, sck pin)
HX711_ADC LoadCell_1(HX711_dout_1, HX711_sck_1); //HX711 1
HX711_ADC LoadCell_2(HX711_dout_2, HX711_sck_2); //HX711 2
HX711_ADC LoadCell_3(HX711_dout_3, HX711_sck_3); //HX711 3
HX711_ADC LoadCell_4(HX711_dout_4, HX711_sck_4); //HX711 4

unsigned long t = 0;
const int serialPrintInterval = 12.5; //80Hz to match 80SPS
static boolean togo = 0;


void setup() {
  Serial.begin(57600);
  delay(10);
  Serial.println();
  Serial.println("Starting...");


  float calibrationValue_1; // calibration value load cell 1
  float calibrationValue_2; // calibration value load cell 2
  float calibrationValue_3; // calibration value load cell 3
  float calibrationValue_4; // calibration value load cell 4


  calibrationValue_1 = 13.3; // uncomment this if you want to set this value in the sketch
  calibrationValue_2 = 13.3; // uncomment this if you want to set this value in the sketch
  calibrationValue_3 = 13.3; // uncomment this if you want to set this value in the sketch
  calibrationValue_4 = 13.3; // uncomment this if you want to set this value in the sketch

  LoadCell_1.begin();
  LoadCell_2.begin();
  LoadCell_3.begin();
  LoadCell_4.begin();

  unsigned long stabilizingtime = 2000; // tare preciscion can be improved by adding a few seconds of stabilizing time
  boolean _tare = true; //set this to false if you don't want tare to be performed in the next step
  byte loadcell_1_rdy = 0;
  byte loadcell_2_rdy = 0;
  byte loadcell_3_rdy = 0;
  byte loadcell_4_rdy = 0;


  while ((loadcell_1_rdy + loadcell_2_rdy + loadcell_3_rdy + loadcell_4_rdy) < 4) { //run startup, stabilization and tare, both modules simultaniously
    if (!loadcell_1_rdy) loadcell_1_rdy = LoadCell_1.startMultiple(stabilizingtime, _tare);
    if (!loadcell_2_rdy) loadcell_2_rdy = LoadCell_2.startMultiple(stabilizingtime, _tare);
    if (!loadcell_3_rdy) loadcell_3_rdy = LoadCell_3.startMultiple(stabilizingtime, _tare);
    if (!loadcell_4_rdy) loadcell_4_rdy = LoadCell_4.startMultiple(stabilizingtime, _tare);
  }
  if (LoadCell_1.getTareTimeoutFlag()) {
    Serial.println("Timeout, check MCU>HX711 no.1 wiring and pin designations");
  }
  if (LoadCell_2.getTareTimeoutFlag()) {
    Serial.println("Timeout, check MCU>HX711 no.2 wiring and pin designations");
  }
  if (LoadCell_3.getTareTimeoutFlag()) {
    Serial.println("Timeout, check MCU>HX711 no.3 wiring and pin designations");
  }
  if (LoadCell_4.getTareTimeoutFlag()) {
    Serial.println("Timeout, check MCU>HX711 no.4 wiring and pin designations");
  }
  LoadCell_1.setCalFactor(calibrationValue_1); // user set calibration value (float)
  LoadCell_2.setCalFactor(calibrationValue_2); // user set calibration value (float)
  LoadCell_3.setCalFactor(calibrationValue_3); // user set calibration value (float)
  LoadCell_4.setCalFactor(calibrationValue_4); // user set calibration value (float)
  Serial.print("HX711 measured sampling rate Hz: ");
  Serial.println(LoadCell_1.getSPS());
  Serial.println("Startup is complete");
}


void loop() {
  while (togo==0){
    char inByte = Serial.read();
    if (inByte == 'g') {
      togo=1;
      //Serial.println("ms,lc1,lc2,lc3,lc4");
      //Serial.println("experiments on");
    }
    if (inByte == 't') {
      LoadCell_1.tareNoDelay();
      LoadCell_2.tareNoDelay();
      LoadCell_3.tareNoDelay();
      LoadCell_4.tareNoDelay();
      delay(5000);
    }    
  }
  if(togo){
    LoadCell_1.update();
    LoadCell_2.update();
    LoadCell_3.update();
    LoadCell_4.update();

    if (millis() >= t + serialPrintInterval) {
      t = millis();
      float a = LoadCell_1.getData();
      float b = LoadCell_2.getData();
      float c = LoadCell_3.getData();
      float d = LoadCell_4.getData();
      
      Serial.print(t);
      Serial.print(",");
      Serial.print(a);
      Serial.print(",");
      Serial.print(b);
      Serial.print(",");
      Serial.print(c);
      Serial.print(",");
      Serial.println(d);
    }
  }
  if (Serial.read() == 'q') {
    togo=0;
  }
}
