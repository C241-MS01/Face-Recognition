#include <Wire.h>
#include <LiquidCrystal_I2C.h>

#define SDA 15
#define SCL 14

int buzzerPin = 5;
int lcdAddress = 0x27;

LiquidCrystal_I2C lcd(0x3F,16,2);

void setup() {
  Serial.begin(9600);
  pinMode(buzzerPin, OUTPUT);
  Wire.begin(SDA, SCL);
  lcd.init();   
  lcd.clear();           
  lcd.backlight(); 
  lcd.begin(16,2);
  lcd.backlight();
  lcd.clear();

  lcd.setCursor(6,0);
  lcd.print("UJI");
  lcd.setCursor(5,1);
  lcd.print("COBA");
  delay(3000);
  lcd.clear();
}

void loop() {
  digitalWrite(buzzerPin, HIGH);
  lcd.setCursor(0,0);
  lcd.print("NYOBA");

  }

