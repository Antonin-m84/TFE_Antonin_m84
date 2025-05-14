#include <Wire.h>
#include "Adafruit_AS726x.h"
#include <stdio.h>

#define INT_PIN 2

int led1 = 8;
int led2 = 9;
int led3 = 10;
int led4 = 11;

int sampleCount = 0;
bool signalInProgress = false;

int serial_putc(char c, FILE *) {
  Serial.write(c);
  return c;
}

Adafruit_AS726x ams;

// On utilise un compteur pour enregistrer toutes les impulsions
volatile uint16_t signalRec = 0;

// L'ISR incrémente signalRec à chaque impulsion
void handleInterrupt() {
  signalRec++;
}

void setup() {
  Serial.begin(460800);
  Wire.begin();
  fdevopen(&serial_putc, 0);
  
  if (!ams.begin()) {
    Serial.println("Erreur de connexion capteur!");
    while (1);
  }
  
  ams.setIntegrationTime(3);
  ams.setConversionType(MODE_1);
  ams.setGain(GAIN_64X);
  
  pinMode(INT_PIN, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(INT_PIN), handleInterrupt, RISING);
  
  while (!ams.dataReady());
  
  pinMode(led1, OUTPUT);
  pinMode(led2, OUTPUT);
  pinMode(led3, OUTPUT);
  pinMode(led4, OUTPUT);
  
  digitalWrite(led1, HIGH);
  digitalWrite(led2, HIGH);
  digitalWrite(led3, HIGH);
  digitalWrite(led4, HIGH);
  
  Serial.println("initok");
}


void loop() {
  static unsigned long lastSampleTime = 0;
  static unsigned long currentTime = 0;
  const unsigned long realTimeoutMs = 200;  // Temps réel entre les signaux en millisecondes
  
  currentTime = millis();
  

  // Traitement de nouveaux échantillons
  if (signalRec > 0) {
    lastSampleTime = currentTime; // Mise à jour du moment de la dernière mesure
    
    // Si c'est notre premier échantillon après une pause, marquer le début d'un nouveau signal
    if (!signalInProgress) {
      signalInProgress = true;
    }
    
    // Traiter chaque échantillon disponible
    while (signalRec > 0) {
      noInterrupts();
      signalRec--;
      interrupts();
      
      uint8_t green_hgt = ams.virtualRead(AS7262_GREEN);
      uint8_t green_low = ams.virtualRead(AS7262_GREEN + 1);
      uint16_t green = (green_hgt << 8) | green_low;
      
      sampleCount++;
      
      if (sampleCount >= 200) {
        printf("%d,\r\n ", green);
        sampleCount = 0;
      } else {
        printf("%d, ", green);
      }
    }
  }

  // Vérification de la fin d'une série d'échantillons, soit un signal a été reçu, qu'il n'y a plus de pulse a 250Hz et que temps timeout dépassé
  if (signalInProgress && signalRec == 0 && currentTime - lastSampleTime > realTimeoutMs) {
    // Signal terminé
    if (sampleCount > 0) {
      // Fermeture propre du dernier buffer non complet
      printf("\r\ntimeout\r\n");
    }
    // Réinitialisation des compteurs
    signalInProgress = false;
    sampleCount = 0;
  }
}