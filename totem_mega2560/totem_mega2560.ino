#include <SPI.h>
#include <MFRC522.h>
#define SS_PIN 53
#define RST_PIN 5
#define LED_VERDE 8
#define LED_VERMELHO 9
#define BUZZER 7
MFRC522 leitor(SS_PIN, RST_PIN);
const String cartao1 = "9E49D706";
const String cartao2 = "EE0A8906";
void erro() {
  digitalWrite(LED_VERMELHO, HIGH);
  tone(BUZZER, 400, 350);
  delay(450);
  digitalWrite(LED_VERMELHO, LOW);
  noTone(BUZZER);
}
void setup() {
  Serial.begin(9600);
  SPI.begin();
  leitor.PCD_Init();
  pinMode(LED_VERDE, OUTPUT);
  pinMode(LED_VERMELHO, OUTPUT);
  pinMode(BUZZER, OUTPUT);
  digitalWrite(LED_VERDE, LOW);
  digitalWrite(LED_VERMELHO, LOW);
  Serial.println("TOTEM_PRONTO");
}
void loop() {
  if (!leitor.PICC_IsNewCardPresent()) return;
  if (!leitor.PICC_ReadCardSerial()) return;
  String uid = "";
  for (byte i = 0; i < leitor.uid.size; i++) {
    if (leitor.uid.uidByte[i] < 16) uid += "0";
    uid += String(leitor.uid.uidByte[i], HEX);
  }
  uid.toUpperCase();
  if (uid == cartao1 || uid == cartao2) {
    Serial.print("CARTAO:");
    Serial.println(uid);
    Serial.println("VERIFICACAO_FACIAL_PENDENTE");
  } else {
    Serial.println("CARTAO_DESCONHECIDO");
    erro();
  }
  leitor.PICC_HaltA();
  leitor.PCD_StopCrypto1();
  delay(700);
}
