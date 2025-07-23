
  #include <Wire.h>

  #include <ESP8266WiFi.h>

  #include <PubSubClient.h>  // 添加MQTT客户端库
 #include <WiFiClientSecure.h>
   

  const char* ssid = "";

  const char* password = " ";


  const char* mqtt_server = "me90f19a.ala.cn-hangzhou.emqxsl.cn";  // MQTT服务器地址

  const int mqtt_port = 8883;                      // MQTT端口

  const char* mqtt_user = "Esp8266";         // MQTT用户名（如果需要）

  const char* mqtt_pass = "Esp8266";         // MQTT密码（如果需要）

  const char* mqtt_topic = "sensors/acceleration";  // 发布主题

   

  WiFiClientSecure espClient;

  PubSubClient mqttClient(espClient);

   

  #define WT931_I2C_ADDR 0x50

  #define GRAVITY 9.80665f

   

  // 指令定义

  const uint8_t UNLOCK_CMD[5] = {0xFF, 0xAA, 0x69, 0x88, 0xB5}; // 解锁指令

  const uint8_t SAVE_CMD[5] = {0xFF, 0xAA, 0x00, 0x00, 0x00};   // 保存指令

   

  struct WT931_Data {

    uint8_t header1;

    uint8_t header2;

    uint8_t AxL, AxH, AyL, AyH, AzL, AzH;

    uint8_t TL, TH;

    uint8_t checksum;

  };

   

  // 发送指令到WT931传感器，并返回发送状态

  bool sendCommand(const uint8_t* cmd, uint8_t len) {

    Wire.beginTransmission(WT931_I2C_ADDR);

    for (uint8_t i = 0; i < len; i++) {

      Wire.write(cmd[i]);

    }

   

    uint8_t error = Wire.endTransmission(true);

    delay(10); // 等待传感器处理指令

   

    if (error == 0) {

      Serial.println("Command sent successfully");

      return true;

    } else {

      Serial.print("Error sending command: ");

      Serial.println(error);

      return false;

    }

  }

   

  // 解锁传感器

  bool unlockSensor() {

    sendCommand(UNLOCK_CMD, 5);

    // 可以添加解锁验证逻辑

    return true;

  }

   

  // 保存传感器设置

  void saveSettings() {

    sendCommand(SAVE_CMD, 5);

  }

   

 void connectMQTT() {
    String clientId = "ESP8266-";
    clientId += String(random(0xffff), HEX);
    int retryCount = 0;
    const int maxRetries = 5;
 
    while (!mqttClient.connected() && retryCount < maxRetries) {
        Serial.print("Attempting MQTT connection...");
        if (mqttClient.connect(clientId.c_str(), mqtt_user, mqtt_pass)) {
            Serial.println("connected");
            mqttClient.subscribe(mqtt_topic);
        } else {
            Serial.print("failed, rc=");
            Serial.print(mqttClient.state());
            retryCount++;
            delay(5000);
        }
    }
    if (retryCount >= maxRetries) {
        Serial.println("Max retries reached. MQTT connection failed.");
    }
}
 
void setup() {
    Serial.begin(115200);
    Wire.begin();
    Wire.setClock(1000);
 
    // 连接 WiFi
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("\nWiFi Connected");
 
    // 测试 MQTT 端口可达性
    Serial.println("Testing MQTT server port...");
    WiFiClient testClient;
    if (!testClient.connect(mqtt_server, mqtt_port)) {
        Serial.println("Error: Cannot connect to MQTT server port!");
    } else {
        Serial.println("MQTT server port is reachable");
        testClient.stop();
    }
 
    // 初始化 MQTT
    mqttClient.setServer(mqtt_server, mqtt_port);
    mqttClient.setCallback([](char* topic, byte* payload, unsigned int length) {
        // ... 回调函数 ...
    });
    espClient.setInsecure();
}
   

  void loop() {
     if (WiFi.status() != WL_CONNECTED) {
        Serial.println("WiFi disconnected! Reconnecting...");
        WiFi.reconnect();
        delay(5000);
        return;
    }
    if (!mqttClient.connected()) {
        connectMQTT();
    }

    static unsigned long lastUnlockTime = 0;

    const unsigned long UNLOCK_INTERVAL = 9000; // 每9秒解锁一次，避免自动上锁

    //scanI2C();

    // 定期解锁，防止自动上锁

    if (millis() - lastUnlockTime > UNLOCK_INTERVAL) {

      unlockSensor();

      lastUnlockTime = millis();

    }

   

    WT931_Data sensorData;

    float accX = 0.0f, accY = 0.0f, accZ = 0.0f, temperature = 0.0f; // 声明在loop作用域内

   

    // 发送命令0x51请求加速度和温度数据

    Wire.beginTransmission(WT931_I2C_ADDR);

    Wire.write(0x51);

    Wire.endTransmission(false); // 保持总线连接

   

    // 请求读取11字节数据

    Wire.requestFrom(WT931_I2C_ADDR, 11);

    if (Wire.available() >= 11) {

      // 读取所有数据

      for (uint8_t i = 0; i < 11; i++) {

        ((uint8_t*)&sensorData)[i] = Wire.read();

      }

   

      // 校验数据包头

//      if (sensorData.header1 != 0x55 || sensorData.header2 != 0x51) {
//
//        Serial.println("Error: Invalid header");
//
//        return; // 跳过本次循环，避免发布错误数据
//
//      }

   

      // 计算校验和

      uint8_t sum = 0x55 + 0x51;

      sum += sensorData.AxL + sensorData.AxH;

      sum += sensorData.AyL + sensorData.AyH;

      sum += sensorData.AzL + sensorData.AzH;

      sum += sensorData.TL + sensorData.TH;

   

//      if (sum != sensorData.checksum) {
//
//        Serial.println("Error: Checksum mismatch");
//
//        return; // 跳过本次循环，避免发布错误数据
//
//      }

   

      // 解析加速度

      accX = ((int16_t)(sensorData.AxH << 8) | sensorData.AxL) / 32768.0f * 16 * GRAVITY;

      accY = ((int16_t)(sensorData.AyH << 8) | sensorData.AyL) / 32768.0f * 16 * GRAVITY;

      accZ = ((int16_t)(sensorData.AzH << 8) | sensorData.AzL) / 32768.0f * 16 * GRAVITY;

   

      // 解析温度

      temperature = ((int16_t)(sensorData.TH << 8) | sensorData.TL) / 100.0f;

   

      // 打印结果

      Serial.print("AccX: "); Serial.print(accX); Serial.print(" m/s² | ");

      Serial.print("AccY: "); Serial.print(accY); Serial.print(" m/s² | ");

      Serial.print("AccZ: "); Serial.print(accZ); Serial.println(" m/s²");

      Serial.print("Temperature: "); Serial.print(temperature); Serial.println(" ℃");

      Serial.println("-------------------");

    } else {

      Serial.println("Error: Incomplete sensor data");

      return; // 跳过本次循环，避免发布错误数据

    }

   

    if (!mqttClient.connected()) {

      connectMQTT();  // 确保MQTT连接

    }

   

    String jsonData = "{\"accX\":";

    jsonData += accX;

    jsonData += ",\"accY\":";

    jsonData += accY;

    jsonData += ",\"accZ\":";

    jsonData += accZ;

    jsonData += ",\"temp\":";

    jsonData += temperature;

    jsonData += "}";

   

    if (mqttClient.publish(mqtt_topic, jsonData.c_str())) {

      Serial.println("Data published to MQTT");

    } else {

      Serial.println("MQTT publish failed");

    }

   

    delay(20);  // 50Hz采样率

  }
