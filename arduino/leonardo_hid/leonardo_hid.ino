/*
 * leonardo_hid.ino  -  USB serial -> USB keyboard + mouse bridge for Minecraft Biogaming
 * ======================================================================================
 *
 * Board: Arduino Leonardo (or Micro / Pro Micro - any ATmega32u4 board, they can act as
 * a real USB keyboard and mouse). In the Arduino IDE choose
 *   Tools > Board > Arduino Leonardo,  Tools > Port > /dev/cu.usbmodem...  then Upload.
 *
 * The Python program (mcbio.outputs.SerialLeonardoOutput) sends one ASCII command per
 * line, ending in '\n':
 *
 *   K+<key>      press a key        e.g.  K+w   K+space   K+shift
 *   K-<key>      release a key      e.g.  K-w
 *                <key> is a single character (w, a, s, d, e, 1 ...) or one of the names
 *                space shift ctrl alt cmd tab enter esc backspace up down left right f1..f12
 *   M <dx> <dy>  move the mouse by dx, dy pixels (each -127..127; + dy = down)
 *   B+L / B-L    left mouse button down / up      (attack / mine)
 *   B+R / B-R    right mouse button down / up     (use / place)
 *   R            release every key and button
 *   P            ping: the board answers "OK"
 *
 * SAFETY WATCHDOG: if no byte at all arrives for 500 ms (Python crashed, cable pulled,
 * laptop asleep) everything is released, so your character never keeps running or
 * mining on its own. The Python side sends "P" every 200 ms when it has nothing else
 * to say, which keeps the watchdog happy.
 *
 * Tip: if a sketch ever spams keys and makes re-uploading hard, hold the board's reset
 * button, click Upload, and release reset when the IDE says "Uploading...".
 */

#include <Keyboard.h>
#include <Mouse.h>

const unsigned long BAUD = 115200;       // ignored by USB CDC serial, must match Python anyway
const unsigned long WATCHDOG_MS = 500;   // release everything after this much silence
const uint8_t LINE_MAX = 24;             // longest command we accept ("M -127 -127" = 11)

char line[LINE_MAX + 1];                 // current, partially received command
uint8_t lineLen = 0;
bool lineOverflow = false;               // true -> discard the rest of an over-long line
unsigned long lastByteMs = 0;            // millis() when the last byte arrived
bool anythingHeld = false;               // so the watchdog only acts when needed

// ---------------------------------------------------------------------------------------
// Release every key and both mouse buttons.
void releaseAll() {
  Keyboard.releaseAll();
  Mouse.release(MOUSE_LEFT);
  Mouse.release(MOUSE_RIGHT);
  anythingHeld = false;
}

// Translate a key token from the protocol into the code Keyboard.press() expects.
// Returns 0 for an unknown name (the command is then ignored).
uint8_t keyCode(const char *name) {
  if (name[0] != '\0' && name[1] == '\0') return (uint8_t)name[0];   // single character
  if (strcmp(name, "space") == 0)     return ' ';
  if (strcmp(name, "shift") == 0)     return KEY_LEFT_SHIFT;
  if (strcmp(name, "ctrl") == 0)      return KEY_LEFT_CTRL;
  if (strcmp(name, "alt") == 0)       return KEY_LEFT_ALT;     // Option on a Mac
  if (strcmp(name, "cmd") == 0)       return KEY_LEFT_GUI;     // Command on a Mac
  if (strcmp(name, "tab") == 0)       return KEY_TAB;
  if (strcmp(name, "enter") == 0)     return KEY_RETURN;
  if (strcmp(name, "esc") == 0)       return KEY_ESC;
  if (strcmp(name, "backspace") == 0) return KEY_BACKSPACE;
  if (strcmp(name, "up") == 0)        return KEY_UP_ARROW;
  if (strcmp(name, "down") == 0)      return KEY_DOWN_ARROW;
  if (strcmp(name, "left") == 0)      return KEY_LEFT_ARROW;
  if (strcmp(name, "right") == 0)     return KEY_RIGHT_ARROW;
  if (name[0] == 'f') {                                       // f1 .. f12
    int n = atoi(name + 1);
    if (n >= 1 && n <= 12) return KEY_F1 + (n - 1);          // KEY_F1..KEY_F12 are consecutive
  }
  return 0;
}

// Clamp to the signed-byte range a HID mouse report can carry.
int8_t clampStep(long v) {
  if (v > 127) return 127;
  if (v < -127) return -127;
  return (int8_t)v;
}

// Execute one complete command line (without the trailing newline).
void handleLine(char *cmd) {
  switch (cmd[0]) {
    case 'K': {                       // K+<key> / K-<key>
      uint8_t code = keyCode(cmd + 2);
      if (code == 0) return;
      if (cmd[1] == '+') { Keyboard.press(code); anythingHeld = true; }
      else if (cmd[1] == '-') Keyboard.release(code);
      break;
    }
    case 'M': {                       // M <dx> <dy>
      char *end;
      long dx = strtol(cmd + 1, &end, 10);
      long dy = strtol(end, NULL, 10);
      Mouse.move(clampStep(dx), clampStep(dy), 0);
      break;
    }
    case 'B': {                       // B+L / B-L / B+R / B-R
      uint8_t button = (cmd[2] == 'R') ? MOUSE_RIGHT : MOUSE_LEFT;
      if (cmd[2] != 'L' && cmd[2] != 'R') return;
      if (cmd[1] == '+') { Mouse.press(button); anythingHeld = true; }
      else if (cmd[1] == '-') Mouse.release(button);
      break;
    }
    case 'R':                         // release all
      releaseAll();
      break;
    case 'P':                         // ping
      Serial.println("OK");
      break;
    default:                          // unknown command: ignore
      break;
  }
}

void setup() {
  Serial.begin(BAUD);
  Keyboard.begin();
  Mouse.begin();
  releaseAll();
  lastByteMs = millis();
}

void loop() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    lastByteMs = millis();
    if (c == '\r') continue;                  // tolerate Windows-style line endings
    if (c == '\n') {
      if (!lineOverflow && lineLen > 0) {
        line[lineLen] = '\0';
        handleLine(line);
      }
      lineLen = 0;
      lineOverflow = false;
    } else if (lineLen < LINE_MAX) {
      line[lineLen++] = c;
    } else {
      lineOverflow = true;                    // too long: drop this whole line
    }
  }

  // Watchdog: the host has gone quiet -> let go of everything.
  if (anythingHeld && millis() - lastByteMs > WATCHDOG_MS) {
    releaseAll();
  }
}
