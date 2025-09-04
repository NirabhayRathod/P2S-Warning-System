# THIS IS MY FIRST SELF CREATED PROJECT OF MACHINE LEARNING , NAME -> P2S-WARNING-SYSTEM



# P2S-WARNING-SYSTEM

## Overview  
Earthquakes don’t happen instantly — they unfold in stages.  
When the Earth’s crust suddenly shifts, two main types of seismic waves are generated:

- **P-Waves (Primary Waves):**  
  These are the fastest seismic waves. They arrive first but are usually not destructive.  
  Think of them as a "gentle tap" from the earthquake — a short warning before the real shaking begins.

- **S-Waves (Secondary Waves):**  
  These arrive after the P-waves. They travel slower but cause the **strong shaking** that can damage buildings and injure people.

The time gap between detecting a **P-wave** and the arrival of an **S-wave** is called **Time to Failure (TTF)**.  
This small window — sometimes just a few seconds — can be enough to:

- Stop trains and elevators  
- Shut down gas lines  
- Alert people to take cover  
- Save lives and reduce damage

---

## What This Project Does  
This project uses machine learning to **predict**:

1. **P-wave Detection (Classification Task)**  
   - Given live seismic sensor readings, determine if a P-wave has been detected.  
   - Output: `p_wave_detected` (0 = No, 1 = Yes)

2. **Time to Failure Prediction (Regression Task)**  
   - If a P-wave is detected, estimate how many seconds remain before the destructive S-wave arrives.  
   - Output: `ttf_seconds` (e.g., 3.5 seconds)

---

## How It Works  
1. **Sensors collect real-time seismic data** — measuring vibration intensity, background noise, signal-to-noise ratio (SNR), etc.  
2. **Step 1 – Classification:** The system checks if the signal matches patterns of a P-wave.  
3. **Step 2 – Regression:**  
   - If **No P-wave** is detected → TTF is set to `0` seconds.  
   - If **Yes**, the system estimates TTF using historical earthquake data patterns.  
4. **Early Warning Alert:** If TTF is above a certain threshold, an alert can be triggered.

---

## Why It Matters  
Even a **5–10 second warning** before an S-wave hits can give people enough time to move to safety, prevent train derailments, or stop critical operations.

This system is designed to be **fast, lightweight, and accurate**, making it useful for both **research** and **real-time applications**.

---

## Dataset Used  
- Features:  
  - `sensor_reading` – vibration intensity  
  - `noise_level` – background noise around the sensor  
  - `rolling_avg` – smoothed average of readings over time  
  - `reading_diff` – change in vibration between readings  
  - `pga` – peak ground acceleration  
  - `snr` – signal-to-noise ratio  

- Targets:  
  - `p_wave_detected` (classification) 
     if p_wave detected then only
  - `ttf_seconds` (regression)

---

## Author  
*Nirabhay Singh Rathod*  
B.Tech CSE – Final Year
Student of Nims University Jaipur Rajasthan . 

