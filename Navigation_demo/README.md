# Navigation_demo

<div align="center">
<a href="README_zh.md">简体中文</a> | <a href="README.md">English</a>
<hr>
</div>

A true GUI agent's core capability lies in executing multi-step complex tasks like humans, rather than merely recognizing individual buttons. Focusing solely on element grounding significantly limits its potential.

To this end, we have successfully implemented and tested the core navigation capabilities based on the open-source Mobile-Agent-V3. We enabled it to complete tasks requiring multi-step reasoning and operations, such as deep navigation within Taobao and cross-application interactions, demonstrating the coherent execution ability that a true GUI agent should possess.

### Make Video Feature

To facilitate developers in promoting their work, we have added an automatic screen recording feature for mobile Navigation tasks based on the Mobile-Agent-V3 source code.

- `run_mobileagentv3.py`: The `run_instruction` function now accepts a `screenrecord` flag. It creates a `recordings` folder in each run's result directory and immediately starts the controller's screen recording. `save_task_result` records the MP4 path when writing `task_result.json`. The entire iteration logic is wrapped with `try/finally` to ensure `controller.stop_recording()` is called regardless of the exit scenario. The CLI adds a `--screenrecord` switch passed to the run function.
- `MobileAgent/Mobile-Agent-v3/mobile_v3/utils/controller.py`: Defines abstract `start_recording`/`stop_recording` interfaces for all controllers, facilitating unified screen recording processes across different devices.
- `utils/android_controller.py` & `utils/harmonyos_controller.py`: Manage screen recording subprocess handles and output paths respectively: delete old files, start `screenrecord`, create output directories, stop recording via `pkill -SIGINT screenrecord`, wait for the subprocess to exit, and use `adb pull`/`hdc file recv` to retrieve `screenrecord.mp4` before clearing internal state.

## Deploy Mobile-Agent-v3 on Your Phone
❗Currently, only Android and HarmonyOS support tool debugging. Other systems like iOS do not support Mobile-Agent yet.

### Install Dependencies for Qwen Model

```
pip install qwen_agent
pip install qwen_vl_utils
pip install numpy
```

### Prepare Your Mobile Device via ADB Connection

#### Download ADB

ADB, short for Android Debug Bridge, acts as a **debugging bridge**, functioning as a client-server program. The client is the computer used for operation, and the server is the Android device. ADB is also a tool in the Android SDK, allowing direct operation and management of Android emulators or real Android devices.

Download [Android Debug Bridge](https://developer.android.com/tools/releases/platform-tools?hl=en) (ADB), i.e., SDK-Platform-Tools, which includes the ADB application automatically. Choose the appropriate package for your system.

#### Enable USB Debugging

For general Android mobile devices, find Settings > About Phone > Detailed System Parameters, tap the Version Number field 7-10 times until you see the "You are now in developer mode" prompt. You can then find the "Developer Options" entry in the system's Advanced Settings or other settings menu, where you can enable "USB Debugging" or "ADB Debugging". For HyperOS systems, also enable "[USB Debugging (Security Settings)](https://github.com/user-attachments/assets/05658b3b-4e00-43f0-87be-400f0ef47736)".

#### Connect Device

Connect your mobile device to the computer via a USB cable, and select "File Transfer" mode on the phone's connection options.

You can test your connection in the terminal using: `/path/to/adb devices`. If the output shows your device list is not empty, the connection is successful.

#### Notes

1. If using MacOS or Linux, grant permissions to ADB first: `sudo chmod +x /path/to/adb`.
2. `/path/to/adb` on Windows will be in the format `xx/xx/adb.exe`, while on MacOS or Linux it will be `xx/xx/adb`.

### Install ADB Keyboard on Your Mobile Device

ADBKeyBoard is a virtual keyboard tool designed for Android automation testing, enabling text input via ADB commands. It addresses the limitation where the built-in Android `input` command cannot send Unicode characters, making Chinese input and special character handling simple and efficient.

Download the ADB Keyboard [apk](https://github.com/senzhk/ADBKeyBoard/blob/master/ADBKeyboard.apk) installation package, install it by clicking the apk on your device, and switch the default input method to "ADB Keyboard" in system settings. If you see "ADB Keyboard(on)" in the bottom bar, ADB Keyboard has started successfully.

## Run

#### Android

```
cd Mobile-Agent-v3/mobile_v3
python run_mobileagentv3.py
--adb_path "Your ADB path"
--api_key "Your api key of vllm service"
--base_url "Your base url of vllm service"
--model "Your model name of vllm service"
--instruction "The instruction you want Mobile-Agent-v3 to complete"
--add_info "Some supplementary knowledge, can also be empty"
```

#### HarmonyOS

```
cd Mobile-Agent-v3/mobile_v3
python run_mobileagentv3.py
--hdc_path "Your HDC path"
--api_key "Your api key of vllm service"
--base_url "Your base url of vllm service"
--model "Your model name of vllm service"
--instruction "The instruction you want Mobile-Agent-v3 to complete"
--add_info "Some supplementary knowledge, can also be empty"
```

### Notes
1. If the model you are using outputs relative coordinates from 0 to 1000, such as Seed-VL, Qwen-VL-2, or Qwen-VL-3, please set:

```
--coor_type "qwen-vl" # This means 0-1000 coordinates will be mapped to the actual device resolution.
```

Note: If your model outputs absolute coordinates, such as Qwen-VL-2.5 or GUI-Owl, do not set coordinate mapping.

2. If your instruction requires remembering content from certain pages, please set:

```
--notetaker True
```

​	3. If you need to enable automatic recording of mobile Navigation videos, please set:

```
--screenrecord --recording_dir "E:\captures"
```

Note: The directory after `--recording_dir` can be changed according to your needs.

## Test Examples

| Task Instruction                                             | Status  | Additional Notes                                             |
| :----------------------------------------------------------- | :-----: | ------------------------------------------------------------ |
| Log in to the ZhiXing app and check orders                   | Success | -                                                            |
| Enter system settings, adjust screen brightness to 50%       | Success | -                                                            |
| Search for "wireless mouse" in Taobao, sort by sales, view the first product details and add to cart | Success | Attempted right swipe when Taobao wasn't on home screen (should be left), entered failure loop after reaching rightmost page |
| Open NetEase Cloud Music, search for Lala Hsu's "到此为止", play "到此为止" and add to "My Favorites" | Success | -                                                            |
| Set current location in Meituan Waimai, select a restaurant, add any product to cart and proceed to checkout page | Success | -                                                            |
| Search for bus routes from "Beijing West Railway Station" to "Forbidden City" in Baidu Map, select the first option | Success | -                                                            |
| Open Calculator, calculate (15+28)×3, then clear and recalculate 365÷12 | Failure | For unknown reason, Agent deletes numbers before operators after inputting them, causing input errors |
| Add a new contact named "Zhang San Test" with phone "13800138000" to Contacts, and mark as favorite | Success | -                                                            |
| Create a new note titled "Shopping List" with content "milk, bread, eggs" in Memo, then pin it | Failure | Process mostly correct, but wrote "Shopping List" into the content field when separate title/body input was needed |
| Create a meeting schedule for tomorrow 2 PM in System Calendar, set a 30-minute advance reminder, add location and notes | Success | -                                                            |

## Demo Videos

Search for bus routes from "Beijing West Railway Station" to "Forbidden City" in Baidu Map, select the first option

<video width="500" height="340" controls>
    <source src="./Navigation_video1.mp4" type="video/mp4">
</video>

Open NetEase Cloud Music, search for Lala Hsu's "到此为止", play "到此为止" and add to "My Favorites"

<video width="500" height="340" controls>
    <source src="./Navigation_video2.mp4" type="video/mp4">
</video>

## Deployment

Please refer to the specific model's README for optimal performance.

## Acknowledgments & Citation

This project is developed based on Mobile-Agent-v3 & GUI-Owl series models. Thanks to the original authors for open-sourcing high-quality code and models.

If this Demo or Mobile-Agent-v3 is helpful for your work, please cite the official paper:

```
@article{ye2025mobile,
title={Mobile-Agent-v3: Foundational Agents for GUI Automation},
author={Ye, Jiabo and Zhang, Xi and Xu, Haiyang and Liu, Haowei and Wang, Junyang and Zhu, Zhaoqing and Zheng, Ziwei and Gao, Feiyu and Cao, Junjie and Lu, Zhengxi and others},
journal={arXiv preprint arXiv:2508.15144},
year={2025}
}
```

