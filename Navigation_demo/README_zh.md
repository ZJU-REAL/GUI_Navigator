
# Navigation_demo


<div align="center">
<a href="README_zh.md">简体中文</a> | <a href="README.md">English</a>
<hr>
</div>
真正的 GUI 智能体，核心在于能像人一样执行多步骤的复杂任务，而不仅仅是识别单个按钮。只专注于元素定位会大大限制其潜力。

为此，我们基于开源的 Mobile-Agent-V3 ，成功跑通并测试了其核心的 Navigation 能力。我们让它完成了诸如在淘宝内深层跳转、应用间交互等需要多步推理和操作的任务，展示了一个真正GUI智能体应有的连贯执行力。

### Make Video功能

为方便开发者进行工作宣传，我们在 Mobile-Agent-V3 的源代码的基础上增加了自动录制手机端 Navigation 寻路视频的功能。

- ```run_mobileagentv3.py``` 中的 run_instruction 现已接受 screenrecord 标志，在每次运行结果目录下创建 recordings 文件夹并立即拉起控制器的录屏；save_task_result 会在写入 task_result.json 时同时记录 MP4 路径，使用 try/finally 包裹整段迭代逻辑并确保无论何种退出场景都能调用controller.stop_recording()；CLI 则新增 --screenrecord 开关并传递给运行函数。
- ```MobileAgent/Mobile-Agent-v3/mobile_v3/utils/controller.py``` 中为所有控制器定义了 start_recording/stop_recording 抽象接口，便于在不同设备上复用统一录屏流程。
- ```utils/android_controller.py``` 与 ```utils/harmonyos_controller.py``` 中分别维护录屏子进程句柄与输出路径：删除旧文件、启动 screenrecord 并创建输出目录，通过 pkill -SIGINT screenrecord 停止录制、等待子进程退出，并使用 adb pull/hdc file recv 拉取 ```screenrecord.mp4``` 再清空内部状态。

## 在你的手机上部署Mobile-Agent-v3
❗目前仅安卓和鸿蒙系统支持工具调试。其他系统如iOS暂时不支持使用Mobile-Agent。


### 安装 qwen 模型所需的依赖项
```
pip install qwen_agent
pip install qwen_vl_utils
pip install numpy
```

### 准备通过ADB连接你的移动设备

#### 下载ADB

ADB 全称为 Android Debug Bridge，起到**调试桥的作用**，是一个客户端-服务器端程序。 其中客户端是用来操作的电脑，服务端是 Android 设备。 ADB 也是 Android SDK 中的一个工具，可以直接操作管理Android模拟器或者真实的Android设备。

下载 [Android Debug Bridge](https://developer.android.com/tools/releases/platform-tools?hl=en)（ADB），即 SDK-Platform-Tools，里面自动包含了ADB应用程序，根据系统选择合适的文件包。

#### 打开USB调试

对于一般的Android移动设备，找到设置-关于手机-系统详细参数界面，连续点击7次~10次版本号栏，就能收到“您已处于开发者模式”的提示，此时你可以在系统的高级设置或其他设置界面中就能找到“开发者选项”功能的入口，即可开启“USB调试”或“ADB调试”，如果是HyperOS系统需要同时打开 "[USB调试(安全设置)](https://github.com/user-attachments/assets/05658b3b-4e00-43f0-87be-400f0ef47736)"。

#### 连接设备

通过数据线连接移动设备和电脑，同时在手机的连接选项中选择“传输文件”。

你可以在终端通过下面的命令来测试你的连接是否成功: ```/path/to/adb devices```。如果输出的结果显示你的设备列表不为空，则说明连接成功。

#### 注意事项

1. 如果你是用的是MacOS或者Linux，请先为 ADB 开启权限: ```sudo chmod +x /path/to/adb```。
2. ```/path/to/adb```在Windows电脑上将是```xx/xx/adb.exe```的文件格式，而在MacOS或者Linux则是```xx/xx/adb```的文件格式。

### 在你的移动设备上安装 ADB 键盘

ADBKeyBoard 是一款专为 Android 自动化测试设计的虚拟键盘工具，通过 ADB 命令实现文本输入功能。该工具解决了 Android 系统内置 input 命令无法发送Unicode 字符的痛点，让中文输入和特殊字符处理变得简单高效。

下载 ADB 键盘的 [apk](https://github.com/senzhk/ADBKeyBoard/blob/master/ADBKeyboard.apk)  安装包，在设备上点击该 apk 来安装，在系统设置中将默认输入法切换为 “ADB Keyboard” 。如果在手机下边栏看到 “ADB Keyboard(on)” 则说明已成功启动 ADB 键盘。

## 运行

#### 安卓
```
cd Mobile-Agent-v3/mobile_v3
python run_mobileagentv3.py \
    --adb_path "Your ADB path" \
    --api_key "Your api key of vllm service" \
    --base_url "Your base url of vllm service" \
    --model "Your model name of vllm service" \
    --instruction "The instruction you want Mobile-Agent-v3 to complete" \
    --add_info "Some supplementary knowledge, can also be empty"
    --
```

#### 鸿蒙
```
cd Mobile-Agent-v3/mobile_v3
python run_mobileagentv3.py \
    --hdc_path "Your HDC path" \
    --api_key "Your api key of vllm service" \
    --base_url "Your base url of vllm service" \
    --model "Your model name of vllm service" \
    --instruction "The instruction you want Mobile-Agent-v3 to complete" \
    --add_info "Some supplementary knowledge, can also be empty"
```

### 注意
1. 如果您使用的模型输出的是 0 到 1000 的相对坐标，例如 Seed-VL、Qwen-VL-2 或 Qwen-VL-3，请设置：
```
--coor_type "qwen-vl" # 这意味着 0-1000 的坐标会映射到实际设备分辨率。
```

注意：如果您使用的模型输出的是绝对坐标，例如 Qwen-VL-2.5 或 GUI-Owl，请不要设置坐标映射。

​	2.如果您的指令需要记忆某些页面中内容，请设置：

```
--notetaker True
```

​	3.如果您需要开启自动录制手机端Navigation寻路的视频，请设置：

```
--screenrecord --recording_dir "E:\captures"
```

注意：--recording_dir 后的目录位置可以根据自己的需要更改。

## 测试实例

| 任务指令                                                     | 完成情况 | 额外说明                                                     |
| :----------------------------------------------------------- | :------: | ------------------------------------------------------------ |
| 登录智行app并查看订单                                        |   成功   | 无                                                           |
| 进入系统设置，将屏幕亮度调整为50%                            |   成功   | 无                                                           |
| 在淘宝中搜索"无线鼠标"，按销量排序，查看第一个商品详情      并加入购物车 |   成功   | 尝试如果首页面不存在淘宝时，需要进行屏幕翻页行动，Agent进行右翻页动作，但正确为左翻页，在翻到最右页面后陷入死循环失败 |
| 打开网易云音乐，搜索徐佳莹的到此为止，播放《到此为止》并添加到我喜欢 |   成功   | 无                                                           |
| 在美团外卖中定位到当前位置，选择一家餐厅，将任意商品加入购物车并进入结算页面 |   成功   | 无                                                           |
| 在百度地图中搜索从"北京西站"到"故宫"的公交路线，选择第一个方案 |   成功   | 无                                                           |
| 打开计算器，计算(15+28)×3的结果，然后清空重新计算365÷12      |   失败   | 由于未知原因在输入完操作符后Agent总会删除操作符前的数字导致输入错误 |
| 在通讯录中添加一个新联系人，姓名为"张三测试"，手机号码"13800138000"，并设置为收藏联系人 |   成功   | 无                                                           |
| 在备忘录中新建一条笔记，标题"购物清单"，内容包含"牛奶、面包、鸡蛋"，然后置顶该笔记 |   失败   | 流程基本正确，但在需要标题栏和正文栏分别输入时出现错误，将购物清单写入了正文栏 |
| 在系统日历中创建明天下午2点的会议日程，设置提醒为提前30分钟，添加地点和备注 |   成功   | 无                                                           |

## 演示视频

在百度地图中搜索从"北京西站"到"故宫"的公交路线，选择第一个方案

<video width="500" height="340" controls>
    <source src="./Navigation_video1.mp4" type="video/mp4">
</video>



打开网易云音乐，搜索徐佳莹的到此为止，播放《到此为止》并添加到我喜欢

<video width="500" height="340" controls>
    <source src="./Navigation_video2.mp4" type="video/mp4">
</video>

## 部署

请参考具体模型的README以获得最佳性能。

## 致谢与引用

本项目基于 Mobile-Agent-v3 & GUI-Owl 系列模型进行二次开发，感谢原作者开源高质量代码与模型。

如果该 Demo 或 Mobile-Agent-v3 对你的工作有帮助，欢迎引用官方论文：

```
@article{ye2025mobile,
  title={Mobile-Agent-v3: Foundamental Agents for GUI Automation},
  author={Ye, Jiabo and Zhang, Xi and Xu, Haiyang and Liu, Haowei and Wang, Junyang and Zhu, Zhaoqing and Zheng, Ziwei and Gao, Feiyu and Cao, Junjie and Lu, Zhengxi and others},
  journal={arXiv preprint arXiv:2508.15144},
  year={2025}
}
```

