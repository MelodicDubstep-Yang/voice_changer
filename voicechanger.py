import customtkinter as ctk
import sounddevice as sd
import numpy as np
from typing import Callable
import librosa.effects


ctk.set_appearance_mode('light')
ctk.set_default_color_theme('blue')

# 创建主窗口

class VoiceChanger(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("简易变声器")
        self.geometry('600x390')
        self.resizable(False, False)
        self.user_name = ctk.StringVar(value='未知用户')
        # 使窗口始终置顶
        # root.attributes("-topmost", True)

        self.samplerate = 48000
        self.recording = False

        # 指定设备
        # 22 麦克风阵列 (适用于数字麦克风的英特尔® 智音技术), Windows WASAPI (4 in, 0 out)
        self.input_device_index = 22
        # 21 CABLE Input (VB-Audio Virtual Cable), Windows WASAPI (0 in, 2 out)
        self.output_device_index = 21

        # 音效属性
        self.robot_phase = 0.0
        self.ghost_delayed_sample = int(self.samplerate * 0.28)  # 或者取得frame的倍数，如12288？
        self.ghost_delay_buffer = np.zeros(self.ghost_delayed_sample, dtype=np.float32)
        self.ghost_lowpass_last_status = 0.0
        self.loli_bright_effect_last_status = 0.0

        # 颜色
        self.colors = {
            # 基础背景
            "bg": "#F3F4F6",
            "frame": "#FFFFFF",
            "frame_2": "#E5E7EB",
            "border": "#D1D5DB",

            # 主色：蓝紫
            "primary": "#93C5FD",
            "primary_hover": "#60A5FA",

            # segmented button 未选中
            "unselected": "#E5E7EB",
            "unselected_hover": "#D1D5DB",

            # 文字
            "text": "#111827",
            "subtext": "#4B5563",
            "text_on_primary": "#FFFFFF",

            # 开始/停止按钮
            "start": "#10B981",
            "start_hover": "#059669",
            "stop": "#EF4444",
            "stop_hover": "#DC2626",

            # 用户栏
            "user_frame": "#FFFFFF",
            "edit_button": "#6B7280",
            "edit_button_hover": "#4B5563",
        }
        # 设置窗口背景色
        self.configure(fg_color=self.colors["bg"])

        self._build_ui()


    # ------ ui界面 -------

    def _build_ui(self):
        """ui界面布局"""
        # 字体
        self.title_font = ctk.CTkFont('新宋体', size=27, weight='bold')
        self.text_font = ctk.CTkFont('新宋体', size=20)


        # 标题与整体框架
        self.title_label = ctk.CTkLabel(
            self,
            text="简易变声器🔈",
            font=self.title_font,
            text_color=self.colors["text"]
        )
        self.title_label.pack(padx = 10, pady = 10)

        self.frame = ctk.CTkFrame(
            self,
            width=450,
            height=260,
            corner_radius=20,
            fg_color=self.colors["frame"],
            border_color=self.colors["border"],
            border_width=1
        )
        self.frame.pack(padx=(20, 20), pady=30, fill="both", expand=False)

        # 用户栏
        self.user_frame = ctk.CTkFrame(
            self,
            height=48,
            fg_color=self.colors["user_frame"],
            border_color=self.colors["border"],
            border_width=1,
            corner_radius=0
        )
        self.user_frame.pack(side="bottom", fill="x", padx=0, pady=0)

        self.user_frame.grid_columnconfigure(0, weight=1)
        self.user_frame.grid_columnconfigure(1, weight=0)

        self.user_name_label = ctk.CTkLabel(
            self.user_frame,
            textvariable=self.user_name,
            height=40,
            anchor="w",
            text_color=self.colors["text"]
        )
        self.user_name_label.grid(row=0, column=0, padx=15, pady=4, sticky="ew")

        self.edit_name_button = ctk.CTkButton(
            self.user_frame,
            text='修改用户名',
            fg_color=self.colors["edit_button"],
            hover_color=self.colors["edit_button_hover"],
            text_color=self.colors["text_on_primary"],
            width=90,
            height=32,
            command=self.set_user_name
        )
        self.edit_name_button.grid(row=0, column=1, padx=(0, 15), pady=4)

        # 变声模式选择
        self.mode = ctk.StringVar(value='原声')

        self.mode_button = ctk.CTkSegmentedButton(
            self.frame,
            values=['原声', '幽灵', '机器人', '小女孩'],
            variable=self.mode,
            fg_color=self.colors["unselected"],
            selected_color=self.colors["primary"],
            selected_hover_color=self.colors["primary_hover"],
            unselected_color=self.colors["unselected"],
            unselected_hover_color=self.colors["unselected_hover"],
            text_color=self.colors["text"],
            width=320,
            height=45,
            corner_radius=14,
            font=self.text_font
        )
        self.mode_button.pack(padx=30, pady=(35, 20))

        # 变声开启/关闭状态与按钮
        self.status_label = ctk.CTkLabel(
            self.frame,
            text="变声未开始",
            font=self.text_font,
            text_color=self.colors["subtext"]
        )
        self.status_label.pack(padx=20, pady=10)

        self.rec_button = ctk.CTkButton(
            self.frame,
            text="点击开始变声",
            height=50,
            corner_radius=15,
            font=self.text_font,
            fg_color=self.colors["start"],
            hover_color=self.colors["start_hover"],
            text_color=self.colors["text_on_primary"],
            command=self.toggle_recording
        )
        self.rec_button.pack(padx=60, pady=10, fill="x")
        # self.rec_button.bind('<Button-1>', self.toggle_recording)
        # 修改此处不再使用bind，因为bind会自动传入事件，而该程序未给


    # ------ 播放音频处理 ------

    def callback(self, indata, outdata, frames, time, status):
        """回调函数，sounddevice自动调用，进行实时音频处理"""
        if status:
            print(status)
        # 清空outdata，防止有未输出的音频而出现杂音
        # 取左声道处理后复制到左右声道，比两声道处理再合并更稳定;* 0.7,降声音防止爆炸
        x_l = indata[:, 0].copy() * 0.7
        effect_func = self.effect_mode()
        y = effect_func(x_l, frames)
        # 软削波
        y = np.tanh(y * 1.2) * 0.75
        y = y.astype(np.float32)

        outdata[:, 0], outdata[:, 1] = y, y

    # 加入了类型标注指明为可调用函数，消除IDE报错提示(x_l, frames)为意外实参
    def effect_mode(self) -> Callable[[np.ndarray, int], np.ndarray]:
        """获取变音效果模式"""
        audio_effect = {'原声': self.original_voice, '机器人': self.robot_effect, '幽灵': self.ghost_effect, '小女孩': self.loli_effect}
        get_effect_mode = audio_effect.get(self.mode.get(), self.original_voice)
        return get_effect_mode


    # ------ 播放与暂停设置 ------

    def toggle_recording(self):
        if self.recording:
            self.end_recording()
        else:
            self.start_recording()

    def start_recording(self):
        """开始变音"""
        # 处于True状态时，不执行start操作
        if self.recording:
            return

        self.stream = sd.Stream(samplerate=self.samplerate,
                                blocksize=2048,
                                channels=(4, 2),
                                dtype='float32',
                                callback=self.callback,
                                device=(self.input_device_index, self.output_device_index)
                                )
        self.stream.start()
        self.recording = True
        self.rec_button.configure(
            text='点击停止变声',
            fg_color=self.colors["stop"],
            hover_color=self.colors["stop_hover"]
        )
        self.status_label.configure(text='变声ing……')

    def end_recording(self):
        """结束变音"""
        # 处于False状态时，不执行start操作
        if not self.recording:
            return
        self.recording = False

        # 安全条件：该属性存在时再执行“关闭”操作
        if hasattr(self, 'stream') and self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.rec_button.configure(
            text="点击开始变声",
            fg_color=self.colors["start"],
            hover_color=self.colors["start_hover"]
        )
        self.after(0, self._reset_ui)

    def _reset_ui(self):
        """回复gui初始状态"""
        self.recording = False
        self.status_label.configure(text="变声未开始")
        self.rec_button.configure(text="点击开始变声")


    # ------ 原声效果 -------
    def original_voice(self, x, frame):
        return x


    # ------ 机器人效果 -------
    def robot_effect(self, x, frame):
        """机器人音效实现"""
        # 载波频率实现抖动的快慢
        carrier_freq = 85

        # 每经过一个采样点，相位增加多少，所以要用 w / sr
        # 一秒钟转了2pi * carrier_freq圈，所以没经过一个相位点会多走多长的弧，也就是相位
        phase_incre = 2 * np.pi * carrier_freq / self.samplerate
        phase = self.robot_phase + phase_incre * np.arange(0, frame)

        # 先生成正弦波，然后sign把正弦波变成只有1，-1的方波，乘法调制让原声一会正一会负，产生抖动感
        carrier = np.sign(np.sin(phase))

        # 因为sin的相位是0-2pi，也就是只有0-2pi间的相位是有特异性的；
        # 这里取这一秒走完后所停留在哪一处相位，先移动到下一秒开始的位置，然后除以完整周期来取余数，最后robot_phase仍然落在0-2pi范围内
        # 防止robot_phase无限增大
        self.robot_phase = (phase[-1] + phase_incre) % (2 * np.pi)

        # 0.8防止音量过大，相当于整体把波形变矮
        robot_voice = x * carrier * 0.75
        bright_robot = self.bright_effect(robot_voice, alpha=0.1, amount=0.1)

        return bright_robot


    # ------- 实现幽灵音效 --------

    def naive_pitch_down(self, x, pitch_factor = 0.65):
        """简易降调效果"""
        x_len = len(x)
        x_loc = np.arange(x_len)
        new_sample_loc = x_loc * pitch_factor
        # 此处理论上不可能越界，但是保险起见使用硬削波
        new_sample_loc = np.clip(new_sample_loc, 0, x_len - 1)
        y = np.interp(new_sample_loc, x_loc, x).astype(np.float32)

        return y

    def lowpass(self, x, alpha = 0.1):
        """低通滤波器"""
        # alpha: 平滑系数，越小声音越暗
        last = self.ghost_lowpass_last_status
        y = np.empty_like(x)

        # 过去的声音总是会以 (0.92)**n 的占比影响当下的声音
        for i, sample in enumerate(x):
            last = sample * alpha + last * (1 - alpha)
            y[i] = last

        self.ghost_lowpass_last_status = last

        return y.astype(np.float32)

    def echo(self, x, frame):
        """回声部分"""
        # 回声：0.45降低音量
        delayed = 0.3 * self.ghost_delay_buffer[:frame].copy()

        # 该实现类似栈：把缓冲最前面那个frame挤出去，把新frame添加到buffer尾部，在滚到头的时候输出并被roll挤出
        self.ghost_delay_buffer[:-frame] = self.ghost_delay_buffer[frame:]
        # 或：self.ghost_delay_buffer = np.roll(self.ghost_delay_buffer, -frame)

        self.ghost_delay_buffer[-frame:] = x

        return delayed

    def ghost_effect(self, x, frame):
        """幽灵音效实现"""
        pitch_down = self.naive_pitch_down(x)
        dark = self.lowpass(pitch_down)
        echo = self.echo(dark, frame)
        ghost_voice = dark * 0.7 + echo * 0.3

        # 软削波，1.4控制失真程度，0.8控制最终音量大小
        ghost_voice = np.tanh(ghost_voice * 1.1) * 0.7

        return ghost_voice


    # ------- 实现小女孩音效 --------

    def naive_pitch_up(self, x, pitch_factor = 1.1):
        """简易升调效果"""
        x_len = len(x)

        x_loc = np.arange(x_len)
        new_sample_loc = x_loc * pitch_factor
        new_sample_loc = np.clip(new_sample_loc, 0, x_len - 1)

        y = np.interp(new_sample_loc, x_loc, x).astype(np.float32)

        return y

    def bright_effect(self, x, alpha=0.15, amount=0.4):
        """明亮效果：先计算出低通部分，然后得到高频部分并增强"""
        low_last = self.loli_bright_effect_last_status
        y = np.empty_like(x)
        for i, sample in enumerate(x):
            low_last = alpha * sample + (1 - alpha) * low_last
            high = sample - low_last
            y[i] = sample + amount * high
        self.loli_bright_effect_last_status = low_last

        return y.astype(np.float32)

    def loli_effect(self, x, frame):
        pitch_up = self.naive_pitch_up(x)
        bright_pitch_up = self.bright_effect(pitch_up)
        loli_voice = np.tanh(bright_pitch_up * 1.1) * 0.6

        return loli_voice


    # ------- 用户设置 -------

    def set_user_name(self):
        """设置用户名"""
        dialog = ctk.CTkInputDialog(text = '起个名字：', title = '用户登录')
        new_name = dialog.get_input()
        if new_name is not None and new_name.strip():
            self.user_name.set(value=new_name)

# 运行
if __name__ == "__main__":
    app = VoiceChanger()
    app.mainloop()
